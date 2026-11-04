"""No-network tests for strict bounded random access used by video decoding."""
import io
import unittest

from segmentation_pilot.acquire_frames import ByteBudget, HTTPRangeIO, http_fetcher


class RangeTests(unittest.TestCase):
    def source(self, *, limit=100, block=4, cache=2):
        data, requests = bytes(range(20)), []
        def fetch(start, count):
            requests.append((start, count))
            return 206, f"bytes {start}-{start+count-1}/20", data[start:start+count]
        budget = ByteBudget(limit)
        return HTTPRangeIO(20, fetch, budget, block_size=block, cache_blocks=cache), budget, requests

    def test_seek_read_cache_and_eof(self):
        stream, budget, requests = self.source()
        self.assertEqual(stream.read(6), bytes(range(6)))
        self.assertEqual(budget.used, 8)
        stream.seek(1)
        self.assertEqual(stream.read(2), b"\x01\x02")
        self.assertEqual(len(requests), 2)
        stream.seek(-2, io.SEEK_END)
        self.assertEqual(stream.read(8), b"\x12\x13")
        self.assertEqual(stream.read(4), b"")
        stream.seek(-1, io.SEEK_CUR)
        self.assertEqual(stream.tell(), 19)

    def test_cache_eviction_is_bounded(self):
        stream, budget, requests = self.source(cache=1)
        stream.read(8)
        self.assertEqual(len(stream.cache), 1)
        stream.seek(0)
        stream.read(1)
        self.assertEqual(budget.used, 12)
        self.assertEqual(len(requests), 3)

    def test_budget_prevents_fetch_and_unbounded_reads(self):
        stream, budget, requests = self.source(limit=5)
        stream.read(4)
        with self.assertRaises(RuntimeError):
            stream.read(4)
        self.assertEqual(requests, [(0, 4)])
        with self.assertRaises(RuntimeError):
            stream.read()
        self.assertEqual(budget.used, 4)

    def test_wrong_range_and_truncation_fail(self):
        for response in ((200, "bytes 0-3/20", b"1234"), (206, "bytes 1-4/20", b"1234"),
                         (206, "bytes 0-3/20", b"123")):
            budget = ByteBudget(20)
            stream = HTTPRangeIO(20, lambda *_: response, budget, block_size=4)
            with self.assertRaises(RuntimeError):
                stream.read(1)
            self.assertEqual(stream.tell(), 0)
            self.assertEqual(len(stream.cache), 0)
            self.assertEqual(budget.used, len(response[-1]))

    def test_invalid_seek_and_readinto(self):
        stream, _, _ = self.source()
        for offset, origin in ((-1, 0), (21, 0), (0, 7)):
            with self.assertRaises(ValueError):
                stream.seek(offset, origin)
        target = bytearray(3)
        self.assertEqual(stream.readinto(target), 3)
        self.assertEqual(target, bytes(range(3)))

    def test_http_200_rejected_before_body_and_private_error_redacted(self):
        class Response:
            status_code = 200
            headers = {}
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def iter_raw(self, **kwargs):
                raise AssertionError("Body must never be consumed")
        class Client:
            def stream(self, *args, **kwargs):
                self.kwargs = kwargs
                return Response()
        client = Client()
        fetch = http_fetcher(client, "https://cdn.example/video?secret=private", 20, "private-token")
        with self.assertRaises(RuntimeError) as error:
            fetch(0, 4)
        self.assertNotIn("secret", str(error.exception))
        self.assertNotIn("Authorization", client.kwargs["headers"])

    def test_partial_http_response_counts_wire_bytes_without_caching(self):
        class Response:
            status_code = 206
            headers = {"content-range": "bytes 0-3/20", "content-length": "4"}
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def iter_raw(self, **kwargs):
                yield b"12"
                raise OSError("connection closed at https://private-signed-url")
        class Client:
            def stream(self, *args, **kwargs): return Response()
        budget = ByteBudget(8)
        fetch = http_fetcher(Client(), "https://cdn.example/signed", 20, budget=budget)
        stream = HTTPRangeIO(20, fetch, budget, block_size=4)
        with self.assertRaises(RuntimeError):
            stream.read(1)
        self.assertEqual(budget.used, 2)
        self.assertEqual(len(stream.cache), 0)


if __name__ == "__main__":
    unittest.main()
