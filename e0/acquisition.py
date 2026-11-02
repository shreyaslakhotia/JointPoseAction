"""Bounded ZIP range access. A server ignoring Range must never trigger a download."""
import io
import json
import zipfile


def validate_range_headers(status, content_range, start, count, total):
    if status != 206 or content_range != f"bytes {start}-{start + count - 1}/{total}":
        raise RuntimeError("Server did not honor exact HTTP Range; no full-download fallback")


class RangeFile(io.RawIOBase):
    def __init__(self, size, fetch, max_request=2_000_000, max_total=4_000_000):
        self.size, self.fetch = size, fetch
        self.max_request, self.max_total = max_request, max_total
        self.position, self.transferred = 0, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if whence not in (0, 1, 2) or not 0 <= position <= self.size:
            raise ValueError("Invalid archive seek")
        self.position = position
        return position

    def read(self, size=-1):
        count = min(self.size - self.position, size if size >= 0 else self.size)
        if count > self.max_request or self.transferred + count > self.max_total:
            raise RuntimeError(f"Archive range budget exceeded; requested {count} bytes")
        if not count:
            return b""
        raw = self.fetch(self.position, count)
        if len(raw) != count:
            raise RuntimeError("Truncated range response")
        self.position += count
        self.transferred += count
        return raw


def inspect_archive(stream):
    with zipfile.ZipFile(stream) as archive:
        members = [x for x in archive.infolist() if "/landmarks3D/" in "/" + x.filename and x.filename.endswith(".json")]
        result = {"cleaned_members": [{"path": x.filename, "compressed_bytes": x.compress_size,
                                      "uncompressed_bytes": x.file_size} for x in members]}
        readmes = [x for x in archive.infolist() if "readme" in x.filename.lower() and x.file_size <= 65536 and not x.is_dir()]
        result["release_notes"] = []
        for item in readmes[:3]:
            with archive.open(item) as notes:
                result["release_notes"].append({"path": item.filename, "text": notes.read(65536).decode("utf-8", errors="replace")})
        if not members:
            raise RuntimeError("Documented landmarks3D members absent; no alternate schema substitution")
        member = min(members, key=lambda x: (x.compress_size, x.filename))
        # A prefix only, not an extracted/full recording. ZipExtFile fetches
        # compressed input via the same strict request/transfer budget.
        with archive.open(member) as sample:
            prefix = sample.read(32_768).decode("utf-8-sig")
        decoder = json.JSONDecoder()
        text = prefix.lstrip()
        if not text.startswith("{"):
            raise RuntimeError("Cleaned JSON prefix is not an object")
        position, rows = 1, {}
        for _ in range(3):
            while position < len(text) and text[position] in " \r\n\t,":
                position += 1
            try:
                key, end = decoder.raw_decode(text, position)
                position = end
                while text[position].isspace():
                    position += 1
                if text[position] != ":" or not isinstance(key, str) or not key.isdigit():
                    raise RuntimeError("Unexpected cleaned frame key")
                position += 1
                while text[position].isspace():
                    position += 1
                value, position = decoder.raw_decode(text, position)
            except (json.JSONDecodeError, IndexError):
                break
            rows[key] = value
        if not rows:
            raise RuntimeError("No complete frame in bounded sample prefix")
        result.update(sample_member=member.filename, sample_frames=rows,
                      sample_is_partial=True, full_member_crc_verified=False)
        return result
