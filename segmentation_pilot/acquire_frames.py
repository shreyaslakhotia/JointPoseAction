"""Bounded original-video survey. Native frames are not synchronized pilots.

No complete video is cached. HTTP 200/range mismatches are fatal for a view;
there is no full-download fallback. Signed locations and access tokens never
enter the report or exception text. This survey cannot authorize segmentation.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from segmentation_pilot.calibration import CAMERA_IDS, TOOLKIT_REVISION, load_nimble_calibration

DATASET_REVISION = "bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967"
NETWORK_LIMIT = 805_306_368
OUTPUT_LIMIT = 536_870_912
ALLOCATION_LIMIT = 50_000_000_000
HEADROOM_BYTES = 2_000_000_000


class ByteBudget:
    def __init__(self, limit):
        if limit <= 0:
            raise ValueError("Positive byte budget required")
        self.limit, self.used = int(limit), 0

    def check(self, count):
        if count < 0 or count > self.limit - self.used:
            raise RuntimeError("Explicit survey byte budget exhausted")

    def consume(self, count):
        self.check(count)
        self.used += count


class HTTPRangeIO(io.RawIOBase):
    """Seekable bounded block reader with exact-range checks and small LRU.

    fetch(start, count) returns (status, content_range, raw_bytes). Production
    fetch checks headers before consuming response data and counts every byte;
    the reader separately counts successful exact payloads for testable logic.
    """
    def __init__(self, size, fetch, budget, *, block_size=512*1024, cache_blocks=8):
        super().__init__()
        if size <= 0 or block_size <= 0 or cache_blocks < 1:
            raise ValueError("Positive file/block/cache sizes required")
        self.size, self.fetch, self.budget = int(size), fetch, budget
        self.block_size, self.cache_blocks = int(block_size), int(cache_blocks)
        self.position, self.transferred = 0, 0
        self.cache = OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek origin")
        position = int(offset) + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if not 0 <= position <= self.size:
            raise ValueError("Seek outside immutable video")
        self.position = position
        return position

    def _block(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index*self.block_size
        count = min(self.block_size, self.size-start)
        self.budget.check(count)
        status, content_range, payload = self.fetch(start, count)
        # Count received bytes even if invalid; never retain an invalid block.
        if not getattr(self.fetch, "manages_budget", False):
            self.budget.consume(len(payload))
        self.transferred += len(payload)
        if status != 206 or content_range != f"bytes {start}-{start+count-1}/{self.size}":
            raise RuntimeError("Server did not honor exact HTTP Range")
        if len(payload) != count:
            raise RuntimeError("Truncated HTTP range")
        self.cache[index] = payload
        while len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return payload

    def read(self, size=-1):
        if size is None or size < 0:
            raise RuntimeError("Unbounded reads are disabled for video surveys")
        count = min(int(size), self.size-self.position)
        if count <= 0:
            return b""
        chunks = []
        end = self.position+count
        while self.position < end:
            index, offset = divmod(self.position, self.block_size)
            block = self._block(index)
            length = min(len(block)-offset, end-self.position)
            chunks.append(block[offset:offset+length])
            self.position += length
        return b"".join(chunks)

    def readinto(self, buffer):
        payload = self.read(len(buffer))
        buffer[:len(payload)] = payload
        return len(payload)


def http_fetcher(client, location, total_size, token=None, *, budget=None):
    """Exact HTTP206 fetch; reject response headers before reading a body."""
    # Only the original HF host may receive a bearer token, never signed CDN URLs.
    host = urlsplit(location).hostname
    auth = {"Authorization": f"Bearer {token}"} if host == "huggingface.co" and token else {}

    def fetch(start, count):
        if budget is not None:
            budget.check(count)
        headers = {**auth, "Range": f"bytes={start}-{start+count-1}", "Accept-Encoding": "identity"}
        try:
            with client.stream("GET", location, headers=headers, follow_redirects=True) as response:
                expected = f"bytes {start}-{start+count-1}/{total_size}"
                if response.status_code != 206 or response.headers.get("content-range") != expected:
                    raise RuntimeError("HTTP server ignored exact bounded range")
                if response.headers.get("content-encoding", "identity") not in ("", "identity"):
                    raise RuntimeError("Encoded HTTP range is unsupported")
                if response.headers.get("content-length") not in (None, str(count)):
                    raise RuntimeError("HTTP range Content-Length mismatch")
                raw = bytearray()
                for piece in response.iter_raw(chunk_size=min(65536, count)):
                    if budget is not None:
                        budget.consume(len(piece))
                    if len(raw)+len(piece) > count:
                        raise RuntimeError("Oversized HTTP range response")
                    raw.extend(piece)
                if len(raw) != count:
                    raise RuntimeError("Truncated HTTP range response")
                return response.status_code, response.headers.get("content-range"), bytes(raw)
        except Exception:
            # httpx exceptions may include presigned URLs; never propagate them.
            raise RuntimeError("Bounded HTTP video range failed; no fallback") from None
    fetch.manages_budget = budget is not None
    return fetch


def check_project_quota(root, reserve_bytes):
    root = Path(root).resolve()
    if root != Path("/projects/_ssd/jointposeaction"):
        raise RuntimeError("Dedicated resolved project root required")
    try:
        quota = int(os.getxattr(root, "ceph.quota.max_bytes").decode())
        used = int(os.getxattr(root, "ceph.dir.rbytes").decode())
    except (OSError, AttributeError, ValueError):
        raise RuntimeError("Measured dedicated Ceph quota unavailable") from None
    if not 0 < quota <= ALLOCATION_LIMIT or used < 0 or quota-used < reserve_bytes+HEADROOM_BYTES:
        raise RuntimeError("Insufficient dedicated quota plus 2 GB headroom")
    return {"quota_bytes": quota, "used_bytes": used, "free_bytes": quota-used,
            "output_reservation_bytes": reserve_bytes, "required_headroom_bytes": HEADROOM_BYTES}


def save_bytes(path, payload, budget):
    budget.check(len(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    budget.consume(len(payload))
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def encoded_png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=3)
    return buffer.getvalue()


def decode_survey(stream, *, seconds=3., sample_times=(0., 1., 2.5)):
    """Sequentially decode from source beginning; preserve actual native PTS."""
    import av
    import numpy as np
    if not 0 < seconds <= 3 or any(t < 0 or t >= seconds for t in sample_times):
        raise ValueError("Survey sampling must stay within the first <=3 seconds")
    selected, distances, motion = {}, {}, []
    first_time, previous, previous_time = None, None, None
    dimensions = None
    decoded_count = 0
    with av.open(stream, mode="r", format="mp4", buffer_size=32768) as container:
        if len(container.streams.video) != 1:
            raise RuntimeError("Exactly one video stream required")
        video = container.streams.video[0]
        video.thread_type = "AUTO"
        for ordinal, frame in enumerate(container.decode(video)):
            if frame.pts is None or frame.time_base is None:
                raise RuntimeError("Decoded frames need explicit PTS and time base")
            source_seconds = float(frame.pts*frame.time_base)
            if previous_time is not None and source_seconds <= previous_time:
                raise RuntimeError("Nonmonotonic decoded frame timestamps")
            previous_time = source_seconds
            if first_time is None:
                first_time = source_seconds
            elapsed = source_seconds-first_time
            if elapsed < 0:
                raise RuntimeError("Nonmonotonic decoded frame timestamps")
            if elapsed >= seconds:
                break
            image = frame.to_image().convert("RGB")
            if image.size != (1920, 1080):
                raise RuntimeError("Expected original 1920 x 1080 RGB video")
            dimensions = list(image.size)
            decoded_count += 1
            if decoded_count > 360:
                raise RuntimeError("Unexpected frame density in bounded original-video window")
            preview = np.asarray(image.convert("L").resize((96, 54)), dtype=np.float32)
            energy = None if previous is None else float(np.mean(np.abs(preview-previous)))
            previous = preview
            motion.append({"decoded_frame_ordinal_from_stream_start": ordinal,
                           "pts": int(frame.pts), "pts_seconds": source_seconds,
                           "elapsed_from_first_decoded_pts_seconds": elapsed,
                           "mean_abs_gray_frame_difference": energy})
            for target in sample_times:
                distance = abs(elapsed-target)
                if distance < distances.get(target, float("inf")):
                    distances[target] = distance
                    selected[target] = (image.copy(), {"requested_elapsed_seconds": target,
                        "pts": int(frame.pts), "time_base_numerator": frame.time_base.numerator,
                        "time_base_denominator": frame.time_base.denominator, "pts_seconds": source_seconds,
                        "elapsed_from_first_decoded_pts_seconds": elapsed,
                        "sampling_error_seconds": distance,
                        "decoded_frame_ordinal_from_stream_start": ordinal})
    if not selected or max(selected) < max(sample_times) or any(d > 1/15 for d in distances.values()):
        raise RuntimeError("Survey does not cover requested source time samples")
    return selected, {"decoded_frame_count": decoded_count, "image_size": dimensions,
                      "first_decoded_pts_seconds": first_time, "motion_trace": motion,
                      "synchronization_verified": False}


def contact_sheet(recording, view_rows, output, budget):
    from PIL import Image, ImageDraw
    sheet = Image.new("RGB", (1920, 620), "#202020")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), "RAW PTS survey - synchronization UNVERIFIED", fill="white")
    draw.text((12, 26), recording, fill="white")
    for index, camera_id in enumerate(CAMERA_IDS):
        x, y = (index % 4)*480, 60+(index//4)*280
        row = next((r for r in view_rows if r["camera_id"] == camera_id), None)
        frame = None if row is None else next((f for f in row.get("frames", []) if f["requested_elapsed_seconds"] == 1.), None)
        if frame:
            with Image.open(frame["path"]) as image:
                sheet.paste(image.resize((480, 270)), (x, y))
            label = f"{camera_id} sourcePTS={frame['pts_seconds']:.4f}s"
        else:
            label = f"{camera_id} unavailable"
        draw.rectangle((x, y, x+479, y+19), fill="black")
        draw.text((x+5, y+4), label, fill="white")
    return save_bytes(output, encoded_png(sheet), budget)


def acquire(discovery_path, output, *, max_recordings=3, seconds=3., max_network_bytes=NETWORK_LIMIT, max_output_bytes=OUTPUT_LIMIT):
    if not 1 <= max_recordings <= 3 or not 2.5 < seconds <= 3:
        raise ValueError("Survey permits one to three recordings and >2.5 to 3 seconds")
    if not 0 < max_network_bytes <= NETWORK_LIMIT or not 4_194_304 <= max_output_bytes <= OUTPUT_LIMIT:
        raise ValueError("Cannot exceed reviewed survey caps")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CPU Slurm execution required")
    root = Path("/projects/jointposeaction").resolve()
    output = Path(output).resolve()
    if not output.is_relative_to(root / "experiments/segmentation_pilot") or output.exists():
        raise RuntimeError("A new directory inside dedicated pilot experiment storage is required")
    discovery = json.loads(Path(discovery_path).read_text())
    if discovery.get("dataset_revision") != DATASET_REVISION or discovery.get("toolkit_revision") != TOOLKIT_REVISION:
        raise RuntimeError("Discovery source revisions changed")
    quota = check_project_quota(root, max_output_bytes)
    # Retain room to describe partial failures even when native PNG storage fills.
    network_budget, output_budget = ByteBudget(max_network_bytes), ByteBudget(max_output_bytes-2_097_152)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "raw_survey_not_validated_pilot", "segmentation_ready": False,
              "dataset_revision": DATASET_REVISION, "toolkit_revision": TOOLKIT_REVISION,
              "job_id": os.environ["SLURM_JOB_ID"], "quota_before": quota,
              "max_network_bytes": max_network_bytes, "max_output_bytes": max_output_bytes,
              "discovery_path": str(Path(discovery_path).resolve()),
              "discovery_sha256": hashlib.sha256(Path(discovery_path).read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cameras": list(CAMERA_IDS), "recordings": [], "video_files_downloaded": False,
              "limitations": ["Nearest source-PTS samples do not establish temporal correspondence.",
                              "Motion traces are descriptive; independent held-out events remain required.",
                              "Physical target identities, visibility and segmentation remain unvalidated."]}
    from huggingface_hub import get_token, get_hf_file_metadata, hf_hub_url
    import httpx
    token = get_token()
    try:
        candidates = [r for r in discovery["candidates"] if r.get("recognition_split") == "train"][:max_recordings]
        with httpx.Client(timeout=httpx.Timeout(45., read=60.), headers={"User-Agent": "JointPoseAction-bounded-survey"}) as client:
            for candidate in candidates:
                recording_id = candidate["recording_id"]
                if not re.fullmatch(r"nusar-2021_action_both_[A-Za-z0-9_-]+", recording_id):
                    raise RuntimeError("Invalid candidate recording identifier")
                by_camera = {v["camera_id"]: v for v in candidate["views"]}
                if set(by_camera) != set(CAMERA_IDS) or len(candidate["views"]) != 8:
                    raise RuntimeError("Candidate must contain all eight exact fixed cameras")
                calibration = load_nimble_calibration(candidate["calibration_path"], recording_id)
                row = {"recording_id": recording_id, "recognition_split": "train", "views": [],
                       "calibration_path": candidate["calibration_path"], "calibration_sha256": calibration["sha256"],
                       "calibration_structural_checks_passed": True, "synchronization_verified": False,
                       "segmentation_ready": False}
                report["recordings"].append(row)
                for camera_id in CAMERA_IDS:
                    view, before = by_camera[camera_id], network_budget.used
                    result = {"camera_id": camera_id, "source_path": view["path"], "status": "failed", "frames": []}
                    row["views"].append(result)
                    try:
                        expected_path = f"recordings/{recording_id}/{camera_id}_rgb.mp4"
                        if view["path"] != expected_path:
                            raise RuntimeError("Candidate source path mismatch")
                        meta = get_hf_file_metadata(hf_hub_url("cvml-nus/assembly101", view["path"], repo_type="dataset", revision=DATASET_REVISION), token=token)
                        if meta.size != view["bytes"] or meta.commit_hash != DATASET_REVISION:
                            raise RuntimeError("Immutable video metadata mismatch")
                        result.update({"source_bytes": meta.size, "source_etag": meta.etag,
                                       "source_revision": meta.commit_hash})
                        stream = HTTPRangeIO(meta.size, http_fetcher(client, meta.location, meta.size, token, budget=network_budget), network_budget)
                        selected, diagnostics = decode_survey(stream, seconds=seconds)
                        for target, (image, details) in sorted(selected.items()):
                            name = f"{camera_id}_elapsed-{target:04.1f}_pts-{details['pts']}.png"
                            saved = save_bytes(output/recording_id/"native_frames"/name, encoded_png(image), output_budget)
                            result["frames"].append({**saved, **details, "image_size": list(image.size), "synchronization_verified": False})
                        result.update({"status": "decoded", "source_bytes": meta.size,
                                       "source_etag": meta.etag, "source_revision": meta.commit_hash, **diagnostics})
                    except Exception as error:
                        # Never serialize exception text: network URLs can contain secrets.
                        result["error"] = {"type": type(error).__name__, "reason": "Bounded source view survey failed; no full-video fallback"}
                    result["network_payload_bytes"] = network_budget.used-before
                    print(json.dumps({"recording_id": recording_id, "camera_id": camera_id,
                                      "status": result["status"], "network_bytes": network_budget.used,
                                      "output_bytes": output_budget.used}), flush=True)
                row["decoded_views"] = sum(v["status"] == "decoded" for v in row["views"])
                row["contact_sheet"] = contact_sheet(recording_id, row["views"], output/recording_id/"RAW_UNVERIFIED_contact_sheet.png", output_budget)
    finally:
        report.update({"network_payload_bytes": network_budget.used, "output_bytes_before_manifest": output_budget.used})
        payload = (json.dumps(report, indent=2)+"\n").encode()
        output_budget.limit = max_output_bytes
        save_bytes(output/"survey.json", payload, output_budget)
        print(json.dumps({"survey": str(output/"survey.json"), "network_bytes": network_budget.used,
                          "output_bytes": output_budget.used, "segmentation_ready": False}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-recordings", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=3.)
    parser.add_argument("--max-network-bytes", type=int, default=NETWORK_LIMIT)
    parser.add_argument("--max-output-bytes", type=int, default=OUTPUT_LIMIT)
    args = parser.parse_args()
    acquire(args.discovery, args.output, max_recordings=args.max_recordings, seconds=args.seconds,
            max_network_bytes=args.max_network_bytes, max_output_bytes=args.max_output_bytes)


if __name__ == "__main__":
    main()
