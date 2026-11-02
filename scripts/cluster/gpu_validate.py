"""Small numerical and project I/O test; must run inside a Slurm GPU allocation."""

import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this check through Slurm, not on a login node")
    root = Path(os.environ["JPA_ROOT"])
    if root.resolve() != Path("/projects/_ssd/jointposeaction"):
        raise RuntimeError("Unexpected project storage")
    import cv2
    import numpy as np
    import torch

    result = {"job_id": os.environ["SLURM_JOB_ID"], "python": sys.version,
              "numpy": np.__version__, "opencv": cv2.__version__,
              "pytorch": torch.__version__, "torch_cuda": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(),
              "compiled_architectures": torch.cuda.get_arch_list()}
    print(json.dumps(result, indent=2), flush=True)
    if not result["cuda_available"]:
        raise RuntimeError("CUDA unavailable in Slurm job")
    result["gpu"] = torch.cuda.get_device_name(0)
    result["capability"] = torch.cuda.get_device_capability(0)
    # Matrix multiplication exercises an actual GPU kernel, not just enumeration.
    a = torch.ones((32, 32), device="cuda")
    product = a @ a
    torch.cuda.synchronize()
    if not torch.equal(product, torch.full_like(product, 32)):
        raise RuntimeError("GPU numerical check failed")
    source = root / "repo/reproducibility/upstreams.json"
    if not json.loads(source.read_text(encoding="utf-8"))["dependencies"]:
        raise RuntimeError("Project input JSON is empty")
    with tempfile.TemporaryDirectory(prefix="jpa-smoke-", dir=root / "data/processed") as directory:
        target = Path(directory) / "roundtrip.png"
        pixels = np.zeros((8, 8, 3), dtype=np.uint8)
        if not cv2.imwrite(str(target), pixels):
            raise RuntimeError("OpenCV project write failed")
        if not np.array_equal(cv2.imread(str(target)), pixels):
            raise RuntimeError("OpenCV project read failed")
    result.update(status="passed", project_input=str(source), io_roundtrip="passed")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
