"""CPU import/ABI check for the pinned E0 stack, executed in the install job."""

import json
import platform


def main():
    import cv2
    import huggingface_hub
    import matplotlib
    import numpy as np
    import pandas as pd
    import scipy
    from scipy.spatial.transform import Rotation
    import torch
    import yaml

    if tuple(map(int, platform.python_version_tuple()[:2])) != (3, 12):
        raise RuntimeError("Expected Python 3.12")
    if torch.__version__.split("+")[0] != "2.14.1" or torch.version.cuda != "13.0":
        raise RuntimeError("Unexpected PyTorch/CUDA build")
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    if not np.array_equal(torch.from_numpy(values).numpy(), values):
        raise RuntimeError("PyTorch/NumPy bridge failed")
    if not np.allclose(Rotation.identity().apply(values), values):
        raise RuntimeError("SciPy/NumPy check failed")
    if cv2.resize(values, (3, 2)).shape != (2, 3):
        raise RuntimeError("OpenCV/NumPy check failed")
    if pd.DataFrame(values).shape != (4, 3):
        raise RuntimeError("pandas/NumPy check failed")
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                      "torch_cuda": torch.version.cuda, "numpy": np.__version__,
                      "opencv": cv2.__version__, "scipy": scipy.__version__,
                      "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
                      "yaml": yaml.__version__, "huggingface_hub": huggingface_hub.__version__,
                      "cpu_abi_checks": "passed"}, indent=2))


if __name__ == "__main__":
    main()
