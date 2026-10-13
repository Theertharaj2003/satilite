from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import platform
import sys


def check(name):
    try:
        mod = __import__(name)
        return True, getattr(mod, "__version__", "")
    except Exception as exc:
        return False, str(exc)


def main():
    print("=== UAV Defence AI Environment Check ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OS: {platform.platform()}")
    for name in ["numpy", "cv2", "yaml", "pandas", "sklearn", "torch", "ultralytics", "streamlit", "onnxruntime"]:
        ok, detail = check(name)
        print(f"{name:14s}: {'OK' if ok else 'MISSING'} {detail}")
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"CUDA check failed: {exc}")


if __name__ == "__main__":
    main()
