"""Launch instagram using the shared project runtime."""
from pathlib import Path
import os
import runpy
import sys


def main():
    root = Path(__file__).resolve().parents[3]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    args = sys.argv[1:]
    defaults = {
        "--config": str(root / "config/config_instagram.json"),
        "--model_config": str(root / "config/model_config.json"),
        "--env": "instagram",
        "--mode": "single",
    }
    for key, value in defaults.items():
        if not any(arg == key or arg.startswith(key + "=") for arg in args):
            args.extend([key, value])
    sys.argv = [str(root / "src/main.py"), *args]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
