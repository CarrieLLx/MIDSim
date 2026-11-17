"""Load UTF-8 prompt files once, following the project's prompt-cache pattern."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()
