"""Load all prompt resources once, preserving explicit-only substitution."""
from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

_ROOT = Path(__file__).resolve().parents[1] / "prompts"
_FIELD_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_PROMPT_CACHE = {
    path.relative_to(_ROOT).as_posix(): path.read_text(encoding="utf-8")
    for path in _ROOT.rglob("*.txt")
}
_POLICY_CACHE = {}
for _path in (_ROOT / "policies").glob("*.toml"):
    with _path.open("rb") as _stream:
        _document = tomllib.load(_stream)
    for _section, _fields in _document.items():
        if not isinstance(_fields, dict) or not all(isinstance(v, str) for v in _fields.values()):
            raise ValueError(f"Policy {_path}:{_section} must contain named text fields")
    _POLICY_CACHE[_path.stem] = _document


def load_prompt(rel_path: str) -> str:
    key = rel_path.replace("\\", "/").lstrip("./")
    try:
        return _PROMPT_CACHE[key]
    except KeyError:
        raise FileNotFoundError(f"Prompt not found: {rel_path!r} under {_ROOT}") from None


def format_text(text: str, **kwargs) -> str:
    return _FIELD_RE.sub(
        lambda match: str(kwargs[match.group(1)]) if match.group(1) in kwargs else match.group(0),
        text,
    )


def format_prompt(rel_path: str, **kwargs) -> str:
    return format_text(load_prompt(rel_path), **kwargs)


def load_policy(group: str, section: str, field: str = "guidance") -> str:
    try:
        return _POLICY_CACHE[group][section][field]
    except KeyError:
        raise KeyError(f"Missing policy: {group}.{section}.{field}") from None


def resolve_prompt(rel_path: str, fallback: str) -> str:
    key = rel_path.replace("\\", "/").lstrip("./")
    return key if key in _PROMPT_CACHE else fallback.replace("\\", "/").lstrip("./")
