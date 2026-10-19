"""Configuration loading helpers shared by training recipes."""

from __future__ import annotations

import tomllib
from pathlib import Path
import re
from typing import Any

import yaml

_REFERENCE = re.compile(r"\$\{([^{}]+)\}")


def _resolve_references(settings: dict[str, Any]) -> dict[str, Any]:
    """Resolve simple ``${section.field}`` references used by YAML recipes."""

    def lookup(path: str) -> Any:
        value: Any = settings
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f"Unknown YAML reference: ${{{path}}}")
            value = value[part]
        return value

    resolving: set[str] = set()

    def resolve(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if not isinstance(value, str) or not _REFERENCE.search(value):
            return value
        match = _REFERENCE.fullmatch(value)
        if match:
            path = match.group(1)
            if path in resolving:
                raise ValueError(f"Cyclic YAML reference: ${{{path}}}")
            resolving.add(path)
            try:
                return resolve(lookup(path))
            finally:
                resolving.remove(path)

        def replace(match: re.Match[str]) -> str:
            path = match.group(1)
            resolved = resolve(lookup(path))
            return str(resolved)

        return _REFERENCE.sub(replace, value)

    return resolve(settings)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a mapping from a UTF-8 YAML configuration file.

    Keeping this small helper in the package makes all recipe entry points use
    the same safe loader and gives callers a useful error for empty or scalar
    YAML documents.
    """

    config_path = Path(path).expanduser()
    # TOML was used by earlier releases.  Keep reading an explicitly supplied
    # legacy file while making YAML the default and documented format.  Trying
    # TOML first also lets older callers that monkeypatch ``tomllib.load``
    # continue to validate configuration mappings without touching the YAML
    # parser.
    with config_path.open("rb") as source:
        try:
            settings = tomllib.load(source)
        except tomllib.TOMLDecodeError:
            source.seek(0)
            settings = yaml.safe_load(source)
    if not isinstance(settings, dict):
        raise ValueError(f"Config {config_path} must contain a YAML mapping")
    return _resolve_references(settings)
