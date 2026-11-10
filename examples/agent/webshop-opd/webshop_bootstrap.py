"""Select the repository package and reject modules cached from older copies."""

import sys
from pathlib import Path


def require_repository_triotune():
    repository = Path(__file__).resolve().parents[3]
    expected = (repository / "triotune").resolve()
    # Application modules can be imported directly, before an executable has
    # arranged sys.path. Keep an installed or leftover copy from winning then.
    if not sys.path or sys.path[0] != str(repository):
        sys.path.insert(0, str(repository))
    import triotune

    for name, module in tuple(sys.modules.items()):
        if name != "triotune" and not name.startswith("triotune."):
            continue
        origin = getattr(module, "__file__", None)
        if isinstance(origin, (str, Path)):
            actual = Path(origin).resolve()
            if (name == "triotune" and actual.parent == expected) or (
                name != "triotune" and actual.is_relative_to(expected)
            ):
                continue
        # Do not remove or reload cached modules: existing class instances can
        # retain the old implementations even after their files are deleted.
        raise RuntimeError(
            f"WebShop requires the repository triotune package at {expected}, "
            f"but {name} is already imported from {origin!r}. "
            "Start a fresh Python process before running the WebShop entry point."
        )
