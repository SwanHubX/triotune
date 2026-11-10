"""Reject a cached experimental package before starting a shared-package run."""

from pathlib import Path


def require_repository_triotune():
    import triotune

    expected = Path(__file__).resolve().parents[3] / "triotune"
    actual = Path(triotune.__file__).resolve().parent
    if actual != expected:
        raise RuntimeError(
            "WebShop requires the repository triotune package, but another copy "
            "is already imported. Start a fresh Python process before running "
            "the WebShop entry point."
        )
