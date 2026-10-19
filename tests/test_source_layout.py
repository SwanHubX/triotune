"""Regression checks for the repository source-directory layout."""

from pathlib import Path
import tomllib
import unittest

import trio_cookbook


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "triotune"


class SourceLayoutTests(unittest.TestCase):
    def test_source_uses_triotune_directory(self) -> None:
        self.assertTrue((SOURCE_ROOT / "__init__.py").is_file())
        self.assertFalse((REPOSITORY_ROOT / "trio-cookbook").exists())

    def test_public_import_resolves_from_triotune_directory(self) -> None:
        self.assertEqual(
            Path(trio_cookbook.__file__).resolve(),
            SOURCE_ROOT / "__init__.py",
        )

    def test_setuptools_maps_public_packages_to_triotune(self) -> None:
        config = tomllib.loads(
            (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        package_dir = config["tool"]["setuptools"]["package-dir"]

        self.assertEqual(package_dir["trio_cookbook"], "triotune")
        self.assertEqual(package_dir["trio_cookbook.trainer"], "triotune/trainer")

    def test_runtime_files_do_not_reference_legacy_source_path(self) -> None:
        paths = [
            *SOURCE_ROOT.rglob("*.py"),
            *SOURCE_ROOT.rglob("*.sh"),
            *REPOSITORY_ROOT.joinpath("tests").rglob("*.py"),
        ]
        stale = [
            path.relative_to(REPOSITORY_ROOT)
            for path in paths
            if path.resolve() != Path(__file__).resolve()
            and "trio-cookbook/" in path.read_text(encoding="utf-8")
        ]

        self.assertEqual(stale, [])


if __name__ == "__main__":
    unittest.main()
