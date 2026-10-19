"""Release metadata and source-distribution boundary checks."""

from pathlib import Path
import tomllib
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DistributionMetadataTests(unittest.TestCase):
    def test_release_uses_triotune_name_and_root_readme(self) -> None:
        config = tomllib.loads(
            (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        project = config["project"]

        self.assertEqual(project["name"], "triotune")
        self.assertEqual(project["version"], "0.1.2")
        self.assertEqual(project["readme"], "README.md")

    def test_source_distribution_excludes_repository_only_content(self) -> None:
        manifest = (REPOSITORY_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

        self.assertIn("include README.md", manifest)
        self.assertIn("prune docs", manifest)
        self.assertIn("prune tests", manifest)


if __name__ == "__main__":
    unittest.main()
