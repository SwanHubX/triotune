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
        # 发布 workflow 会递增版本，测试不应绑定到某一个已发布版本。
        self.assertRegex(
            project["version"],
            r"\A(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z",
        )
        self.assertEqual(project["readme"], "README.md")

    def test_lockfile_matches_project_version(self) -> None:
        config = tomllib.loads(
            (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        lock = tomllib.loads(
            (REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8")
        )
        project = config["project"]
        locked_project = next(
            package for package in lock["package"]
            if package["name"] == project["name"]
        )

        self.assertEqual(locked_project["version"], project["version"])
        self.assertEqual(locked_project["source"], {"editable": "."})

    def test_source_distribution_excludes_repository_only_content(self) -> None:
        manifest = (REPOSITORY_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

        self.assertIn("include README.md", manifest)
        self.assertIn("prune docs", manifest)
        self.assertIn("prune tests", manifest)


if __name__ == "__main__":
    unittest.main()
