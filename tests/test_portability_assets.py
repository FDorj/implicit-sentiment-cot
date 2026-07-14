from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class PortabilityAssetTests(unittest.TestCase):
    def test_windows_entry_points_exist_and_are_machine_independent(self):
        expected = {
            "setup_windows.ps1": [
                ".venv",
                "ensurepip",
                "Lib\\site-packages\\pip",
                "-m",
                "pip",
                "requirements.txt",
            ],
            "build_thesis.ps1": ["xelatex", "bibtex", "AUTthesis.tex", "AUTthesis.pdf"],
            "verify_project.ps1": ["unittest", "run_final_pipeline.py", "build_thesis.ps1"],
        }

        for filename, tokens in expected.items():
            path = REPO_ROOT / "scripts" / filename
            self.assertTrue(path.is_file(), path)
            text = path.read_text(encoding="utf-8")
            for token in tokens:
                self.assertIn(token, text, f"{token!r} missing from {path}")
            self.assertNotRegex(text, r"[A-Z]:\\Users\\")

    def test_readme_documents_fresh_clone_and_thesis_build(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("scripts/setup_windows.ps1", readme)
        self.assertIn("ollama pull qwen3:8b", readme)
        self.assertIn("scripts/build_thesis.ps1", readme)
        self.assertIn("AUTthesis.pdf", readme)
        self.assertIn("Install missing packages on-the-fly", readme)


if __name__ == "__main__":
    unittest.main()
