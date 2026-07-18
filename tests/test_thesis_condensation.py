import hashlib
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT / "قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir"


class ThesisCondensationTests(unittest.TestCase):
    def test_complete_pdf_has_60_to_65_pages(self):
        completed = subprocess.run(
            ["pdfinfo", str(THESIS / "AUTthesis.pdf")],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        match = re.search(r"^Pages:\s+(\d+)$", completed.stdout, re.MULTILINE)
        self.assertIsNotNone(match, completed.stdout)
        pages = int(match.group(1))
        self.assertGreaterEqual(pages, 60, f"Thesis is too short: {pages} pages")
        self.assertLessEqual(pages, 65, f"Thesis is too long: {pages} pages")

    def test_condensed_chapters_preserve_scientific_anchors(self):
        required = {
            "chapter1.tex": ["پرسش‌های پژوهش", "نوآوری‌ها"],
            "chapter2.tex": ["تحلیل احساسات ضمنی", "اف‌یک ماکرو"],
            "chapter3.tex": ["SCAPT", "THOR", "SAoT", "شکاف تحقیقاتی"],
            "chapter4.tex": ["P_R", "اعتبارسنجی داخلی", "پیش‌بینی نهایی"],
            "chapter5.tex": ["۰٫۷۲۳۹۸۲", "۰٫۷۱۹۱۱۹", "Gemini 2.5 Flash"],
            "chapter6.tex": ["محدودیت‌های پژوهش", "کارهای آتی"],
        }
        for filename, anchors in required.items():
            content = (THESIS / filename).read_text(encoding="utf-8")
            for anchor in anchors:
                with self.subTest(filename=filename, anchor=anchor):
                    self.assertIn(anchor, content)

    def test_defense_approval_content_is_unchanged(self):
        content = (THESIS / "taid.tex").read_text(encoding="utf-8")
        normalized = content.replace("\r\n", "\n").encode("utf-8")
        digest = hashlib.sha256(normalized).hexdigest()
        self.assertEqual(
            digest,
            "8212555f994bea6aae5976199921b1cc55c925063440dd862e5cc3d4ac9adab8",
        )


if __name__ == "__main__":
    unittest.main()
