import hashlib
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
THESIS_DIR = (
    REPO_ROOT
    / "قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir"
)


def read_thesis_file(name: str) -> str:
    return (THESIS_DIR / name).read_text(encoding="utf-8")


class ThesisFinalizationTests(unittest.TestCase):
    def test_defense_approval_source_is_unchanged(self):
        content = (THESIS_DIR / "taid.tex").read_text(encoding="utf-8")
        normalized = content.replace("\r\n", "\n").encode("utf-8")
        digest = hashlib.sha256(normalized).hexdigest()
        self.assertEqual(
            digest,
            "8212555f994bea6aae5976199921b1cc55c925063440dd862e5cc3d4ac9adab8",
        )

    def test_personal_pages_use_approved_text(self):
        dedication = read_thesis_file("Chant.tex")
        acknowledgement = read_thesis_file("acknowledgement.tex")

        self.assertIn("تقدیم به پدر و مادرم", dedication)
        self.assertIn("مسیر آموختن را برایم هموار کردند", dedication)
        self.assertIn("دکتر مصطفی حقیر چهرقانی", acknowledgement)
        self.assertIn("از خانواده‌ام", acknowledgement)
        self.assertNotIn("سپاس خدای را", acknowledgement)
        self.assertNotIn("نویسندۀ پایان‌نامه می‌تواند", acknowledgement)
        self.assertNotIn("نويسنده پايان‌نامه", dedication)

    def test_english_front_matter_is_complete(self):
        title = read_thesis_file("en_title.tex")
        abstract = read_thesis_file("en-abstract.tex")
        combined = title + abstract

        for placeholder in [
            "Department of ...",
            "Title of Thesis",
            "Dr. }",
            "Name}",
            "Surname}",
            "Month \\& Year",
            "Write a 3 to 5 KeyWords",
            "This page is accurate translation",
        ]:
            self.assertNotIn(placeholder, combined)
        self.assertIn("Implicit Sentiment Analysis", combined)
        self.assertIn("Fatemeh", title)
        self.assertIn("Darj", title)
        self.assertIn("Mostafa Haghir Chehreghani", title)
        self.assertIn("Large Language Models", abstract)

    def test_symbols_match_the_actual_thesis(self):
        symbols = read_thesis_file("list-of-symbols.tex")

        for symbol in [
            r"\symb{s}",
            r"\symb{t}",
            r"\symb{a}",
            r"\symb{o}",
            r"\symb{y}",
            r"\symb{P}",
            r"\symb{R}",
            r"F_1",
            r"\pi",
            r"\mathcal{D}_{\mathrm{train}}",
            r"\mathcal{D}_{\mathrm{test}}",
        ]:
            self.assertIn(symbol, symbols)
        for unrelated in ["فضای اقلیدسی", "خمینه", "ریچی", "ساساکی"]:
            self.assertNotIn(unrelated, symbols)

    def test_appendix_documents_project_reproducibility(self):
        appendix = read_thesis_file("appendix1.tex")

        for expected in [
            "qwen3:8b",
            "Ollama",
            "۰/۷",
            "سه مسیر استدلالی",
            "run_final_pipeline.py",
            "generate_thesis_result_figures.py",
            "build_thesis.ps1",
            "مدل زبانی را دوباره اجرا نمی‌کند",
        ]:
            self.assertIn(expected, appendix)
        for unrelated in ["کد میپل", "DifferentialGeometry", "DGsetup"]:
            self.assertNotIn(unrelated, appendix)
        self.assertNotRegex(appendix, r"\\lr\{[^}]*_")

    def test_dictionaries_contain_only_project_terms(self):
        fa_to_en = read_thesis_file("dicfa2en.tex")
        en_to_fa = read_thesis_file("dicen2fa.tex")
        combined = fa_to_en + en_to_fa

        for expected in [
            "تحلیل احساسات ضمنی",
            "Implicit Sentiment Analysis",
            "زنجیرۀ تفکر",
            "Chain of Thought",
            "انتخاب منبع",
            "Source Selection",
            "ماتریس درهم‌ریختگی",
            "Confusion Matrix",
            "Macro-F1",
        ]:
            self.assertIn(expected, combined)
        for unrelated in ["Automorphism", "Permutation", "Quotient graph", "زیرمدول"]:
            self.assertNotIn(unrelated, combined)

    def test_dictionary_macros_force_the_latin_font(self):
        thesis_class = read_thesis_file("AUTthesis.cls")

        self.assertIn(r"\lr{\resetlatinfont #1}", thesis_class)
        self.assertIn(r"\lr{\resetlatinfont #2}", thesis_class)

    def test_unrelated_template_references_are_removed(self):
        bibliography = read_thesis_file("references.bib")

        for key in [
            "bidabad2007classification",
            "@book{aa,",
            "najafi2008finsler",
            "@book{najafi,",
            "@book{zakeri,",
            "obradovic2023decentralized",
        ]:
            self.assertNotIn(key, bibliography)

    def test_thesis_prose_uses_academic_terminology(self):
        prose = "\n".join(
            read_thesis_file(name)
            for name in [
                "fa_title.tex",
                "chapter1.tex",
                "chapter2.tex",
                "chapter3.tex",
                "chapter4.tex",
                "chapter5.tex",
                "chapter6.tex",
            ]
        )

        self.assertNotIn("original-ish", prose)
        self.assertNotIn("Macro-F۱", prose)
        self.assertNotIn("F۱-score", prose)
        self.assertIn("Reasoning Implicit Sentiment with Chain-of-Thought Prompting", prose)
        self.assertIn("Supervised Contrastive Pre-Training", prose)
        self.assertIn("Sentiment Analysis of Thinking", prose)
        self.assertIn(r"\ref{tab:ch5-class-f1}", prose)

    def test_persian_prose_avoids_unsupported_combining_hamza(self):
        prose = "\n".join(
            read_thesis_file(name)
            for name in [
                "chapter1.tex",
                "chapter2.tex",
                "chapter3.tex",
                "chapter4.tex",
                "chapter5.tex",
                "chapter6.tex",
            ]
        )

        self.assertNotIn("ٔ", prose)

    def test_ltr_footnotes_contain_only_latin_text(self):
        prose = "\n".join(
            read_thesis_file(name)
            for name in [
                "chapter1.tex",
                "chapter2.tex",
                "chapter3.tex",
                "chapter4.tex",
                "chapter5.tex",
                "chapter6.tex",
            ]
        )
        ltr_footnotes = re.findall(r"\\LTRfootnote\{([^}]*)\}", prose)

        self.assertGreater(len(ltr_footnotes), 0)
        for footnote in ltr_footnotes:
            with self.subTest(footnote=footnote):
                self.assertIsNone(re.search(r"[\u0600-\u06ff]", footnote))

    def test_latin_terms_do_not_fall_back_to_the_persian_font(self):
        chapter2 = read_thesis_file("chapter2.tex")
        chapter4 = read_thesis_file("chapter4.tex")
        prose = chapter2 + chapter4

        self.assertNotIn("parser", prose)
        self.assertNotIn("خروجی CSV", prose)
        self.assertIn(r"خروجی \lr{CSV}", chapter4)
        self.assertNotIn("نمونه--منبع", prose)
        for label in ["positive", "negative", "neutral"]:
            self.assertNotIn(rf"\text{{{label}}}", chapter4)
            self.assertIn(rf"\text{{\lr{{{label}}}}}", chapter4)

    def test_pdf_links_and_metadata(self):
        commands = read_thesis_file("commands.tex")

        self.assertIn("hidelinks", commands)
        self.assertNotIn("linkcolor=blue", commands)
        self.assertNotIn("citecolor=red", commands)
        for field in ["pdftitle", "pdfauthor", "pdfsubject", "pdfkeywords"]:
            self.assertIn(field, commands)


if __name__ == "__main__":
    unittest.main()
