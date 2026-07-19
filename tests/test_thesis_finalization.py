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
    def test_pipeline_compare_description_uses_normal_weight(self):
        figure = read_thesis_file("Images/Chapter4/ch4_proposed_pipeline.tex")
        compare_style = re.search(
            r"compare/\.style=\{(.*?)\n  \},",
            figure,
            flags=re.DOTALL,
        )

        self.assertIsNotNone(compare_style)
        self.assertNotIn(r"\bfseries", compare_style.group(1))
        self.assertIn(r"\textbf{مقایسۀ دو مسیر}", figure)

    def test_pipeline_input_description_uses_normal_weight(self):
        figure = read_thesis_file("Images/Chapter4/ch4_proposed_pipeline.tex")
        input_style = re.search(
            r"input/\.style=\{(.*?)\n  \},",
            figure,
            flags=re.DOTALL,
        )

        self.assertIsNotNone(input_style)
        self.assertNotIn(r"\bfseries", input_style.group(1))
        self.assertIn(r"\textbf{ورودی سامانه}", figure)

    def test_pipeline_training_connector_has_visible_clearance(self):
        figure = read_thesis_file("Images/Chapter4/ch4_proposed_pipeline.tex")

        def style_text_width(style_name):
            style = re.search(
                rf"{style_name}/\.style=\{{(.*?)\n  \}},",
                figure,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(style)
            width = re.search(r"text width=([0-9.]+)cm", style.group(1))
            self.assertIsNotNone(width)
            return float(width.group(1))

        def node_x(node_name):
            node = re.search(
                rf"\\node\[{node_name}\] \({node_name}\) at \(([-0-9.]+),",
                figure,
            )
            self.assertIsNotNone(node)
            return float(node.group(1))

        inner_xsep = re.search(r"inner xsep=([0-9.]+)pt", figure)
        self.assertIsNotNone(inner_xsep)
        inner_xsep_cm = float(inner_xsep.group(1)) * 2.54 / 72.27

        training_right = (
            node_x("training")
            + style_text_width("training") / 2
            + inner_xsep_cm
        )
        selector_left = (
            node_x("selector")
            - style_text_width("selector") / 2
            - inner_xsep_cm
        )

        self.assertGreaterEqual(selector_left - training_right, 0.9)
        self.assertIn(
            r"\draw[auxiliary] (training.east) -- (selector.west);",
            figure,
        )

    def test_pipeline_figure_uses_clear_approved_copy(self):
        figure = read_thesis_file("Images/Chapter4/ch4_proposed_pipeline.tex")

        for expected in [
            "جمله و هدفی که باید احساس نسبت به آن تعیین شود",
            "پیش‌بینی مستقیم",
            "مسیر استدلالی \\lr{THOR}",
            "تعیین برچسب با رأی اکثریت",
            "تحلیل اختلاف",
            "نوع خطا، برچسب پیشنهادی و سطح اطمینان تعیین می‌شود",
            "ساخت پروفایل انتخاب",
            "دو برچسب، نوع خطا، اطمینان و دامنه",
            "تنظیم با دادۀ آموزش",
            "انتخاب منبع نهایی",
            "در غیر این صورت، پاسخ مستقیم حفظ می‌شود",
            "فقط یک تصمیم میانی برای مقایسه می‌سازد",
            "پیش‌بینی نهایی",
        ]:
            self.assertIn(expected, figure)

        for unclear in [
            "عبارتی که احساس نسبت به آن سنجیده می‌شود",
            "برچسب‌های تولیدشده و نتیجۀ بررسی اختلاف",
            "پروفایل عملکردی پنج‌جزئی و شروط محافظ",
            "تولید تصمیم کمکی برای تحلیل",
        ]:
            self.assertNotIn(unclear, figure)

    def test_defense_approval_placeholder_is_removed_but_declaration_is_kept(self):
        content = (THESIS_DIR / "taid.tex").read_text(encoding="utf-8")
        self.assertNotIn("صفحه فرم ارزیابی و تصویب پایان نامه", content)
        self.assertNotIn("در این صفحه فرم دفاع", content)
        self.assertIn("تعهدنامه اصالت اثر", content)
        self.assertIn("متعهد می‌شوم", content)

    def test_persian_abstract_is_rendered_before_front_matter_lists(self):
        main = read_thesis_file("AUTthesis.tex")
        abstract = read_thesis_file("fa-abstract.tex")

        self.assertIn(r"\input{fa-abstract}", main)
        self.assertIn(r"\clearpage" + "\n" + r"\pagenumbering{alph}", main)
        self.assertFalse(abstract.lstrip().startswith(r"\newpage"))
        self.assertLess(
            main.index(r"\pagenumbering{alph}"),
            main.index(r"\input{fa-abstract}"),
        )
        self.assertLess(
            main.index(r"\input{fa-abstract}"),
            main.index(r"\input{TOC-TOF-LOT}"),
        )
        self.assertIn(r"\ffa-abstract", abstract)
        self.assertIn(r"\fkeywords", abstract)
        self.assertIn("چکیده", abstract)

    def test_acronym_footnotes_do_not_repeat_abbreviations(self):
        chapters = "\n".join(
            read_thesis_file(name)
            for name in ["chapter1.tex", "chapter2.tex", "chapter3.tex", "chapter5.tex"]
        )
        self.assertNotIn(
            "Reasoning Implicit Sentiment with Chain-of-Thought Prompting",
            chapters,
        )
        for redundant in [
            "Implicit Sentiment Analysis (ISA)",
            "Supervised Contrastive Pre-Training (SCAPT)",
            "Aspect-Based Sentiment Analysis (ABSA)",
            "Large Language Model (LLM)",
            "Chain-of-Thought (CoT)",
            "Sentiment Analysis of Thinking (SAoT)",
            "Error-Type-Aware Reflection; ETC",
        ]:
            with self.subTest(redundant=redundant):
                self.assertNotIn(redundant, chapters)

    def test_thesis_uses_approved_replacement_figures(self):
        chapter4 = read_thesis_file("chapter4.tex")
        chapter5 = read_thesis_file("chapter5.tex")
        expected = {
            "Images/Chapter4/ch4_proposed_pipeline.png": (
                chapter4,
                "6fce01e625bb9739af2378427ebb58e4186d805010146540f117260051975f30",
            ),
            "Images/Chapter5/ch5_direct_vs_final_confusion.png": (
                chapter5,
                "c839b7579afb79323d6c90f1b69b1195fb01d2fe358bbc5c0792af71900cb949",
            ),
            "Images/Chapter5/ch5_gemini_direct_vs_selected_confusion.png": (
                chapter5,
                "16745f63d46406e4d4db2fe42edc0fccd5c57deae8e6fd8ae3b68aa6b8222d58",
            ),
        }

        for relative_path, (chapter, expected_hash) in expected.items():
            with self.subTest(relative_path=relative_path):
                self.assertIn(relative_path, chapter)
                actual_hash = hashlib.sha256(
                    (THESIS_DIR / relative_path).read_bytes()
                ).hexdigest()
                self.assertEqual(expected_hash, actual_hash)

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
            r"\fanum{۰٫۷}",
            "سه نمونه‌گیری درون هر پیش‌بینی و رأی اکثریت",
            "run_final_pipeline.py",
            "generate_thesis_result_figures.py",
            "build_thesis.ps1",
            "مدل زبانی را دوباره اجرا نمی‌کند",
        ]:
            self.assertIn(expected, appendix)
        for unrelated in ["کد میپل", "DifferentialGeometry", "DGsetup"]:
            self.assertNotIn(unrelated, appendix)
        self.assertNotRegex(appendix, r"\\lr\{(?!\\texttt\{)[^}]*_")

    def test_appendix_matches_implemented_alignment_key(self):
        appendix = read_thesis_file("appendix1.tex")
        self.assertIn(
            r"[id, source\_sentence\_id, sentence, target, from, to, polarity]",
            appendix,
        )
        self.assertNotIn("بازۀ هدف، دامنه و بخش داده", appendix)

    def test_appendix_describes_sc3_as_within_prediction_sampling(self):
        appendix = read_thesis_file("appendix1.tex")
        self.assertIn("سه نمونه‌گیری درون هر پیش‌بینی و رأی اکثریت", appendix)
        self.assertNotIn("سه مسیر استدلالی، سه اجرا و رأی اکثریت", appendix)

    def test_selector_comparison_methods_are_explained_before_table(self):
        text = (THESIS_DIR / "chapter5.tex").read_text(encoding="utf-8")
        table_pos = text.index(r"\label{tab:ch5-selector-comparison}")
        for phrase in [
            "اوراکل انتخاب منبع",
            "کران بالای تحلیلی",
            "هر زوج نمونه و منبع",
            "بیشترین امتیاز",
            "انتخاب‌گر فراسطح مبتنی بر درخت",
            "تقسیم داخلی آموزش",
        ]:
            self.assertGreaterEqual(text[:table_pos].find(phrase), 0)

    def test_appendix_is_clean_reproduction_guide(self):
        text = (THESIS_DIR / "appendix1.tex").read_text(encoding="utf-8")
        for phrase in [
            "پیوست: راهنمای بازتولید آزمایش‌ها",
            "داده و کلید نمونه",
            "مدل‌ها و تنظیمات تولید",
            "مراحل بازتولید",
            "نیاز به اجرای مدل",
            "run_final_pipeline.py",
            "generate_thesis_result_figures.py",
            r"\fanum{۰٫۷}",
            "۹۰ نمونه",
        ]:
            self.assertIn(phrase, text)
        self.assertNotIn("۲۴۰", text)

    def test_appendix_uses_repo_root_output_paths(self):
        text = (THESIS_DIR / "appendix1.tex").read_text(encoding="utf-8")
        self.assertNotRegex(text, r"\\path\{[^}]*[\u0600-\u06ff][^}]*\}")
        self.assertNotIn(r"\path{_thesis_template_of_Amirkabir", text)
        self.assertNotIn(r"\path{scripts/build_thesis.ps1}", text)
        self.assertIn(r"\lr{\texttt{scripts/build\_thesis.ps1}}", text)
        self.assertIn(r"\newcommand{\pathus}", text)
        self.assertIn(r"\newcommand{\thesisdirfa}", text)
        self.assertIn("قالب", text)
        self.assertIn("تمپلیت", text)
        self.assertIn("پایان", text)
        self.assertIn("نامه", text)
        self.assertIn("امیرکبیر", text)

        for latin_path_line in [
            r"\lr{\texttt{\_thesis\_template\_}}",
            r"\lr{\texttt{of\_Amirkabir/}}",
            r"\lr{\texttt{Images/Chapter5}}",
            r"\lr{\texttt{AUTthesis.pdf}}",
        ]:
            self.assertIn(latin_path_line, text)

        for standalone_claim in [
            r"& \path{Images/Chapter5} &",
            r"& \path{AUTthesis.pdf} &",
        ]:
            self.assertNotIn(standalone_claim, text)

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
        self.assertNotIn("Reasoning Implicit Sentiment with Chain-of-Thought Prompting", prose)
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

    def test_persian_thesis_numbers_are_direction_safe(self):
        thesis_files = [
            "fa_title.tex",
            "chapter1.tex",
            "chapter4.tex",
            "chapter5.tex",
            "chapter6.tex",
            "appendix1.tex",
        ]
        combined = "\n".join(read_thesis_file(name) for name in thesis_files)

        self.assertNotIn(r"\lr{\setpersianfont", combined)
        self.assertNotRegex(combined, r"[۰-۹]+/[۰-۹]+")
        self.assertIn(r"\fanum{۰٫۷}", combined)
        self.assertIn("۹۰ نمون", combined)

    def test_shared_gemini_subset_is_consistently_ninety(self):
        for name in ["fa_title.tex", "en-abstract.tex", "appendix1.tex"]:
            text = read_thesis_file(name)
            self.assertNotIn("۲۴۰", text)
            self.assertNotIn("240 examples", text)

        chapter5 = read_thesis_file("chapter5.tex")
        self.assertIn("۲۴۰ نمونه", chapter5)
        self.assertIn(
            "۱۵۰ نمونۀ آموزش فقط برای تنظیم سیاست انتخاب به کار می‌روند و تمام سنجه‌ها، "
            "شکل‌ها و ادعاهای مقایسه‌ای این زیربخش فقط بر ۹۰ نمونۀ آزمون مشترک "
            "(۳۰ نمونه از هر کلاس) گزارش می‌شوند.",
            chapter5,
        )

    def test_chapter4_documents_real_implementation_components(self):
        text = (THESIS_DIR / "chapter4.tex").read_text(encoding="utf-8")
        required = [
            "ساختار پیاده‌سازی سامانه",
            "run_tfidf_logreg_baseline.py",
            "PromptRunner", "OllamaPromptRunner", "HFSeq2SeqPromptRunner", "OpenAICompatiblePromptRunner",
            "THORPipeline", "SimpleReflectionPipeline", "ErrorTypeReflectionPipeline",
            "data_loader.py", "experiment_config.py", "controller.py",
            "apply_etc_policy.py", "run_meta_selector.py",
            "run_logistic_source_ranker.py", "evaluator.py", "final_results.py",
            "run_final_pipeline.py", "generate_thesis_result_figures.py",
            "اعتبارسنجی پنج‌لایه", "بازبرازش",
        ]
        for token in required:
            self.assertIn(token, text)
        self.assertIn("مرحله‌های مولد", text)
        self.assertNotIn("خروجی هر مرحله به‌صورت \\lr{CSV} شامل پاسخ خام", text)
        self.assertNotIn("هر مرحله خروجی \\lr{CSV} شامل کلید نمونه، پاسخ خام", text)


if __name__ == "__main__":
    unittest.main()
