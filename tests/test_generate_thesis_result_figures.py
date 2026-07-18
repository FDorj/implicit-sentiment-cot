import unittest
from pathlib import Path

from experiments.generate_thesis_result_figures import (
    _format_persian_number,
    build_figures,
    load_confusion_data,
    load_gemini_confusion_data,
    load_main_test_results,
    load_selector_behavior,
    load_shared_subset_comparison,
    render_figure_tex,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = REPO_ROOT / ".test_tmp"
TEST_TMP_ROOT.mkdir(exist_ok=True)


def clean_test_output_dir(name: str) -> Path:
    output_dir = TEST_TMP_ROOT / name
    output_dir.mkdir(exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_file():
            child.unlink()
    return output_dir


class ThesisResultFigureDataTests(unittest.TestCase):
    def test_main_results_are_test_only_and_include_eight_methods(self):
        rows = load_main_test_results(REPO_ROOT)

        self.assertEqual(len(rows), 8)
        self.assertEqual({row["n_eval"] for row in rows}, {442})
        self.assertEqual(rows[0]["method"], "TF-IDF + Logistic Regression")
        self.assertEqual(rows[-1]["method"], "Final selected pipeline")
        self.assertAlmostEqual(rows[0]["accuracy"], 0.5475113122171946)
        self.assertAlmostEqual(rows[0]["macro_f1"], 0.5140820214979989)
        self.assertAlmostEqual(rows[-1]["macro_f1"], 0.7191194191688132)

    def test_main_result_order_places_originalish_thor_after_simplified_thor(self):
        rows = load_main_test_results(REPO_ROOT)

        self.assertEqual(
            [row["method"] for row in rows],
            [
                "TF-IDF + Logistic Regression",
                "Direct Qwen3 8B",
                "THOR simplified",
                "THOR original-ish SC3",
                "Simple reflection",
                "ETC standard",
                "ETC over original-ish SC3",
                "Final selected pipeline",
            ],
        )

    def test_confusion_data_aligns_442_test_rows_and_matches_expected_counts(self):
        result = load_confusion_data(REPO_ROOT)

        self.assertEqual(result["n"], 442)
        self.assertEqual(result["labels"], ["negative", "neutral", "positive"])
        self.assertEqual(
            result["direct_counts"],
            [[66, 13, 3], [59, 146, 43], [6, 18, 88]],
        )
        self.assertEqual(
            result["final_counts"],
            [[70, 9, 3], [51, 160, 37], [3, 19, 90]],
        )

    def test_gemini_confusion_data_aligns_shared_test_rows(self):
        result = load_gemini_confusion_data(REPO_ROOT)

        self.assertEqual(result["n"], 90)
        self.assertEqual(result["labels"], ["negative", "neutral", "positive"])
        self.assertEqual(
            result["direct_counts"],
            [[27, 2, 1], [3, 18, 9], [1, 1, 28]],
        )
        self.assertEqual(
            result["final_counts"],
            [[27, 2, 1], [3, 20, 7], [1, 3, 26]],
        )

    def test_selector_behavior_uses_one_test_population(self):
        result = load_selector_behavior(REPO_ROOT)

        self.assertEqual(result["n"], 442)
        self.assertEqual(
            result["source_counts"],
            {"direct": 409, "thor": 33, "diagnostic": 0},
        )
        self.assertEqual(
            result["transitions"],
            {"both_correct": 294, "gain": 26, "loss": 6, "both_wrong": 116},
        )

    def test_shared_subset_comparison_uses_exactly_90_test_examples(self):
        result = load_shared_subset_comparison(REPO_ROOT)

        self.assertEqual(result["n"], 90)
        self.assertEqual(
            list(result["macro_f1"]),
            ["Direct", "THOR SC3", "Validation-tuned selected"],
        )
        self.assertAlmostEqual(result["macro_f1"]["Direct"]["Qwen3 8B"], 0.7219030716298476)
        self.assertAlmostEqual(result["macro_f1"]["Direct"]["Gemini 2.5 Flash"], 0.8048858887817422)
        self.assertAlmostEqual(result["macro_f1"]["THOR SC3"]["Qwen3 8B"], 0.6064502303653702)
        self.assertAlmostEqual(result["macro_f1"]["THOR SC3"]["Gemini 2.5 Flash"], 0.7161092875378591)
        self.assertAlmostEqual(
            result["macro_f1"]["Validation-tuned selected"]["Qwen3 8B"],
            0.7219030716298476,
        )
        self.assertAlmostEqual(
            result["macro_f1"]["Validation-tuned selected"]["Gemini 2.5 Flash"],
            0.8083395429706904,
        )


class ThesisResultFigureRenderingTests(unittest.TestCase):
    def test_persian_number_formatter_uses_persian_digits_and_decimal_separator(self):
        self.assertEqual(_format_persian_number(0.7, 1), "۰٫۷")
        self.assertEqual(_format_persian_number(90), "۹۰")
        self.assertEqual(_format_persian_number(80.5, 1), "۸۰٫۵")
        self.assertNotIn("/", _format_persian_number(0.7, 1))

    def test_rendering_writes_five_complete_standalone_documents(self):
        paths = render_figure_tex(REPO_ROOT, clean_test_output_dir("render-docs-a"))

        self.assertEqual(
            {path.name for path in paths},
            {
                "ch5_qwen_full_test_methods.tex",
                "ch5_direct_vs_final_confusion.tex",
                "ch5_selector_behavior.tex",
                "ch5_qwen_gemini_shared_subset.tex",
                "ch5_gemini_direct_vs_selected_confusion.tex",
            },
        )
        for path in paths:
            content = path.read_text(encoding="utf-8")
            self.assertIn(r"\documentclass[tikz,border=4pt]{standalone}", content)
            self.assertIn(r"\usepackage{pgfplots}", content)
            self.assertIn(r"\usepackage{xepersian}", content)
            self.assertIn(r"\setdigitfont", content)
            self.assertIn(r"\pgfplotsset{compat=1.18}", content)
            self.assertIn("figure-id:", content)
            self.assertNotIn("nan", content.lower())
            self.assertNotIn("None", content)

    def test_sample_count_phrases_are_isolated_as_rtl_without_ltr_digits(self):
        paths = {
            path.stem: path
            for path in render_figure_tex(REPO_ROOT, clean_test_output_dir("render-docs-rtl"))
        }
        selector = paths["ch5_selector_behavior"].read_text(encoding="utf-8")
        comparison = paths["ch5_qwen_gemini_shared_subset"].read_text(encoding="utf-8")
        gemini_confusion = paths["ch5_gemini_direct_vs_selected_confusion"].read_text(
            encoding="utf-8"
        )

        self.assertIn(
            r"title={\RL{منبع منتخب در آزمون؛ تعداد نمونه‌ها: ۴۴۲}}",
            selector,
        )
        self.assertIn(
            r"{\RL{گذار وضعیت صحت؛ تعداد نمونه‌ها: ۴۴۲}}",
            selector,
        )
        self.assertIn(
            r"title={\RL{زیرمجموعۀ متوازن مشترک آزمون؛ تعداد نمونه‌ها: ۹۰}}",
            comparison,
        )
        self.assertIn(
            r"{\RL{زیرمجموعۀ متوازن مشترک آزمون؛ تعداد نمونه‌ها: ۹۰؛ شمار و درصدهای نرمال‌شدۀ سطری}}",
            gemini_confusion,
        )
        for content in [selector, comparison, gemini_confusion]:
            self.assertNotRegex(content, r"\\lr\{[^}]*[۰-۹]")

    def test_rendering_uses_scopes_and_fixed_axis_bounds_from_design(self):
        paths = {
            path.stem: path
            for path in render_figure_tex(REPO_ROOT, clean_test_output_dir("render-docs-b"))
        }

        main = paths["ch5_qwen_full_test_methods"].read_text(encoding="utf-8")
        self.assertIn("xmin=0.00", main)
        self.assertIn("xmax=0.80", main)
        self.assertNotIn("xmin=0.55", main)
        self.assertIn("point meta=explicit symbolic", main)
        self.assertNotIn(r"\pgfmathprintnumber", main)
        self.assertGreaterEqual(main.count("forget plot"), 2)
        self.assertIn("سامانۀ نهایی", main)
        self.assertIn("TF-IDF + Logistic Regression", main)
        self.assertIn(
            r"ytick={\lr{TF-IDF + Logistic Regression},پیش‌بینی مستقیم \lr{Qwen3 8B},"
            r"\lr{THOR} ساده‌شده,\lr{THOR SC3} سه‌اجرایی,"
            r"بازبینی ساده,کنترل‌گر \lr{ETC},کنترل‌گر \lr{ETC} روی \lr{THOR SC3},سامانۀ نهایی}",
            main,
        )
        self.assertIn("xlabel={امتیاز}", main)
        self.assertIn(r"\addlegendentry{دقت}", main)
        self.assertIn(r"\addlegendentry{اف‌یک ماکرو}", main)
        self.assertNotIn("original-ish", main)
        self.assertIn(r"\lr{TF-IDF + Logistic Regression}", main)
        self.assertIn(r"\lr{Qwen3 8B}", main)
        self.assertNotIn("ytick=data", main)

        confusion = paths["ch5_direct_vs_final_confusion"].read_text(encoding="utf-8")
        self.assertIn(r"۶۶\\۸۰٫۵\%", confusion)
        self.assertIn("۶۶", confusion)
        self.assertIn("۱۶۰", confusion)
        self.assertIn("نرمال‌سازی سطری", confusion)
        self.assertIn("برچسب واقعی", confusion)
        self.assertIn("برچسب پیش‌بینی‌شده", confusion)
        self.assertIn(r"\lr{Qwen3 8B}", confusion)
        self.assertIn("at (-2.05,1.35)", confusion)
        self.assertIn("at (4.55,1.35)", confusion)

        selector = paths["ch5_selector_behavior"].read_text(encoding="utf-8")
        self.assertIn("۴۰۹", selector)
        self.assertIn("۳۳", selector)
        self.assertIn("مستقیم نادرست / نهایی درست", selector)
        self.assertIn("منبع منتخب در آزمون", selector)
        self.assertIn("تعداد نمونه‌ها: ۴۴۲", selector)

        comparison = paths["ch5_qwen_gemini_shared_subset"].read_text(encoding="utf-8")
        self.assertIn("point meta=explicit symbolic", comparison)
        self.assertIn("[۰٫۷۲۲]", comparison)
        self.assertIn(
            "yticklabels={۰,۰٫۱,۰٫۲,۰٫۳,۰٫۴,۰٫۵,۰٫۶,۰٫۷,۰٫۸,۰٫۹}",
            comparison,
        )
        self.assertNotIn(r"\pgfmathprintnumber", comparison)
        self.assertIn("ymin=0.00", comparison)
        self.assertIn("ymax=0.90", comparison)
        self.assertNotIn("ymin=0.55", comparison)
        self.assertIn("زیرمجموعۀ متوازن مشترک آزمون", comparison)
        self.assertIn("اف‌یک ماکروی آزمون", comparison)
        self.assertIn(r"\lr{Qwen3 8B}", comparison)
        self.assertIn(r"\lr{Gemini 2.5 Flash}", comparison)
        self.assertIn("تعداد نمونه‌ها: ۹۰", comparison)
        self.assertIn("axis description cs:0.5,1.16", comparison)
        self.assertNotIn("rel axis cs:0.015,0.985", comparison)

        gemini_confusion = paths["ch5_gemini_direct_vs_selected_confusion"].read_text(
            encoding="utf-8"
        )
        self.assertIn("تعداد نمونه‌ها: ۹۰", gemini_confusion)
        self.assertIn(r"پیش‌بینی مستقیم \lr{Gemini 2.5 Flash}", gemini_confusion)
        self.assertIn(r"سیاست منتخب \lr{Gemini}", gemini_confusion)
        self.assertIn("زیرمجموعۀ متوازن مشترک آزمون", gemini_confusion)
        self.assertIn("۲۷", gemini_confusion)
        self.assertIn("۲۰", gemini_confusion)

    def test_shared_subset_ybar_coordinates_put_symbolic_x_first(self):
        paths = {
            path.stem: path
            for path in render_figure_tex(REPO_ROOT, clean_test_output_dir("render-docs-c"))
        }
        comparison = paths["ch5_qwen_gemini_shared_subset"].read_text(encoding="utf-8")

        self.assertIn("(مستقیم,0.721903)", comparison)
        self.assertIn(r"({\lr{THOR SC3}},0.606450)", comparison)
        self.assertIn("({سیاست منتخب},0.721903)", comparison)
        self.assertNotIn("(0.721903,مستقیم)", comparison)


class ThesisResultFigureBuildTests(unittest.TestCase):
    def test_build_produces_five_nonempty_pdfs_and_pngs(self):
        output_dir = clean_test_output_dir("built-assets")

        assets = build_figures(REPO_ROOT, output_dir)

        self.assertEqual(len(assets), 10)
        self.assertEqual(len([path for path in assets if path.suffix == ".pdf"]), 5)
        self.assertEqual(len([path for path in assets if path.suffix == ".png"]), 5)
        for path in assets:
            self.assertTrue(path.is_file(), path)
            self.assertGreater(path.stat().st_size, 1000, path)
            signature = path.read_bytes()[:8]
            if path.suffix == ".pdf":
                self.assertTrue(signature.startswith(b"%PDF"), path)
            else:
                self.assertEqual(signature, b"\x89PNG\r\n\x1a\n", path)


if __name__ == "__main__":
    unittest.main()
