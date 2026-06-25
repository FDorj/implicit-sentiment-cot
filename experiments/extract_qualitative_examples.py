from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qualitative_examples import (  # noqa: E402
    comparison_summary,
    examples_to_markdown,
    select_qualitative_examples,
)


RESULTS_DIR = Path("results")
THESIS_NOTES_DIR = Path("thesis_notes")

FINAL_SELECTED_PATH = RESULTS_DIR / "etc_thor_originalish_sc3_selected_isa_predictions.csv"
QUALITATIVE_EXAMPLES_CSV = RESULTS_DIR / "final_qualitative_examples.csv"
QUALITATIVE_SUMMARY_TXT = RESULTS_DIR / "final_qualitative_summary.txt"
QUALITATIVE_EXAMPLES_MD = THESIS_NOTES_DIR / "final_qualitative_examples_fa.md"

PERSIAN_INTRO = """# نمونه‌های کیفی سیستم نهایی

این فایل چند نمونه از split تست را نشان می‌دهد که برای تحلیل کیفی و دفاع مناسب‌اند.
گروه `gain_vs_direct` یعنی Direct اشتباه کرده ولی سیستم نهایی درست پاسخ داده است.
گروه `loss_vs_direct` یعنی Direct درست بوده ولی سیستم نهایی اشتباه کرده است.

"""


def summary_to_text(summary_df: pd.DataFrame, examples_df: pd.DataFrame) -> str:
    lines = [
        "Final pipeline qualitative summary",
        f"source_file: {FINAL_SELECTED_PATH}",
        f"selected_examples: {len(examples_df)}",
        "",
        "test split comparison counts",
    ]
    for _, row in summary_df.iterrows():
        lines.append(f"{row['direct_comparison_group']}: {int(row['count'])}")

    lines.extend(["", "selected examples by group"])
    selected_counts = examples_df["direct_comparison_group"].value_counts()
    for group in summary_df["direct_comparison_group"].tolist():
        lines.append(f"{group}: {int(selected_counts.get(group, 0))}")

    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_NOTES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FINAL_SELECTED_PATH)
    summary = comparison_summary(df, split="test")
    examples = select_qualitative_examples(df, per_group=8, split="test")

    examples.to_csv(QUALITATIVE_EXAMPLES_CSV, index=False)
    QUALITATIVE_SUMMARY_TXT.write_text(summary_to_text(summary, examples), encoding="utf-8")
    QUALITATIVE_EXAMPLES_MD.write_text(
        PERSIAN_INTRO
        + examples_to_markdown(
            examples,
            summary=summary,
            title="Final Pipeline Qualitative Examples",
        ),
        encoding="utf-8",
    )

    print("Qualitative examples written.")
    print(f"Examples CSV: {QUALITATIVE_EXAMPLES_CSV}")
    print(f"Examples Markdown: {QUALITATIVE_EXAMPLES_MD}")
    print(f"Summary: {QUALITATIVE_SUMMARY_TXT}")
    print()
    print(summary_to_text(summary, examples))


if __name__ == "__main__":
    main()
