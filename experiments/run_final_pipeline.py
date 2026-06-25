from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.final_results import (  # noqa: E402
    MethodSpec,
    compute_method_metrics,
    metrics_to_markdown,
    validate_final_chain,
    validation_to_text,
)


RESULTS_DIR = Path("results")

DIRECT_PATH = RESULTS_DIR / "direct_isa_predictions.csv"
THOR_SIMPLIFIED_PATH = RESULTS_DIR / "thor_isa_predictions.csv"
SIMPLE_REFLECTION_PATH = RESULTS_DIR / "simple_reflection_isa_predictions.csv"
ETC_STANDARD_PATH = RESULTS_DIR / "etc_isa_predictions.csv"
THOR_ORIGINALISH_SC3_PATH = RESULTS_DIR / "thor_originalish_sc3_isa_predictions.csv"
ETC_ORIGINALISH_SC3_PATH = RESULTS_DIR / "etc_thor_originalish_sc3_isa_predictions.csv"
FINAL_SELECTED_PATH = RESULTS_DIR / "etc_thor_originalish_sc3_selected_isa_predictions.csv"

FINAL_RESULTS_CSV = RESULTS_DIR / "final_results_table.csv"
FINAL_RESULTS_MD = RESULTS_DIR / "final_results_table.md"
FINAL_VALIDATION_TXT = RESULTS_DIR / "final_pipeline_validation.txt"


METHODS = [
    MethodSpec(
        name="Direct Qwen3 8B",
        path=DIRECT_PATH,
        pred_col="prediction",
        note="Zero-shot direct prompt baseline.",
    ),
    MethodSpec(
        name="THOR simplified",
        path=THOR_SIMPLIFIED_PATH,
        pred_col="prediction",
        note="Four-step THOR-style decomposition.",
    ),
    MethodSpec(
        name="Simple reflection",
        path=SIMPLE_REFLECTION_PATH,
        pred_col="reflection_prediction",
        note="One-pass review over simplified THOR trace.",
    ),
    MethodSpec(
        name="ETC standard",
        path=ETC_STANDARD_PATH,
        pred_col="controller_prediction",
        note="Error-type reflection and rule-based controller over simplified THOR.",
    ),
    MethodSpec(
        name="THOR original-ish SC3",
        path=THOR_ORIGINALISH_SC3_PATH,
        pred_col="prediction",
        note="Original-ish THOR prompts with three self-consistency samples.",
    ),
    MethodSpec(
        name="ETC over original-ish SC3",
        path=ETC_ORIGINALISH_SC3_PATH,
        pred_col="controller_prediction",
        note="Error-type reflection and controller over THOR original-ish SC3.",
    ),
    MethodSpec(
        name="Final selected pipeline",
        path=FINAL_SELECTED_PATH,
        pred_col="selected_prediction",
        note="Train-calibrated source selection over direct, THOR original-ish SC3, and diagnostic signals.",
    ),
]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    validation = validate_final_chain(
        direct_path=DIRECT_PATH,
        thor_path=THOR_ORIGINALISH_SC3_PATH,
        etc_path=ETC_ORIGINALISH_SC3_PATH,
        selected_path=FINAL_SELECTED_PATH,
    )
    metrics = compute_method_metrics(METHODS)

    metrics.to_csv(FINAL_RESULTS_CSV, index=False)
    FINAL_RESULTS_MD.write_text(metrics_to_markdown(metrics), encoding="utf-8")
    FINAL_VALIDATION_TXT.write_text(validation_to_text(validation), encoding="utf-8")

    print("Final pipeline summary written.")
    print(f"Metrics CSV: {FINAL_RESULTS_CSV}")
    print(f"Metrics Markdown: {FINAL_RESULTS_MD}")
    print(f"Validation: {FINAL_VALIDATION_TXT}")
    print()
    print(metrics_to_markdown(metrics))
    print(validation_to_text(validation))


if __name__ == "__main__":
    main()
