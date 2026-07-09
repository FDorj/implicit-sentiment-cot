from pathlib import Path
import os
import sys

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import evaluate_predictions  # noqa: E402
from src.final_results import KEY_COLS  # noqa: E402


ETC_PATH = os.getenv(
    "LOGISTIC_SOURCE_RANKER_ETC_PATH",
    str(Path("results") / "etc_thor_originalish_sc3_isa_predictions.csv"),
)
THOR_SC_PATH = os.getenv(
    "LOGISTIC_SOURCE_RANKER_THOR_SC_PATH",
    str(Path("results") / "thor_originalish_sc3_isa_predictions.csv"),
)
CURRENT_SELECTED_PATH = os.getenv(
    "LOGISTIC_SOURCE_RANKER_CURRENT_SELECTED_PATH",
    str(Path("results") / "etc_thor_originalish_sc3_selected_isa_predictions.csv"),
)
OUTPUT_PREDICTIONS_PATH = os.getenv(
    "LOGISTIC_SOURCE_RANKER_OUTPUT_PATH",
    str(Path("results") / "logistic_source_ranker_predictions.csv"),
)
OUTPUT_METRICS_PATH = os.getenv(
    "LOGISTIC_SOURCE_RANKER_METRICS_PATH",
    str(Path("results") / "logistic_source_ranker_metrics.txt"),
)
OUTPUT_ABLATION_PATH = os.getenv(
    "LOGISTIC_SOURCE_RANKER_ABLATION_PATH",
    str(Path("results") / "logistic_source_ranker_ablation_metrics.csv"),
)
LOGISTIC_C = float(os.getenv("LOGISTIC_SOURCE_RANKER_C", "1.0"))

SOURCE_ORDER = ["direct", "thor", "diagnostic"]
SOURCE_TO_COLUMN = {
    "direct": "direct_prediction",
    "thor": "thor_prediction",
    "diagnostic": "diagnostic_label",
}
REQUIRED_COLUMNS = [
    "polarity",
    "split",
    "domain",
    "direct_prediction",
    "thor_prediction",
    "diagnostic_label",
    "controller_prediction",
    "error_type",
    "diagnostic_confidence",
]


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def parse_vote_counts(raw_value: object) -> dict[str, int]:
    text = str(raw_value or "").strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return {}

    counts: dict[str, int] = {}
    for part in text.split(";"):
        if ":" not in part:
            continue
        label, value = part.split(":", 1)
        try:
            counts[label.strip()] = int(value.strip())
        except ValueError:
            continue
    return counts


def vote_features(raw_value: object) -> dict[str, float | int]:
    counts = parse_vote_counts(raw_value)
    if not counts:
        return {
            "sc_top_vote_count": 0,
            "sc_vote_margin": 0,
            "sc_top_vote_share": 0.0,
            "sc_vote_unique_labels": 0,
        }

    sorted_counts = sorted(counts.values(), reverse=True)
    total = sum(sorted_counts)
    top_count = sorted_counts[0]
    second_count = sorted_counts[1] if len(sorted_counts) > 1 else 0
    return {
        "sc_top_vote_count": top_count,
        "sc_vote_margin": top_count - second_count,
        "sc_top_vote_share": top_count / total if total else 0.0,
        "sc_vote_unique_labels": len(counts),
    }


def add_vote_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    vote_series = output["sc_vote_counts"] if "sc_vote_counts" in output.columns else pd.Series("", index=output.index)
    vote_df = pd.DataFrame([vote_features(value) for value in vote_series], index=output.index)
    for column in vote_df.columns:
        output[column] = vote_df[column]
    return output


def build_candidate_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = add_vote_features(df)
    parts = []

    for source in SOURCE_ORDER:
        candidate_rows = pd.DataFrame(index=rows.index)
        candidate_predictions = rows[SOURCE_TO_COLUMN[source]].astype(str)

        candidate_rows["row_index"] = rows.index
        candidate_rows["candidate_source"] = source
        candidate_rows["candidate_prediction"] = candidate_predictions
        candidate_rows["direct_prediction"] = rows["direct_prediction"].astype(str)
        candidate_rows["thor_prediction"] = rows["thor_prediction"].astype(str)
        candidate_rows["diagnostic_label"] = rows["diagnostic_label"].astype(str)
        candidate_rows["controller_prediction"] = rows.get("controller_prediction", "").astype(str)
        candidate_rows["error_type"] = rows.get("error_type", "").astype(str)
        candidate_rows["diagnostic_confidence"] = rows.get("diagnostic_confidence", "").astype(str)
        candidate_rows["domain"] = rows.get("domain", "").astype(str)
        candidate_rows["diagnostic_triggered"] = rows.get("diagnostic_triggered", False).map(
            lambda value: yes_no(str(value).strip().lower() in {"1", "true", "yes"})
        )
        candidate_rows["direct_thor_agreement"] = (
            rows["direct_prediction"] == rows["thor_prediction"]
        ).map(yes_no)
        candidate_rows["direct_diagnostic_agreement"] = (
            rows["direct_prediction"] == rows["diagnostic_label"]
        ).map(yes_no)
        candidate_rows["thor_diagnostic_agreement"] = (
            rows["thor_prediction"] == rows["diagnostic_label"]
        ).map(yes_no)
        candidate_rows["candidate_agrees_direct"] = (
            candidate_predictions == rows["direct_prediction"].astype(str)
        ).map(yes_no)
        candidate_rows["candidate_agrees_thor"] = (
            candidate_predictions == rows["thor_prediction"].astype(str)
        ).map(yes_no)
        candidate_rows["candidate_agrees_diagnostic"] = (
            candidate_predictions == rows["diagnostic_label"].astype(str)
        ).map(yes_no)
        candidate_rows["source_index"] = SOURCE_ORDER.index(source)
        for column in [
            "sc_top_vote_count",
            "sc_vote_margin",
            "sc_top_vote_share",
            "sc_vote_unique_labels",
        ]:
            candidate_rows[column] = rows[column]

        candidate_rows["candidate_correct"] = (
            rows[SOURCE_TO_COLUMN[source]].values == rows["polarity"].values
        )
        parts.append(candidate_rows)

    return pd.concat(parts, ignore_index=True)


def feature_columns(candidate_rows: pd.DataFrame) -> tuple[list[str], list[str]]:
    ignored = {"row_index", "candidate_correct"}
    numeric_features = [
        column
        for column in candidate_rows.columns
        if column not in ignored and is_numeric_dtype(candidate_rows[column])
    ]
    categorical_features = [
        column
        for column in candidate_rows.columns
        if column not in ignored and column not in numeric_features
    ]
    return categorical_features, numeric_features


def make_logistic_ranker(categorical_features: list[str], numeric_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", StandardScaler(), numeric_features),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    C=LOGISTIC_C,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def select_sources_from_scores(
    df: pd.DataFrame,
    candidate_scores: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    selected_predictions = {}
    selected_sources = {}

    source_rank = {source: index for index, source in enumerate(SOURCE_ORDER)}
    for row_index, group in candidate_scores.groupby("row_index"):
        best = group.sort_values(
            by=["score", "candidate_source"],
            ascending=[False, True],
            key=lambda values: values.map(source_rank) if values.name == "candidate_source" else values,
        ).iloc[0]
        selected_predictions[row_index] = best["candidate_prediction"]
        selected_sources[row_index] = best["candidate_source"]

    return pd.Series(selected_predictions).sort_index(), pd.Series(selected_sources).sort_index()


def predictions_from_sources(df: pd.DataFrame, sources: list[str]) -> list[str]:
    return [
        row[SOURCE_TO_COLUMN[source]]
        for source, (_, row) in zip(sources, df.iterrows())
    ]


def oracle_select_sources(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    predictions = []
    sources = []
    for _, row in df.iterrows():
        selected_source = "direct"
        for source in SOURCE_ORDER:
            if row[SOURCE_TO_COLUMN[source]] == row["polarity"]:
                selected_source = source
                break
        sources.append(selected_source)
        predictions.append(row[SOURCE_TO_COLUMN[selected_source]])
    return predictions, sources


def metric_rows(df: pd.DataFrame, methods: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    for method, pred_col, note in methods:
        for split in ["overall", "train", "test"]:
            split_df = df if split == "overall" else df[df["split"].astype(str).str.lower() == split].copy()
            if split_df.empty:
                continue
            metrics = evaluate_predictions(split_df, gold_col="polarity", pred_col=pred_col)
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "n_total": metrics["n_total"],
                    "n_eval": metrics["n_eval"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "prediction_column": pred_col,
                    "note": note,
                }
            )
    return pd.DataFrame(rows)


def attach_sc_vote_columns(df: pd.DataFrame, thor_sc_path: str) -> pd.DataFrame:
    path = Path(thor_sc_path)
    if not path.exists() or "sc_vote_counts" in df.columns:
        return df

    thor_df = pd.read_csv(path)
    required = [*KEY_COLS, "sc_vote_counts", "sc_labels"]
    if not all(column in thor_df.columns for column in required):
        return df
    return df.merge(thor_df[required], on=KEY_COLS, how="left")


def attach_current_selected(df: pd.DataFrame, current_selected_path: str) -> pd.DataFrame:
    path = Path(current_selected_path)
    if not path.exists():
        return df

    selected_df = pd.read_csv(path)
    required = [*KEY_COLS, "selected_prediction"]
    if not all(column in selected_df.columns for column in required):
        return df

    columns = [*KEY_COLS, "selected_prediction"]
    if "selected_source" in selected_df.columns:
        columns.append("selected_source")
    return df.merge(selected_df[columns], on=KEY_COLS, how="left")


def main() -> None:
    Path("results").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ETC_PATH)
    require_columns(df, REQUIRED_COLUMNS, ETC_PATH)
    df = attach_sc_vote_columns(df, THOR_SC_PATH)
    df = attach_current_selected(df, CURRENT_SELECTED_PATH)

    candidate_rows = build_candidate_rows(df)
    categorical_features, numeric_features = feature_columns(candidate_rows)
    train_row_indices = df[df["split"].astype(str).str.lower() == "train"].index
    train_mask = candidate_rows["row_index"].isin(train_row_indices)

    model = make_logistic_ranker(categorical_features, numeric_features)
    model.fit(
        candidate_rows.loc[train_mask, categorical_features + numeric_features],
        candidate_rows.loc[train_mask, "candidate_correct"],
    )

    scored_candidates = candidate_rows.copy()
    scored_candidates["score"] = model.predict_proba(
        candidate_rows[categorical_features + numeric_features]
    )[:, 1]

    selected_predictions, selected_sources = select_sources_from_scores(df, scored_candidates)
    oracle_predictions, oracle_sources = oracle_select_sources(df)

    output_df = df.copy()
    output_df["logistic_source_ranker_prediction"] = selected_predictions.reindex(output_df.index).values
    output_df["logistic_source_ranker_source"] = selected_sources.reindex(output_df.index).values
    output_df["oracle_source"] = oracle_sources
    output_df["oracle_prediction"] = oracle_predictions
    for source in SOURCE_ORDER:
        source_scores = scored_candidates[scored_candidates["candidate_source"] == source].set_index("row_index")["score"]
        output_df[f"logistic_{source}_score"] = source_scores.reindex(output_df.index).values

    output_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)

    methods = [
        ("direct_baseline", "direct_prediction", ""),
        ("thor_baseline", "thor_prediction", ""),
        ("diagnostic_label", "diagnostic_label", ""),
        ("controller_prediction", "controller_prediction", ""),
        ("oracle_source_upper_bound", "oracle_prediction", "uses gold labels to choose source"),
        (
            "logistic_source_ranker",
            "logistic_source_ranker_prediction",
            f"regularized logistic regression, C={LOGISTIC_C}",
        ),
    ]
    if "selected_prediction" in output_df.columns:
        methods.insert(4, ("current_final_selected", "selected_prediction", ""))

    summary_df = metric_rows(output_df, methods)
    summary_df.to_csv(OUTPUT_ABLATION_PATH, index=False)

    test_summary = summary_df[summary_df["split"] == "test"].sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
    )
    with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"ETC predictions: {ETC_PATH}\n")
        f.write(f"THOR SC predictions: {THOR_SC_PATH}\n")
        f.write(f"Current selected predictions: {CURRENT_SELECTED_PATH}\n")
        f.write(f"Output predictions: {OUTPUT_PREDICTIONS_PATH}\n")
        f.write(f"Output metrics CSV: {OUTPUT_ABLATION_PATH}\n")
        f.write(f"Logistic C: {LOGISTIC_C}\n")
        f.write(f"Training rows: {len(train_row_indices)} samples, {int(train_mask.sum())} source candidates\n")
        f.write("\nFeature columns\n")
        f.write(f"categorical: {categorical_features}\n")
        f.write(f"numeric: {numeric_features}\n\n")
        f.write("summary metrics\n")
        f.write(
            summary_df.sort_values(["split", "macro_f1", "accuracy"], ascending=[True, False, False])
            .to_string(index=False, float_format=lambda value: f"{value:.6f}")
        )
        f.write("\n\nlogistic source counts\n")
        f.write(output_df["logistic_source_ranker_source"].value_counts().to_string())
        f.write("\n\noracle source counts\n")
        f.write(output_df["oracle_source"].value_counts().to_string())
        f.write("\n")

    print("Done.")
    print(f"Saved logistic source-ranker predictions to: {OUTPUT_PREDICTIONS_PATH}")
    print(f"Saved logistic source-ranker metrics to: {OUTPUT_METRICS_PATH}")
    print(
        test_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )


if __name__ == "__main__":
    main()
