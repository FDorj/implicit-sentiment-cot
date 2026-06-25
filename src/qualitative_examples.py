from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.utils import normalize_label


GROUP_ORDER = [
    "gain_vs_direct",
    "loss_vs_direct",
    "both_correct",
    "both_wrong",
]


def add_direct_comparison_group(
    df: pd.DataFrame,
    gold_col: str = "polarity",
    direct_col: str = "direct_prediction",
    final_col: str = "selected_prediction",
) -> pd.DataFrame:
    """Label each row by how the final pipeline compares with Direct."""
    required = [gold_col, direct_col, final_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    labeled = df.copy()
    gold = labeled[gold_col].map(normalize_label)
    direct = labeled[direct_col].map(normalize_label)
    final = labeled[final_col].map(normalize_label)

    labeled["direct_correct"] = direct == gold
    labeled["final_correct"] = final == gold

    conditions = [
        (~labeled["direct_correct"]) & labeled["final_correct"],
        labeled["direct_correct"] & (~labeled["final_correct"]),
        labeled["direct_correct"] & labeled["final_correct"],
    ]
    choices = ["gain_vs_direct", "loss_vs_direct", "both_correct"]

    labeled["direct_comparison_group"] = "both_wrong"
    for condition, choice in zip(conditions, choices):
        labeled.loc[condition, "direct_comparison_group"] = choice

    return labeled


def select_qualitative_examples(
    df: pd.DataFrame,
    per_group: int = 8,
    split: str = "test",
    groups: Iterable[str] = ("gain_vs_direct", "loss_vs_direct"),
) -> pd.DataFrame:
    """Select stable qualitative examples from the final saved pipeline."""
    if per_group <= 0:
        raise ValueError("per_group must be positive")

    selected_groups = list(groups)
    labeled = add_direct_comparison_group(df)

    if "split" in labeled.columns and split is not None:
        labeled = labeled[labeled["split"].astype(str).str.lower() == split.lower()]

    group_rank = {group: rank for rank, group in enumerate(selected_groups)}
    examples = labeled[labeled["direct_comparison_group"].isin(selected_groups)].copy()
    examples["_group_rank"] = examples["direct_comparison_group"].map(group_rank)

    sort_cols = ["_group_rank"]
    for col in ["domain", "polarity", "id"]:
        if col in examples.columns:
            sort_cols.append(col)

    examples = examples.sort_values(sort_cols, kind="mergesort")
    examples = examples.groupby("direct_comparison_group", sort=False, group_keys=False).head(per_group)

    return examples.drop(columns=["_group_rank"]).reset_index(drop=True)


def comparison_summary(df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    labeled = add_direct_comparison_group(df)
    if "split" in labeled.columns and split is not None:
        labeled = labeled[labeled["split"].astype(str).str.lower() == split.lower()]

    counts = labeled["direct_comparison_group"].value_counts()
    return pd.DataFrame(
        {
            "direct_comparison_group": GROUP_ORDER,
            "count": [int(counts.get(group, 0)) for group in GROUP_ORDER],
        }
    )


def _cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\r\n", " ").replace("\n", " ")
    return text.replace("|", "\\|").strip()


def examples_to_markdown(
    examples: pd.DataFrame,
    summary: pd.DataFrame | None = None,
    title: str = "Final Pipeline Qualitative Examples",
) -> str:
    lines = [f"# {title}", ""]

    if summary is not None:
        lines.extend(["## Test Split Comparison Counts", "", "| Group | Count |", "| --- | ---: |"])
        for _, row in summary.iterrows():
            lines.append(f"| {_cell(row['direct_comparison_group'])} | {_cell(row['count'])} |")
        lines.append("")

    if examples.empty:
        lines.append("No examples matched the requested filters.")
        return "\n".join(lines) + "\n"

    display_cols = [
        "direct_comparison_group",
        "id",
        "domain",
        "target",
        "polarity",
        "direct_prediction",
        "thor_prediction",
        "diagnostic_label",
        "error_type",
        "diagnostic_confidence",
        "selected_source",
        "selected_prediction",
        "sentence",
    ]
    display_cols = [col for col in display_cols if col in examples.columns]

    lines.extend(["## Selected Examples", ""])
    for group in GROUP_ORDER:
        group_rows = examples[examples["direct_comparison_group"] == group]
        if group_rows.empty:
            continue

        lines.extend([f"### {group}", ""])
        for _, row in group_rows.iterrows():
            label = f"id={_cell(row.get('id', ''))}, domain={_cell(row.get('domain', ''))}"
            lines.extend([f"#### {label}", ""])
            for col in display_cols:
                lines.append(f"- {col}: {_cell(row.get(col, ''))}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
