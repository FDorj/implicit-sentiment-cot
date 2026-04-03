import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


VALID_LABELS = ["positive", "negative", "neutral"]


def evaluate_predictions(df: pd.DataFrame, gold_col: str = "polarity", pred_col: str = "prediction") -> dict:
    eval_df = df[df[pred_col].isin(VALID_LABELS)].copy()

    y_true = eval_df[gold_col].tolist()
    y_pred = eval_df[pred_col].tolist()

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    report = classification_report(
        y_true,
        y_pred,
        labels=VALID_LABELS,
        digits=4,
        zero_division=0,
    )

    return {
        "n_total": len(df),
        "n_eval": len(eval_df),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "report": report,
    }