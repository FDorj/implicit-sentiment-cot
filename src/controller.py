from src.evaluator import VALID_LABELS


CORRECTABLE_ERROR_TYPES = {
    "missed_implicit_negative",
    "missed_implicit_positive",
    "neutral_overinterpretation",
    "target_scope_shift",
    "aspect_opinion_mismatch",
    "reasoning_label_inconsistency",
}
NON_CORRECTIVE_ERROR_TYPES = {
    "no_error",
    "insufficient_evidence",
}
VALID_ERROR_TYPES = CORRECTABLE_ERROR_TYPES | NON_CORRECTIVE_ERROR_TYPES
VALID_CONFIDENCE = {"low", "medium", "high"}


def normalize_error_type(value: str) -> str:
    normalized = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in VALID_ERROR_TYPES else "insufficient_evidence"


def normalize_confidence(value: str) -> str:
    normalized = (value or "").strip().lower()
    return normalized if normalized in VALID_CONFIDENCE else "low"


def select_final_label(
    direct_label: str,
    thor_label: str,
    proposed_label: str,
    error_type: str,
    confidence: str = "low",
    fallback_policy: str = "direct",
    trust_no_error_on_disagreement: bool = False,
) -> tuple[str, str]:
    direct_valid = direct_label in VALID_LABELS
    thor_valid = thor_label in VALID_LABELS
    proposed_valid = proposed_label in VALID_LABELS
    normalized_error_type = normalize_error_type(error_type)
    normalized_confidence = normalize_confidence(confidence)

    if direct_valid and thor_valid and direct_label == thor_label:
        return thor_label, "agreement_keep_shared_label"

    if normalized_error_type in CORRECTABLE_ERROR_TYPES and proposed_valid:
        if normalized_confidence in {"medium", "high"}:
            return proposed_label, f"accept_{normalized_error_type}"

    if normalized_error_type == "no_error" and thor_valid and trust_no_error_on_disagreement:
        return thor_label, "diagnosis_no_error_keep_thor"

    if fallback_policy == "thor" and thor_valid:
        return thor_label, "fallback_keep_thor"

    if direct_valid:
        return direct_label, "fallback_use_direct"

    if thor_valid:
        return thor_label, "fallback_keep_thor"

    if proposed_valid:
        return proposed_label, "fallback_use_proposed"

    return "unknown", "fallback_unknown"
