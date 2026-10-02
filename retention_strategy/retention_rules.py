import pandas as pd

DEFAULT_RISK_THRESHOLDS = {
    "low": 0.40,
    "medium": 0.70,
}


def assign_churn_risk(
    df,
    probability_col="churn_probability",
    low_threshold=0.40,
    high_threshold=0.70,
):
    """
    Assign configurable churn-risk categories
    based on predicted churn probabilities.
    """
    if probability_col not in df.columns:
        raise ValueError(
            f"Missing column: {probability_col}"
        )

    if not 0 <= low_threshold < high_threshold <= 1:
        raise ValueError("Invalid risk thresholds.")

    result = df.copy()
    probability = result[probability_col]

    if probability.isna().any() or not probability.between(0, 1).all():
        raise ValueError(
            "Churn probabilities must be non-null and between 0 and 1."
        )

    result["risk_category"] = pd.cut(
        probability,
        bins=[-0.001, low_threshold, high_threshold, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    return result
