
import pandas as pd


def assign_customer_value(
    df,
    value_col="totalcharges",
    low_quantile=0.33,
    high_quantile=0.67,
):
    """
    Categorize customers into Low, Medium and High value
    based on their total charges.

    Args:
        df (pd.DataFrame): Customer dataset.
        value_col (str): Column used to estimate customer value.
        low_quantile (float): Lower percentile threshold.
        high_quantile (float): Upper percentile threshold.

    Returns:
        tuple:
            - DataFrame with customer_value column.
            - Dictionary containing value thresholds.
    """

    if value_col not in df.columns:
        raise ValueError(
            f"Column '{value_col}' not found in dataset."
        )

    if not 0 < low_quantile < high_quantile < 1:
        raise ValueError(
            "Quantiles must satisfy 0 < low < high < 1."
        )

    result = df.copy()

    values = pd.to_numeric(
        result[value_col],
        errors="coerce"
    )

    if (values.dropna() < 0).any():
        raise ValueError(
            "Customer value cannot contain negative charges."
        )

    if values.notna().sum() == 0:
        raise ValueError(
            "No valid customer value data available."
        )

    # Calculate percentile-based thresholds
    low_threshold = values.quantile(low_quantile)
    high_threshold = values.quantile(high_quantile)

    # Assign customer value categories
    result["customer_value"] = "Unknown"

    valid = values.notna()

    result.loc[
        valid & (values <= low_threshold),
        "customer_value"
    ] = "Low"

    result.loc[
        valid
        & (values > low_threshold)
        & (values <= high_threshold),
        "customer_value"
    ] = "Medium"

    result.loc[
        valid & (values > high_threshold),
        "customer_value"
    ] = "High"

    thresholds = {
        "value_column": value_col,
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
    }

    return result, thresholds
