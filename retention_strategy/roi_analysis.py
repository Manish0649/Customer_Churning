
import pandas as pd


def estimate_retention_roi(
    df,
    retention_uplift=0.10,
    expected_retention_months=3,
    contribution_margin=0.40,
):
    """
    Estimate retention campaign costs, incremental
    benefits and ROI for customers.

    Args:
        df (pd.DataFrame): Customer data containing
            churn_probability, monthlycharges and reward_cost.
        retention_uplift (float): Assumed incremental
            retention uplift from the campaign.
        expected_retention_months (int): Expected additional
            months retained.
        contribution_margin (float): Revenue contribution
            margin after variable costs.

    Returns:
        tuple:
            - Customer-level DataFrame with ROI calculations.
            - Dictionary containing overall ROI summary.

    Note:
        Results are estimates based on assumptions,
        not measured campaign outcomes.
    """

    # Validate required columns
    required_columns = {
        "churn_probability",
        "monthlycharges",
        "reward_cost",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Validate parameters
    if not 0 <= retention_uplift <= 1:
        raise ValueError(
            "Retention uplift must be between 0 and 1."
        )

    if expected_retention_months < 0:
        raise ValueError(
            "Retention months cannot be negative."
        )

    if not 0 <= contribution_margin <= 1:
        raise ValueError(
            "Contribution margin must be between 0 and 1."
        )

    result = df.copy()

    # Convert input columns to numeric
    for col in [
        "churn_probability",
        "monthlycharges",
        "reward_cost",
    ]:
        result[col] = pd.to_numeric(
            result[col], errors="coerce"
        )

    # Validate customer-level data
    if result[
        ["churn_probability", "monthlycharges", "reward_cost"]
    ].isna().any().any():
        raise ValueError(
            "ROI input columns cannot contain missing or "
            "non-numeric values."
        )

    if not result["churn_probability"].between(0, 1).all():
        raise ValueError(
            "Churn probabilities must be between 0 and 1."
        )

    if (result["monthlycharges"] < 0).any():
        raise ValueError(
            "Monthly charges cannot be negative."
        )

    if (result["reward_cost"] < 0).any():
        raise ValueError(
            "Reward costs cannot be negative."
        )

    # Step 1: Estimate incremental customers retained
    result["expected_incremental_saves"] = (
        result["churn_probability"] * retention_uplift
    )

    # Step 2: Estimate expected reward expenditure
    # Weighted by the probability of churn
    result["expected_reward_cost"] = (
        result["churn_probability"] * result["reward_cost"]
    )

    # Step 3: Estimate incremental contribution benefit
    result["expected_incremental_benefit"] = (
        result["expected_incremental_saves"]
        * result["monthlycharges"]
        * expected_retention_months
        * contribution_margin
    )

    # Step 4: Calculate expected net benefit
    result["expected_net_benefit"] = (
        result["expected_incremental_benefit"]
        - result["expected_reward_cost"]
    )

    # Step 5: Calculate overall campaign metrics
    total_customers = len(result)

    total_reward_cost = result[
        "expected_reward_cost"
    ].sum()

    total_incremental_benefit = result[
        "expected_incremental_benefit"
    ].sum()

    total_net_benefit = (
        total_incremental_benefit - total_reward_cost
    )

    # ROI is undefined when campaign cost is zero
    roi_percent = (
        (total_net_benefit / total_reward_cost) * 100
        if total_reward_cost > 0
        else None
    )

    summary = {
        "total_customers": total_customers,
        "expected_incremental_saves": result[
            "expected_incremental_saves"
        ].sum(),
        "total_reward_cost": total_reward_cost,
        "total_incremental_benefit": total_incremental_benefit,
        "total_net_benefit": total_net_benefit,
        "roi_percent": roi_percent,
    }

    return result, summary
