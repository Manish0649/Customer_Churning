import pandas as pd


# Retention rules based on churn risk and customer value
DEFAULT_RETENTION_RULES = {
    ("Low", "Low"): {
        "action": "Monitor customer",
        "reward": "No reward",
        "reward_cost": 0
    },
    ("Low", "Medium"): {
        "action": "Engagement campaign",
        "reward": "Personalized offer",
        "reward_cost": 50
    },
    ("Low", "High"): {
        "action": "Loyalty retention",
        "reward": "Loyalty points",
        "reward_cost": 100
    },

    ("Medium", "Low"): {
        "action": "Low-cost re-engagement",
        "reward": "Small discount",
        "reward_cost": 50
    },
    ("Medium", "Medium"): {
        "action": "Targeted retention campaign",
        "reward": "10% discount",
        "reward_cost": 100
    },
    ("Medium", "High"): {
        "action": "Priority retention",
        "reward": "Exclusive loyalty offer",
        "reward_cost": 200
    },

    ("High", "Low"): {
        "action": "Cost-effective win-back",
        "reward": "Limited-time coupon",
        "reward_cost": 50
    },
    ("High", "Medium"): {
        "action": "Urgent retention campaign",
        "reward": "Personalized discount",
        "reward_cost": 200
    },
    ("High", "High"): {
        "action": "High-priority retention",
        "reward": "Premium reward and personal outreach",
        "reward_cost": 300
    }
}


def recommend_retention_actions(
    df: pd.DataFrame,
    rules: dict = None
) -> pd.DataFrame:
    """
    Recommend retention actions and rewards based on
    customer churn risk and customer value.

    Required columns:
        - risk_category
        - customer_value

    Returns:
        DataFrame with action, reward, and reward_cost columns.
    """

    required_columns = {"risk_category", "customer_value"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if rules is None:
        rules = DEFAULT_RETENTION_RULES

    result = df.copy()

    def get_recommendation(row):
        key = (
            row["risk_category"],
            row["customer_value"]
        )

        recommendation = rules.get(key)

        if recommendation is None:
            return {
                "retention_action": "Review customer",
                "recommended_reward": "No reward",
                "reward_cost": 0
            }

        return {
            "retention_action": recommendation["action"],
            "recommended_reward": recommendation["reward"],
            "reward_cost": recommendation["reward_cost"]
        }

    recommendations = result.apply(
        get_recommendation,
        axis=1,
        result_type="expand"
    )

    result[
        [
            "retention_action",
            "recommended_reward",
            "reward_cost"
        ]
    ] = recommendations

    return result