import pandas as pd

from .retention_rules import assign_churn_risk
from .customer_value import assign_customer_value
from .reward_recommendations import (
    recommend_retention_actions
)
from .roi_analysis import estimate_retention_roi


def run_retention_analysis(customers):

    # Step 1: Assign churn-risk categories
    customers = assign_churn_risk(
        customers,
        probability_col="churn_probability",
        low_threshold=0.40,
        high_threshold=0.70,
    )

    # Step 2: Assign customer-value categories
    customers, value_thresholds = assign_customer_value(
        customers,
        value_col="totalcharges",
    )

    # Step 3: Recommend retention rewards
    customers = recommend_retention_actions(customers)

    # Step 4: Estimate ROI
    customers, roi_summary = estimate_retention_roi(
        customers
    )

    return customers, roi_summary