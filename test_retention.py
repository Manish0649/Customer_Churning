
import pandas as pd

from retention_strategy.retention_analysis import (
    run_retention_analysis
)

# Step 1: Create sample customer data
customers = pd.DataFrame({
    "customer_id": [101, 102, 103],
    "churn_probability": [0.85, 0.55, 0.20],
    "totalcharges": [5000, 2500, 800],
    "monthlycharges": [500, 300, 100]
})

# Step 2: Run the complete retention analysis
result, roi_summary = run_retention_analysis(customers)

# Step 3: Display customer recommendations
print("\n--- CUSTOMER RETENTION RECOMMENDATIONS ---")
print(result.to_string(index=False))

# Step 4: Display ROI summary
print("\n--- ROI SUMMARY ---")
print(roi_summary)

# Step 5: Validate the output
assert len(result) == 3

required_columns = [
    "risk_category",
    "customer_value",
    "retention_action",
    "recommended_reward",
    "reward_cost"
]

for column in required_columns:
    assert column in result.columns, f"Missing column: {column}"

assert result["reward_cost"].notna().all()

print("\nEnd-to-end retention test passed!")
