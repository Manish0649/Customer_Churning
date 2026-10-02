"""Load one trusted local artifact and serve stable model/retention results."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from retention_strategy.retention_rules import assign_churn_risk
from retention_strategy.reward_recommendations import recommend_retention_actions
from retention_strategy.roi_analysis import estimate_retention_roi
from supervised.model_evaluation import churn_probabilities
from .schemas import Customer, ModelInfo

ROI_NOTE = (
    "Illustrative scenario, not measured causal uplift: 10% assumed incremental retention, "
    "3 additional months, 40% contribution margin; expected reward cost is weighted by churn "
    "probability. Customers with no recommended reward are excluded from the campaign ROI."
)


class ModelService:
    def __init__(self, artifact_path: Path):
        # Only load locally generated/trusted files: joblib is a pickle format.
        bundle = joblib.load(artifact_path)
        if bundle.get("sklearn_version") != sklearn.__version__:
            raise ValueError("Artifact scikit-learn version differs; re-export in this environment")
        self.info = ModelInfo.model_validate(bundle["metadata"])
        if self.info.artifact_version != 1:
            raise ValueError("Unsupported artifact version")
        self.pipeline = bundle["pipeline"]
        self.segmentation = bundle["segmentation"]
        self.numeric_features = bundle["numeric_features"]
        self.clustering_features = bundle["clustering_features"]
        if set(self.pipeline.classes_) != {0, 1}:
            raise ValueError("Classifier must have binary classes 0 and 1")
        if list(self.pipeline.feature_names_in_) != self.info.input_features:
            raise ValueError("Artifact feature schema does not match pipeline")

    def frame(self, customers: list[Customer]):
        data = pd.DataFrame([customer.model_dump() for customer in customers])
        # Same year/month features as training, without pandas nanosecond date limits.
        data["signup_year"] = [c.signup_date.year if c.signup_date else np.nan for c in customers]
        data["signup_month"] = [c.signup_date.month if c.signup_date else np.nan for c in customers]
        features = data.reindex(columns=self.info.input_features)
        for column in features:
            if column in self.numeric_features:
                features[column] = pd.to_numeric(features[column]).astype(float)
            else:
                features[column] = features[column].astype(object).where(features[column].notna(), np.nan)
        return features

    def predict(self, customers: list[Customer]):
        features = self.frame(customers)
        probabilities = churn_probabilities(self.pipeline, features)
        # Preserve the classifier's decision rule (notably SVC probability calibration).
        predictions = self.pipeline.predict(features)
        risks = assign_churn_risk(pd.DataFrame({"churn_probability": probabilities}))["risk_category"]
        return [{"customer_id": c.customer_id, "churn_probability": float(p),
                 "predicted_churn": int(predicted), "risk_level": str(risk),
                 "model_name": self.info.model_name}
                for c, p, predicted, risk in zip(customers, probabilities, predictions, risks)]

    def segment(self, customers: list[Customer]):
        features = self.frame(customers)
        labels = self.segmentation.predict(features[self.clustering_features])
        return [{"customer_id": customer.customer_id, "segment_id": int(label),
                 "customer_segment": f"Segment {int(label) + 1}",
                 "segment_profile": self.info.segment_profiles[str(int(label))]}
                for customer, label in zip(customers, labels)]

    def retain(self, customers: list[Customer], probabilities: list[float]):
        data = pd.DataFrame({"churn_probability": probabilities})
        data = assign_churn_risk(data)
        low, high = self.info.value_thresholds["low"], self.info.value_thresholds["high"]
        data["customer_value"] = ["Low" if c.totalcharges <= low else
                                  "Medium" if c.totalcharges <= high else "High" for c in customers]
        data = recommend_retention_actions(data)
        results = []
        for customer, row in zip(customers, data.to_dict("records")):
            results.append({
                "customer_id": customer.customer_id,
                "churn_probability": row["churn_probability"], "risk_level": row["risk_category"],
                "customer_value": row["customer_value"],
                "recommended_action": row["retention_action"],
                "recommended_reward": row["recommended_reward"], "reward_cost": float(row["reward_cost"]),
                "reason": [f"{row['risk_category']} predicted churn risk",
                           f"{row['customer_value']} historical spending relative to the training population",
                           "Reward selected by the configured risk/value rules; not a causal prediction"],
            })
        return results

    def strategy(self, customers: list[Customer]):
        predictions = self.predict(customers)
        segments = self.segment(customers)
        rewards = self.retain(customers, [row["churn_probability"] for row in predictions])
        return [{**prediction, **segment, **reward}
                for prediction, segment, reward in zip(predictions, segments, rewards)]

    def batch(self, customers: list[Customer]):
        results = self.strategy(customers)
        df = pd.DataFrame(results)
        df["monthlycharges"] = [c.monthlycharges for c in customers]
        campaign = df.loc[df.reward_cost > 0].copy()
        _, roi = estimate_retention_roi(campaign)
        return {
            "results": results,
            "risk_distribution": df.risk_level.value_counts().to_dict(),
            "segment_distribution": df.customer_segment.value_counts().to_dict(),
            "high_value_high_risk_count": int(((df.risk_level == "High") & (df.customer_value == "High")).sum()),
            "roi": roi, "roi_note": ROI_NOTE,
        }
