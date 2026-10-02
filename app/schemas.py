"""Pydantic models are the source of truth for the OpenAPI contract."""
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Amount = Annotated[float, Field(ge=0, le=1e12)]
Count = Annotated[int, Field(ge=0, le=1_000_000)]
Probability = Annotated[float, Field(ge=0, le=1)]
Risk = Literal["Low", "Medium", "High"]
ALIASES = {
    "monthly_charges": "monthlycharges",
    "total_charges": "totalcharges",
    "complaints": "num_complaints",
    "service_calls": "num_service_calls",
}
EXAMPLE = {
    "customer_id": "C10291", "tenure": 14,
    "monthlycharges": 89.5, "totalcharges": 1253.0,
    "num_complaints": 3, "num_service_calls": 5,
    "customer_satisfaction": 4.0,
}


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Customer(Contract):
    """Core fields are required. Omitted optional features use training imputations.

    Accepted aliases: monthly_charges, total_charges, complaints, service_calls.
    Supply only one spelling of each field. The training target is not an input.
    """
    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE]})
    customer_id: str = Field(min_length=1, max_length=100, pattern=r"\S")
    tenure: Count
    monthlycharges: Amount
    totalcharges: Amount
    signup_date: datetime | None = None
    age: Annotated[int, Field(ge=0, le=120)] | None = None
    gender: str | None = Field(default=None, max_length=100)
    annual_income: Amount | None = None
    education: str | None = Field(default=None, max_length=100)
    marital_status: str | None = Field(default=None, max_length=100)
    dependents: Count | None = None
    contract: str | None = Field(default=None, max_length=100)
    payment_method: str | None = Field(default=None, max_length=100)
    paperless_billing: Literal["Yes", "No"] | None = None
    senior_citizen: Literal[0, 1] | None = None
    num_services: Count | None = None
    has_phone_service: Literal[0, 1] | None = None
    has_internet_service: Literal[0, 1] | None = None
    has_online_security: Literal[0, 1] | None = None
    has_online_backup: Literal[0, 1] | None = None
    has_device_protection: Literal[0, 1] | None = None
    has_tech_support: Literal[0, 1] | None = None
    has_streaming_tv: Literal[0, 1] | None = None
    has_streaming_movies: Literal[0, 1] | None = None
    customer_satisfaction: Annotated[float, Field(ge=0, le=10)] | None = None
    num_complaints: Count | None = None
    num_service_calls: Count | None = None
    late_payments: Count | None = None
    avg_monthly_gb: Amount | None = None
    days_since_last_interaction: Count | None = None
    credit_score: Annotated[float, Field(ge=0, le=1000)] | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for alias, canonical in ALIASES.items():
            if alias in data:
                if canonical in data:
                    raise ValueError(f"Supply only {canonical} or {alias}, not both")
                data[canonical] = data.pop(alias)
        return data


class Prediction(Contract):
    customer_id: str
    churn_probability: Probability
    predicted_churn: Literal[0, 1]
    risk_level: Risk
    model_name: str


class Segment(Contract):
    customer_id: str
    segment_id: int
    customer_segment: str
    segment_profile: dict[str, float]


class RetentionRequest(Contract):
    customer: Customer
    churn_probability: Probability


class Retention(Contract):
    customer_id: str
    churn_probability: Probability
    risk_level: Risk
    customer_value: Risk
    recommended_action: str
    recommended_reward: str
    reward_cost: float
    reason: list[str]


class Strategy(Prediction, Segment, Retention):
    pass


class BatchRequest(Contract):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class RoiSummary(Contract):
    total_customers: int
    expected_incremental_saves: float
    total_reward_cost: float
    total_incremental_benefit: float
    total_net_benefit: float
    roi_percent: float | None


class BatchResponse(Contract):
    results: list[Strategy]
    risk_distribution: dict[str, int]
    segment_distribution: dict[str, int]
    high_value_high_risk_count: int
    roi: RoiSummary
    roi_note: str


class Health(Contract):
    status: Literal["ready", "unavailable"]
    model_loaded: bool
    detail: str


class ModelInfo(Contract):
    artifact_version: int
    model_name: str
    trained_at: str
    dataset_rows: int
    sample_rows: int
    training_rows: int
    holdout_rows: int
    metrics: dict[str, float]
    feature_importance: dict[str, float]
    input_features: list[str]
    value_thresholds: dict[str, float]
    segment_profiles: dict[str, dict[str, float]]
    note: str


class ErrorResponse(Contract):
    detail: str


class ValidationIssue(Contract):
    loc: list[str | int]
    msg: str
    type: str


class ValidationErrorResponse(Contract):
    detail: list[ValidationIssue]
