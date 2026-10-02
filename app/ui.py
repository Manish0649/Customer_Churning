"""Gradio client: every model result is obtained from the FastAPI HTTP API."""
import json
import os
from pathlib import Path
import tempfile

import gradio as gr
import httpx
import pandas as pd

from .schemas import EXAMPLE


def api_request(path, payload=None):
    base = os.getenv("CHURN_API_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        with httpx.Client(timeout=60, trust_env=False) as client:
            response = client.get(base + path) if payload is None else client.post(base + path, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", "API request failed")
        raise gr.Error(json.dumps(detail, ensure_ascii=False)) from exc
    except httpx.RequestError as exc:
        raise gr.Error("Cannot reach the backend. Start FastAPI and check CHURN_API_URL.") from exc


def parse_json(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError) as exc:
        raise gr.Error("Enter valid JSON.") from exc


def predict_customer(customer_id, tenure, monthlycharges, totalcharges, optional_json):
    optional = parse_json(optional_json or "{}")
    if not isinstance(optional, dict):
        raise gr.Error("Additional attributes must be a JSON object.")
    core = dict(customer_id=customer_id, tenure=tenure,
                monthlycharges=monthlycharges, totalcharges=totalcharges)
    if set(core).intersection(optional):
        raise gr.Error("Set the four core attributes in the form, not the additional JSON.")
    result = api_request("/customer-strategy", {**core, **optional})
    summary = (
        f"### Churn probability: {result['churn_probability']:.1%}\n"
        f"**Risk:** {result['risk_level']} · **Segment:** {result['customer_segment']} · "
        f"**Customer value:** {result['customer_value']}\n\n"
        f"**Action:** {result['recommended_action']}\n\n"
        f"**Reward:** {result['recommended_reward']} · **Estimated cost:** {result['reward_cost']:.2f}"
    )
    return summary, result


def batch_customers(file_path, records_json):
    if file_path:
        if Path(file_path).stat().st_size > 2 * 1024 * 1024:
            raise gr.Error("CSV must be at most 2 MB; submit at most 1,000 customers per batch.")
        try:
            frame = pd.read_csv(file_path, nrows=1001, dtype={"customer_id": str})
        except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
            raise gr.Error("Unable to read the CSV; use a UTF-8 file with a header row.") from exc
        # A scored source dataset may contain churn. Explicitly exclude that label from inference.
        frame = frame.drop(columns=["churn"], errors="ignore")
        records = json.loads(frame.to_json(orient="records"))
    else:
        records = parse_json(records_json)
    if not isinstance(records, list) or not 1 <= len(records) <= 1000:
        raise gr.Error("Provide an array or CSV containing 1–1,000 customers.")
    result = api_request("/customer-strategy/batch", {"customers": records})
    rows = pd.DataFrame(result["results"])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", prefix="churn-results-", delete=False) as output:
        rows.to_csv(output, index=False)
        download_path = output.name
    summary = {key: value for key, value in result.items() if key != "results"}
    risk = pd.DataFrame(list(result["risk_distribution"].items()), columns=["Risk", "Customers"])
    segments = pd.DataFrame(list(result["segment_distribution"].items()), columns=["Segment", "Customers"])
    return rows, summary, risk, segments, download_path


def model_info():
    result = api_request("/model-info")
    metrics = pd.DataFrame(list(result["metrics"].items()), columns=["Metric", "Value"])
    importance = pd.DataFrame(list(result["feature_importance"].items()), columns=["Feature", "Importance"])
    return result, metrics, importance


def build_ui():
    with gr.Blocks(title="Customer Churn & Retention", delete_cache=(3600, 3600), analytics_enabled=False) as demo:
        gr.Markdown("# Customer Churn & Retention\nPredict customer risk and review retention suggestions.")
        gr.Markdown("[Open API documentation](/docs) · [OpenAPI contract](/openapi.json)")
        with gr.Tab("Customer strategy"):
            with gr.Row():
                customer_id = gr.Textbox(label="Customer ID", value="C10291")
                tenure = gr.Number(label="Tenure (months)", value=14, precision=0, minimum=0)
            with gr.Row():
                monthly = gr.Number(label="Monthly charges", value=89.5, minimum=0)
                total = gr.Number(label="Total charges", value=1253, minimum=0)
            optional = gr.Textbox(label="Additional customer attributes (JSON)", lines=4,
                value='{"num_complaints": 3, "num_service_calls": 5, "customer_satisfaction": 4}',
                info="Supply known attributes for better-informed predictions. Missing attributes use training imputations.")
            predict = gr.Button("Get customer strategy", variant="primary")
            summary = gr.Markdown()
            details = gr.JSON(label="Full API result")
            predict.click(predict_customer, [customer_id, tenure, monthly, total, optional], [summary, details])
        with gr.Tab("Batch & retention dashboard"):
            gr.Markdown("Score up to 1,000 customers. A CSV takes priority over JSON. "
                        "If a CSV contains the historical churn label, that column is excluded from prediction.")
            upload = gr.File(label="Customer CSV (up to 2 MB)", file_types=[".csv"], type="filepath")
            records = gr.Textbox(label="Customer records (JSON array)", lines=6,
                                 value=json.dumps([EXAMPLE], indent=2))
            score = gr.Button("Score customers", variant="primary")
            table = gr.Dataframe(label="Customer results", interactive=False)
            stats = gr.JSON(label="Retention summary and illustrative ROI")
            with gr.Row():
                risk_chart = gr.BarPlot(x="Risk", y="Customers", title="Churn risk distribution")
                segment_chart = gr.BarPlot(x="Segment", y="Customers", title="Customer segments")
            download = gr.File(label="Download scored customers")
            score.click(batch_customers, [upload, records], [table, stats, risk_chart, segment_chart, download])
        with gr.Tab("Model results"):
            refresh = gr.Button("Load model information")
            metrics = gr.Dataframe(label="Holdout metrics", interactive=False)
            importance = gr.Dataframe(label="Global feature importance (if supported)", interactive=False)
            metadata = gr.JSON(label="Model metadata and segment profiles")
            refresh.click(model_info, outputs=[metadata, metrics, importance])
        gr.Markdown("Rewards use configurable rules. ROI is an illustrative scenario, not measured "
                    "retention uplift. Segment profiles and feature importance do not establish causation.")
    return demo
