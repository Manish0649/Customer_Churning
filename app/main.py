"""FastAPI backend. Start with: uvicorn app.main:app --host 127.0.0.1 --port 8000"""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from .schemas import (BatchRequest, BatchResponse, Customer, ErrorResponse, Health,
                      ModelInfo, Prediction, Retention, RetentionRequest, Segment,
                      Strategy, ValidationErrorResponse)
from .service import ModelService

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)
ERRORS = {
    422: {"model": ValidationErrorResponse, "description": "Invalid customer data or batch size"},
    503: {"model": ErrorResponse, "description": "Model artifact unavailable; export it and restart the server"},
}


def create_app(include_ui=True):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = None
        artifact = Path(os.getenv("CHURN_MODEL_PATH", str(ROOT / "artifacts/churn_bundle.joblib")))
        try:
            app.state.service = ModelService(artifact)
        except Exception:
            logger.exception("Unable to load the model artifact; export a compatible artifact and restart")
        yield
        app.state.service = None

    app = FastAPI(
        title="Customer Churn and Retention API", version="1.0.0", lifespan=lifespan,
        description=("Predict churn, retrieve K-Means segments and generate rule-based retention "
                     "recommendations. Optional customer attributes use saved training imputations. "
                     "Training is offline. Monetary values use the source dataset's currency. "
                     "Gradio is available at /ui. This local demo has no authentication."),
        openapi_tags=[{"name": "Operations"}, {"name": "Model results"}, {"name": "Retention"}],
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Consistent documented shape, excluding raw input and non-JSON exception contexts.
        issues = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": issues})

    def get_service(request: Request):
        service = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(503, "Model unavailable. Run python -m scripts.export_model and restart the API.")
        return service

    Service = Annotated[ModelService, Depends(get_service)]

    @app.get("/", include_in_schema=False)
    def home():
        return RedirectResponse("/docs")

    @app.get("/health", response_model=Health, tags=["Operations"], operation_id="getHealth",
             responses={503: {"model": Health, "description": "Application running, model not ready"}})
    def health(request: Request):
        loaded = getattr(request.app.state, "service", None) is not None
        result = Health(status="ready" if loaded else "unavailable", model_loaded=loaded,
                        detail="Model loaded" if loaded else "Export a compatible model artifact and restart")
        return JSONResponse(status_code=200 if loaded else 503, content=result.model_dump())

    @app.get("/model-info", response_model=ModelInfo, tags=["Model results"],
             operation_id="getModelInfo", responses={503: ERRORS[503]})
    def model_info(service: Service):
        return service.info

    @app.post("/predict-churn", response_model=Prediction, tags=["Model results"],
              operation_id="predictChurn", responses=ERRORS)
    def predict_churn(customer: Customer, service: Service):
        return service.predict([customer])[0]

    @app.post("/segment-customer", response_model=Segment, tags=["Model results"],
              operation_id="segmentCustomer", responses=ERRORS)
    def segment_customer(customer: Customer, service: Service):
        return service.segment([customer])[0]

    @app.post("/recommend-retention", response_model=Retention, tags=["Retention"],
              operation_id="recommendRetention", responses=ERRORS,
              description="Apply retention rules to a supplied churn probability and customer. No classifier call.")
    def recommend_retention(payload: RetentionRequest, service: Service):
        return service.retain([payload.customer], [payload.churn_probability])[0]

    @app.post("/customer-strategy", response_model=Strategy, tags=["Retention"],
              operation_id="getCustomerStrategy", responses=ERRORS)
    def customer_strategy(customer: Customer, service: Service):
        return service.strategy([customer])[0]

    @app.post("/customer-strategy/batch", response_model=BatchResponse, tags=["Retention"],
              operation_id="getBatchStrategies", responses=ERRORS,
              description="Score 1–1000 customers in input order; include campaign scenario ROI.")
    def batch_strategy(payload: BatchRequest, service: Service):
        return service.batch(payload.customers)

    if include_ui:
        import gradio as gr
        from .ui import build_ui
        app = gr.mount_gradio_app(app, build_ui(), path="/ui", max_file_size="2mb",
                                 show_error=False)
    return app


app = create_app(include_ui=os.getenv("CHURN_ENABLE_UI", "1") != "0")
