import logging
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from sentiment.config import MODEL_DIR
from sentiment.logging_conf import init_logger
from sentiment.predict import LogisticPredictor
from sentiment.service import PredictionService
from sentiment.settings import settings

from .schemas import PredictionRequest, PredictionResponse, ResponseEnum

logger = init_logger(name=__name__, level = logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_save_path = MODEL_DIR / settings.MODEL_NAME
    vectorizer_save_path = MODEL_DIR / settings.VECTORIZER_PATH

    logger.debug("loading model in app", extra={"model_save_path": model_save_path, "export_type": settings.EXPORT_TYPE.value})

    predictor = LogisticPredictor(export_type=settings.EXPORT_TYPE)
    predictor.load(model_save_path=model_save_path)
    vectorizer = joblib.load(vectorizer_save_path)
    app.state.prediction_service = PredictionService(predictor=predictor, vectorizer=vectorizer)
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    if app.state.prediction_service.is_loaded:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": ResponseEnum.MODEL_LOAD_SUCCESS.value
            }
        )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": ResponseEnum.MODEL_LOAD_FAIL.value
        }
    )

@app.get("/metadata")
def metadata():
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "method_not_implemented_yet"
        }
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(prediction_request: PredictionRequest):
    label = app.state.prediction_service.predict(prediction_request.text)

    return {
        "label": label,
        "confidence": 1,
        "model_version": "0.1"
    }

@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(prediction_requests: list[PredictionRequest]):
    texts = [request.text for request in prediction_requests]
    labels = app.state.prediction_service.predict(texts)
    
    response = [
        {
        "label": label,
        "confidence": 1,
        "model_version": "0.1"
        }
        for label in labels
    ]
    return response
