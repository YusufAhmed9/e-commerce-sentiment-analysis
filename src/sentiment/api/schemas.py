from enum import Enum

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):

    text: str = Field(min_length=2)

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    model_version: str

class ResponseEnum(Enum):
    MODEL_LOAD_FAIL = "model_failed_to_load"
    MODEL_LOAD_SUCCESS = "model_loaded_successfully"