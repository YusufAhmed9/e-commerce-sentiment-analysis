import logging
from abc import ABC, abstractmethod
from enum import Enum

import joblib
import numpy as np
import onnxruntime as rt

from .export import ExportStatus, ExportType
from .logging_conf import init_logger

logger = init_logger(name=__name__, level=logging.INFO)

class PredictionStatus(Enum):
    MODEL_NOT_LOADED = "model_is_not_loaded"

class BasePredictor(ABC):

    @abstractmethod
    def load(self, model_path: str):
        pass

    @abstractmethod
    def predict_one(self, X):
        pass

    @abstractmethod
    def predict_batch(self, X):
        pass

class LogisticPredictor(BasePredictor):

    def __init__(self, export_type: ExportType):
        self._export_type = export_type
        self._session = None
        self._input_name = None

    def load(self, model_save_path: str):
        logger.info("loading a model", extra={"export_type": self._export_type.value, "save_path": model_save_path})
        if self._export_type == ExportType.EXPORT_PICKLE:
            self._session = joblib.load(model_save_path)

        elif self._export_type == ExportType.EXPORT_ONNX:
            self._session = rt.InferenceSession(
                model_save_path, 
                providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name

        else:
            raise ValueError(ExportStatus.INVALID_EXPORT_TYPE.value)

    def predict_batch(self, X):
        logger.info("predicting a batch", extra={"batch_size": X.shape[0]})
        if self._session is None:
            raise ValueError(PredictionStatus.MODEL_NOT_LOADED.value)

        if self._export_type == ExportType.EXPORT_PICKLE:
            return self._session.predict(X)

        X = np.asarray(X, dtype=np.float32)        
        return self._session.run(None, {self._input_name: X})[0]

    def predict_one(self, X):
        X = np.asarray(X).reshape(1, -1)             
        return self.predict_batch(X)[0]