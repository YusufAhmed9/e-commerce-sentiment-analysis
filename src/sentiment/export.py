import logging
from abc import ABC, abstractmethod
from enum import Enum

import joblib
import numpy as np
from sklearn.base import BaseEstimator

from sentiment.logging_conf import init_logger

logger = init_logger(name=__name__, level=logging.INFO)

class ExportType(str, Enum):
    EXPORT_ONNX = "onnx"
    EXPORT_PICKLE = "pkl"


class ExportStatus(Enum):
    INVALID_EXPORT_TYPE = "invalid_export_type_passed"

class BaseExporter(ABC):

    @abstractmethod
    def export(self, model: BaseEstimator, save_path: str) -> None:...

class PickleExporter(BaseExporter):

    def export(self, model: BaseEstimator, save_path: str) -> None:
        logger.info("exporting model using pickle", extra={"model_class":model.__class__.__name__, "path":save_path})

        joblib.dump(model, f"{save_path}.pkl")


class OnnxExporter(BaseExporter):

    def __init__(self, sample):
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        

        self._convert = convert_sklearn
        self._tensor_type = FloatTensorType
        self.n_features = np.asarray(sample).shape[1]

    def export(self, model: BaseEstimator, save_path: str) -> None:
        logger.info("exporting model using onnx", extra={"model_class":model.__class__.__name__, "path":save_path})

        onnx_model = self._convert(
            model,
            initial_types=[("input", self._tensor_type([None, self.n_features]))],
        )
        with open(f"{save_path}.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())