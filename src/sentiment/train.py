import logging
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from .config import (
    LOGISTIC_01_CONFIGS,
    LOGISTIC_01_SAVE_PATH,
    test_configs,
    train_configs,
    val_configs,
)
from .data import load_all
from .export import ExportType, OnnxExporter, PickleExporter
from .features import preprocessor, vectorizer
from .logging_conf import init_logger
from .predict import LogisticPredictor

logger = init_logger(name=__name__, level = logging.INFO)

class BaseTrainer(ABC):

    def __init__(self, model: BaseEstimator, model_configs: dict, metadata: dict):
        self.model = model
        self.model_configs = model_configs
        self.metadata = metadata

    @abstractmethod
    def fit(self, X, y):...


class LogisticTrainer(BaseTrainer):

    def __init__(self, model_configs: dict, metadata: dict):
        super().__init__(LogisticRegression(**model_configs), model_configs, metadata)

    def fit(self, X, y):
        logger.info("fitting model", extra={"model_class": self.model.__class__.__name__, "data_size": X.shape[0]})

        self.model.fit(X, y)

if __name__ == "__main__":
#------------------------------ Loading Data ------------------------------
    train_df = load_all(configs=train_configs).drop_duplicates().dropna()
    val_df = load_all(configs=val_configs).drop_duplicates().dropna()
    test_df = load_all(configs=test_configs).drop_duplicates().dropna()

#------------------------------ Preprocessing Data ------------------------------
    X_vec_list = []
    y_values = []
    first_time = True

    for data in [train_df, val_df, test_df]:
        data['cleaned_text'] = data['text'].apply(preprocessor)
        if first_time:
            X_vec = vectorizer.fit_transform(data['cleaned_text'])
            first_time = False

        else:
            X_vec = vectorizer.transform(data['cleaned_text'])

        X_vec_list.append(X_vec)
        y_values.append(data['sentiment'].values)
        
    X_train, X_val, X_test = X_vec_list
    y_train, y_val, y_test = y_values

#------------------------------ Model ------------------------------
    trainer = LogisticTrainer(
        model_configs=LOGISTIC_01_CONFIGS,
        metadata={}
    )
    trainer.fit(X=X_train, y=y_train)
    OnnxExporter(sample=X_train[0].toarray()).export(model=trainer.model, save_path=f"{LOGISTIC_01_SAVE_PATH}.onnx")
    PickleExporter().export(model=trainer.model, save_path=f"{LOGISTIC_01_SAVE_PATH}.pkl")

    predictor = LogisticPredictor(ExportType.EXPORT_PICKLE)
    predictor.load(model_save_path=f"{LOGISTIC_01_SAVE_PATH}.pkl")
    y_pred = predictor.predict_batch(X=X_val)

    report = classification_report(y_val, y_pred)
    logger.info(report)

    predictor = LogisticPredictor(ExportType.EXPORT_ONNX)
    predictor.load(model_save_path=f"{LOGISTIC_01_SAVE_PATH}.onnx")
    y_pred = predictor.predict_batch(X=X_val[:10].toarray())

    report = classification_report(y_val[:10], y_pred)
    logger.info(report)