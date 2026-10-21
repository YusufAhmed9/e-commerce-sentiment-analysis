import logging
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sentiment.config import (
    LOGISTIC_01_CONFIGS,
    LOGISTIC_01_SAVE_PATH,
    TFIDF_01_PATH,
    data_configs,
)
from sentiment.data import load_splits
from sentiment.export import OnnxExporter, PickleExporter
from sentiment.features import build_features
from sentiment.logging_conf import init_logger

logger = init_logger(name=__name__, level = logging.INFO)

class BaseTrainer(ABC):

    def __init__(self, model: BaseEstimator, model_configs: dict, metadata: dict):
        self.model = model
        self.model_configs = model_configs
        self.metadata = metadata

    @abstractmethod
    def fit(self, X, y):...
    
    def get_classification_report(self, X, y):
        y_pred = self.model.predict(X)
        return classification_report(y, y_pred, output_dict=True, zero_division=0)


class LogisticTrainer(BaseTrainer):

    def __init__(self, model_configs: dict, metadata: dict):
        super().__init__(LogisticRegression(**model_configs), model_configs, metadata)

    def fit(self, X, y):
        logger.info("fitting model", extra={"model_class": self.model.__class__.__name__, "data_size": X.shape[0]})

        self.model.fit(X, y)



def run_train_pipeline(configs: dict):
    train, val, test = load_splits(configs=configs)

    feature_builder = build_features()
    X_train, y_train = feature_builder.fit_transform(df=train)
    X_val, y_val = feature_builder.transform(df=val)
    X_test, y_test = feature_builder.transform(df=test)

    trainer = LogisticTrainer(
        model_configs=LOGISTIC_01_CONFIGS,
        metadata={}
    )

    trainer.fit(X_train, y_train)
    val_report = trainer.get_classification_report(X_val, y_val)
    test_report = trainer.get_classification_report(X_test, y_test)
    logger.info("validation metrics", extra={"split": "val", "macro_f1": val_report["macro avg"]["f1-score"], "accuracy": val_report["accuracy"]})
    logger.info("test metrics", extra={"split": "test", "macro_f1": test_report["macro avg"]["f1-score"], "accuracy": test_report["accuracy"]})

    OnnxExporter(sample=X_train[0].toarray()).export(model=trainer.model, save_path=LOGISTIC_01_SAVE_PATH)
    PickleExporter().export(model=trainer.model, save_path=LOGISTIC_01_SAVE_PATH)
    feature_builder.save(TFIDF_01_PATH)
if __name__ == "__main__":
    
    run_train_pipeline(configs=data_configs)
