import logging

from sentiment.features import TextProcessingPipeline
from sentiment.logging_conf import init_logger
from sentiment.predict import BasePredictor

logger = init_logger(name=__name__, level = logging.DEBUG)

sentiment_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

class PredictionService:

    def __init__(self, predictor: BasePredictor, vectorizer):
        self._predictor = predictor
        self._preprocessor = TextProcessingPipeline()
        self._vectorizer = vectorizer

    @property
    def is_loaded(self):
        return self._predictor.is_loaded and self._vectorizer is not None
    
    def predict(self, texts: str | list[str]):
        if isinstance(texts, str):
            texts = [texts]
        X = self._vectorizer.transform([self._preprocessor(t) for t in texts])
        X = X.toarray()        

        sentiment = self._predictor.predict_batch(X)
        if len(texts) == 1:
            return sentiment_map.get(sentiment[0])
        
        return [sentiment_map[label] for label in sentiment]