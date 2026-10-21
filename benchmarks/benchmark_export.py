import json
import time
from sentiment.export import ExportType
import numpy as np
from sentiment.data import load_splits
from sentiment.config import data_configs, TFIDF_01_PATH, LOGISTIC_01_SAVE_PATH
from sentiment.features import FeatureBuilder, TextProcessingPipeline
from sentiment.predict import LogisticPredictor
import joblib

def benchmark(predict_fn, inputs, warmup=20):
    for x in inputs[:warmup]:                 # warm-up: first calls are slower
        predict_fn(x)

    times_ms = []
    for x in inputs:
        t0 = time.perf_counter()
        predict_fn(x)
        times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms = np.array(times_ms)
    return {
        "mean_ms": round(float(times_ms.mean()), 4),
        "p95_ms": round(float(np.percentile(times_ms, 95)), 4),
    }

train, val, test = load_splits(configs=data_configs)
train = train.iloc[:500]

vectorizer = joblib.load(f"{TFIDF_01_PATH}.pkl")
feature_builder = FeatureBuilder(TextProcessingPipeline(), vectorizer)
X, _ = feature_builder.transform(df=train)
X = [X[i].toarray() for i in range(X.shape[0])]
print(X[0])
onnx_predictor = LogisticPredictor(ExportType.EXPORT_ONNX)
pkl_predictor = LogisticPredictor(ExportType.EXPORT_PICKLE)
onnx_predictor.load(model_save_path=LOGISTIC_01_SAVE_PATH)
pkl_predictor.load(model_save_path=LOGISTIC_01_SAVE_PATH)

onnx_results = benchmark(onnx_predictor.predict_one, X)
pkl_results = benchmark(pkl_predictor.predict_one, X)
report = {"n_rows": 500, "pickle": pkl_results, "onnx": onnx_results}

print(json.dumps(report, indent=2))
with open("artifacts/benchmark.json", "w") as f:
    json.dump(report, f, indent=2)