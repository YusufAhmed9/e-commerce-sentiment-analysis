from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from sentiment.export import ExportType, PickleExporter
from sentiment.predict import LogisticPredictor

X, y = make_classification(
    n_samples=60, n_features=20, n_informative=6,
    n_classes=3, random_state=42,
)

model = LogisticRegression(max_iter=200).fit(X, y)

def test_pickle(artifacts):
    pkl_exporter = PickleExporter()
    pkl_exporter.export(model, artifacts / "pickled_lr.pkl")

    pkl_predictor = LogisticPredictor(ExportType.EXPORT_PICKLE)
    pkl_predictor.load(artifacts / "pickled_lr.pkl")
    assert pkl_predictor._session != None
