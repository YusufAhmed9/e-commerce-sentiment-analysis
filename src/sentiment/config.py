from pathlib import Path

import pandas as pd

ROOT_DIR = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
DATA_DIR =  ROOT_DIR / "data"  
MODEL_DIR = ROOT_DIR / "models"

LOGISTIC_01_CONFIGS = {
    "max_iter" : 1000
}
LOGISTIC_01_SAVE_PATH = MODEL_DIR / "logistic_01"

DATASETS = {
    "100k": {
        "dir": "100k_arabic_reviews",
        "columns_map": {"text": "text", "sentiment": "sentiment"},
        "label_map": {"negative": 0, "mixed": 1, "positive": 2},
    },
    "twitter": {
        "dir": "arabic_twitter",
        "columns_map": {"text": "text", "sentiment": "sentiment"},
        "label_map": {"neg": 0, "pos": 2},
    },
    "ss30": {
        "dir": "ss2030",
        "columns_map": {"text": "text", "sentiment": "sentiment"},
        "label_map": {0: 0, 1: 2},
    },
}


def map_label(label_map):
    def mapper(col):
        if pd.api.types.is_string_dtype(col):
            col = col.str.lower()
        mapped = col.map(label_map)
        if mapped.isna().any():
            bad = col[mapped.isna()].unique()[:5]
            raise ValueError(f"Unmapped labels: {list(bad)}")
        return mapped
    return mapper


def build_configs(split):
    return {
        name: {
            "path": DATA_DIR / ds["dir"] / f"{split}.csv",
            "columns_map": ds["columns_map"],
            "label_transform": map_label(ds["label_map"]),
        }
        for name, ds in DATASETS.items()
    }


train_configs = build_configs("train")
val_configs = build_configs("val")
test_configs = build_configs("test")