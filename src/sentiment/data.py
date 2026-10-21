import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from sentiment.logging_conf import init_logger

logger = init_logger(name=__name__, level=logging.INFO)

class BaseLoader(ABC):
    def load(self, config: dict) -> pd.DataFrame:
        df = self._read(Path(config["file_name"]))
        df = df.rename(columns={v: k for k, v in config["columns_map"].items()})
        df = df[["text", "sentiment"]].copy()
        df["sentiment"] = config["label_transform"](df["sentiment"])
        return df

    @abstractmethod
    def _read(self, path: Path) -> pd.DataFrame: ...


class CSVLoader(BaseLoader):
    def _read(self, path):
        return pd.read_csv(path)


LOADERS = {
    ".csv": CSVLoader,
}

def get_loader(path) -> BaseLoader:
    logger.info("getting loader", extra={"path": path})   
    suffix = Path(path).suffix.lower()
    try:
        return LOADERS[suffix]()
    except KeyError:
        raise ValueError(f"Unsupported file format: {suffix}")

def split_frame(frame: pd.DataFrame, train_size: float):
    train, rem = train_test_split(
        frame, 
        train_size=train_size,
        stratify=frame["sentiment"],
        random_state=42
    )
    val, test = train_test_split(
        rem, 
        train_size=0.5,
        stratify=rem["sentiment"],
        random_state=42
    )
    return train, val, test

def load_splits(configs: dict):
    train = []
    val = []
    test = []
    for name, cfg in configs.items():
        logger.info("loading entry in config", extra={"source": name, "train_size": cfg["train_size"]})
        df = get_loader(cfg["file_name"]).load(cfg)
        df["source"] = name
        df = df.dropna(subset=["text", "sentiment"]).drop_duplicates(subset="text")
        s_train, s_val, s_test = split_frame(frame=df, train_size=cfg["train_size"])          
        train.append(s_train)
        val.append(s_val)
        test.append(s_test)

    train = pd.concat(train, ignore_index=True).drop_duplicates(subset="text")
    val = pd.concat(val, ignore_index=True).drop_duplicates(subset="text")
    test = pd.concat(test, ignore_index=True).drop_duplicates(subset="text")

    val = val[~val["text"].isin(train["text"])].reset_index(drop=True)
    test = test[~test["text"].isin(train["text"]) & ~test["text"].isin(val["text"])].reset_index(drop=True)

    train = train.sample(frac=1, random_state=42).reset_index(drop=True)

    return train, val, test
