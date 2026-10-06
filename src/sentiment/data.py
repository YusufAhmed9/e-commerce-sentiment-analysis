import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .logging_conf import init_logger

logger = init_logger(name=__name__, level=logging.INFO)

class BaseLoader(ABC):
    def load(self, config: dict) -> pd.DataFrame:
        df = self._read(Path(config["path"]))
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
    suffix = Path(path).suffix.lower()
    try:
        return LOADERS[suffix]()
    except KeyError:
        raise ValueError(f"Unsupported file format: {suffix}")

def load_all(configs: dict) -> pd.DataFrame:
    logger.info("loading data")
    frames = []
    for name, cfg in configs.items():
        df = get_loader(cfg["path"]).load(cfg)
        df["source"] = name                      
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
