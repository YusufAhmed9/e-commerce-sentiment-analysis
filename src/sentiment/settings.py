from pydantic_settings import BaseSettings, SettingsConfigDict

from sentiment.config import ROOT_DIR
from sentiment.export import ExportType


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file= ROOT_DIR / ".env")
    
    MODEL_NAME: str
    VECTORIZER_PATH: str
    EXPORT_TYPE: ExportType

settings = Settings()