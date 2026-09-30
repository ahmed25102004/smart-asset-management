import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "Smart Asset AI & RAG Service"
    OPENAI_API_KEY: str = ""
    VECTOR_DB_TYPE: str = "chroma"  # 'chroma' or 'qdrant'
    VECTOR_DB_URL: str = "http://localhost:6333"
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    COLLECTION_NAME: str = "machine_manuals"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    ENVIRONMENT: str = "development"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()
