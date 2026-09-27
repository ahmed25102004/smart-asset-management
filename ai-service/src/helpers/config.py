from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "AI Maintenance Service"
    OPENAI_API_KEY: str = ""
    VECTOR_DB_URL: str = "http://localhost:6333"
    
    class Config:
        env_file = ".env"

def get_settings():
    return Settings()
