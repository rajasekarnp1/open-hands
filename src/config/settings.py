from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API Server Settings
    ADMIN_TOKEN: Optional[str] = None
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]

    # Database Settings
    DATABASE_URL: str = "sqlite:///./model_memory.db"
    REDIS_URL: Optional[str] = None

    # General Application Settings
    LOG_LEVEL: str = "INFO"

    # LLM Aggregator Settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0

    # Meta Controller Settings
    META_CONTROLLER_LEARNING_RATE: float = 0.1
    META_CONTROLLER_EXPLORATION_RATE: float = 0.1

    # Auto Updater Settings
    AUTO_UPDATE_INTERVAL_MINUTES: int = 60

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore')

settings = Settings()
