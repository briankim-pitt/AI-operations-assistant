from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-5-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    google_service_account_file: Path
    google_drive_folder_id: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()