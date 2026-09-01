from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./loreguard.db"
    redis_url: str = "redis://localhost:6379/0"
    use_celery: bool = False
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    enable_model_extraction: bool = False
    provider_timeout_seconds: float = 45
    provider_max_attempts: int = 3
    model_chunk_max_chars: int = 6000
    model_chunk_overlap_lines: int = 1
    model_max_chunks_per_document: int = 24
    per_run_token_budget: int = 20_000
    daily_token_budget: int = 100_000
    model_input_price_per_million: float | None = None
    model_output_price_per_million: float | None = None
    max_upload_bytes: int = 10 * 1024 * 1024
    diff_max_lines_per_version: int = 20_000
    diff_max_chars_per_version: int = 2_000_000
    diff_max_output_lines: int = 4_000
    rate_limit_per_minute: int = 30
    rate_limit_window_seconds: float = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
