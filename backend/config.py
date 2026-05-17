from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    app_secret_key: str = Field(min_length=16)
    app_base_url: str = "http://localhost:8000"

    database_url: str = "sqlite:///./pinchtab.db"

    worker_base_url: str = "http://localhost:9090"
    worker_shared_secret: str = Field(min_length=16)
    max_concurrent_sessions: int = 10
    session_idle_timeout_seconds: int = 30
    session_hard_cap_minutes: int = 30

    resend_api_key: str = ""
    email_from: str = "noreply@pinchtab.local"

    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_starter: str = ""
    stripe_price_pro: str = ""

    rate_limit_per_user: str = "60/minute"


@lru_cache
def get_settings() -> Settings:
    return Settings()
