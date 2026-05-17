from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    app_secret_key: str = Field(min_length=16)
    app_base_url: str = "http://localhost:8000"

    database_url: str = "sqlite:///./pinchtab.db"

    # The pinchtab daemon. Defaults to localhost in dev; in compose it's
    # set to http://pinchtab:9867 via the docker-internal hostname.
    worker_base_url: str = "http://127.0.0.1:9867"
    # Pinchtab 0.8+ requires bearer auth even on localhost. Read from
    # ~/.pinchtab/config.json server.token when running locally; in compose
    # set explicitly via env var.
    pinchtab_token: str = ""
    # Retained for backward compatibility with .env templates; no longer
    # used after the worker-per-container model was retired in favor of
    # the pinchtab daemon model.
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

    # LLM backend is selected per-request based on whether the POST /tasks
    # body provides an `anthropic_api_key`:
    #   key present  → AsyncAnthropic (any authenticated user, BYO API tier)
    #   key absent   → ClaudeCLIProvider (operator-only, uses subscription)
    # `llm_provider` field is retained for backward compat (.env files);
    # it is currently unused by the routing logic.
    llm_provider: str = "anthropic_sdk"
    # Email allowed to submit tasks without an API key (CLI fallback).
    # Empty → no one may omit the key.
    operator_email: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
