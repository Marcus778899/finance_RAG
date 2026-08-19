from functools import lru_cache
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "finance_rag"
    postgres_user: str = "finance"
    postgres_password: SecretStr

    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    embed_dim: int = 1024

    generator_backend: str = "ollama"
    generator_model: str = "qwen3:4b"
    generator_num_ctx: int = 32768

    anthropic_model: str = "claude-opus-5"
    anthropic_api_key: SecretStr = SecretStr("")

    slack_bot_token: SecretStr = SecretStr("")
    slack_channel_id: str = ""

    timezone: str = "Asia/Taipei"
    context_token_budget: int = 24000

    @property
    def database_url(self) -> str:
        password = quote_plus(self.postgres_password.get_secret_value())
        return (
            f"postgresql+asyncpg://{quote_plus(self.postgres_user)}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


@lru_cache
def get_settings() -> Settings:
    return Settings()
