from functools import lru_cache
from zoneinfo import ZoneInfo

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://finance:finance@localhost:5432/finance_rag"

    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    embed_dim: int = 1024

    generator_backend: str = "ollama"
    generator_model: str = "qwen3:4b"
    generator_num_ctx: int = 32768

    anthropic_model: str = "claude-opus-5"

    slack_bot_token: str = ""
    slack_channel_id: str = ""

    timezone: str = "Asia/Taipei"
    context_token_budget: int = 24000

    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


@lru_cache
def get_settings() -> Settings:
    return Settings()
