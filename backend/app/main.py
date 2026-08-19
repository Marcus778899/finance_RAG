from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings

ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="finance-rag")
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, str | int]:
        return {
            "status": "ok",
            "embed_model": settings.embed_model,
            "embed_dim": settings.embed_dim,
            "generator_backend": settings.generator_backend,
            "generator_model": settings.generator_model,
            "timezone": settings.timezone,
        }

    return app


app = create_app()
