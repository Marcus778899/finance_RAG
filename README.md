# finance_RAG

Slack 金融資訊 RAG demo：每天早上進 Slack channel 的金融摘要 (約 4000~5000 字，中英混合)
會被 ingest 進 pgvector，並提供自然語言查詢介面。

## 快速開始

```bash
docker compose up -d
uv sync --all-extras
cp .env.example .env          # 填入 SLACK_BOT_TOKEN 與 SLACK_CHANNEL_ID
uv run uvicorn app.main:app --reload --app-dir backend
```

需要本地 Ollama 並已 pull `bge-m3` 與 `qwen3:4b`：

```bash
ollama pull bge-m3
ollama pull qwen3:4b
```

## 開發

專案規範與架構說明見 [CLAUDE.md](CLAUDE.md)。
