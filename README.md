# finance_RAG

Slack 金融資訊 RAG demo：每天早上進 Slack channel 的金融摘要 (約 4000~5000 字，中英混合)
會被 ingest 進 pgvector，並提供自然語言查詢介面，例如「請問昨天的美國有什麼重大財經新聞」。

## 架構

| 層 | 選型 |
| --- | --- |
| Embedding | 本地 Ollama `bge-m3`，1024 維 |
| Generator | 本地 Ollama `qwen3:4b`，介面化可切 Claude API |
| Backend | FastAPI + SQLAlchemy async + pgvector |
| Frontend | Vite + React + TypeScript |
| Ingestion | 每日 cron 呼叫 Slack `conversations.history` |

### 檢索設計

這不是純向量 RAG。一天的內容只有約 4000 tokens，直接放進 context 比檢索更準，而
「昨天」這類問句的主角是日期過濾、不是語意相似度。`retrieval/query_parser.py` 先解析
出日期區間，再由 `retrieval/retriever.py` 分流：

1. 有日期且跨度 <= 7 天：直接 SQL 撈該區間全部 chunks，不做向量比對
2. 有日期但跨度過寬：daily digest 當骨幹，向量檢索補細節
3. 無日期：純向量檢索 top-k

## 快速開始

需要本地 Ollama 並已 pull 模型：

```bash
ollama pull bge-m3
ollama pull qwen3:4b
```

啟動：

```bash
cp .env.example .env      # 至少要填 POSTGRES_PASSWORD、SLACK_BOT_TOKEN、SLACK_CHANNEL_ID
docker compose up -d
uv sync --all-extras
uv run uvicorn app.main:create_app --factory --reload --app-dir backend
```

## 設定

所有設定集中在 `.env`，範本見 `.env.example`。專案內不存在寫死的帳號密碼：
`docker-compose.yml` 由 `.env` 取值，`POSTGRES_PASSWORD` 沒有預設值，未設定時
compose 會直接報錯而不是靜默使用弱密碼。後端從這些欄位組出 `DATABASE_URL`，
密碼、API key、Slack token 一律以 `SecretStr` 保存，不會出現在 log 或 repr 中。

## 已知陷阱

- **Ollama `num_ctx` 會靜默截斷**。qwen3:4b 模型本身支援 262144 context，但 Ollama
  runtime 預設只給 4096。檢索到數日內容輕易超過，超出部分會被無聲丟棄而模型仍產出
  看似合理的答案。所有 generate 呼叫必須顯式帶 `options.num_ctx`（設定值
  `GENERATOR_NUM_CTX`，預設 32768）。
- **qwen3 的 thinking 輸出**用 API 參數 `think: false` 關閉，不要用 regex 剝 `<think>` 標籤。
- **時區**。Slack `ts` 是 UTC epoch，但「昨天」指的是 `Asia/Taipei` 的昨天。日期邊界
  一律經 `settings.tzinfo` 換算後再查詢。

## 開發

### 分支流程

```
main (穩定 / milestone)  <-  develop (整合)  <-  feature/*
```

feature 分支開 PR 進 `develop`；`develop` 累積到里程碑再開 PR 進 `main`。
不直接 commit 到 `main` 或 `develop`，pre-commit 的 `no-commit-to-branch` 會擋下。

### 程式風格

註解能不寫就不寫，只在邏輯確實不直觀時留一行。專案內不使用 emoji，包含程式碼、
commit message、PR 內容與文件。ruff 負責 lint 與 format，行寬 100。

### 測試

```bash
uv run pytest                 # 測試 + 全域 coverage 門檻 70%
uv run ruff check . --fix     # lint
pre-commit run --all-files    # 全部 hook
```

核心邏輯（`retrieval/`、`ingest/chunker.py`、`ingest/digest.py`）另有 90% 門檻，
由 CI 第二道閘門把關。測試不得連 ollama 或 Slack，一律注入 `providers/fake.py`；
DB 測試以 testcontainers 起真的 `pgvector/pgvector:pg17`。

## License

[MIT](LICENSE)
