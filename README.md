# FinSight AI 📊

> Automated Multi-Agent Financial Intelligence Platform

FinSight AI is a production-ready Python system that autonomously fetches SEC EDGAR filings, market data, and news — then runs 4 specialist LangGraph agents in parallel to synthesize a structured investment intelligence report.

## Architecture

```
User Input (ticker)
      │
      ▼
FastAPI REST API  ──────────────────────────────────────────────►  Streamlit Dashboard
      │                                                                    │
      ▼                                                                    │
LangGraph StateGraph                                                       │
      │                                                                    │
  [orchestrator] → resolve CIK, validate ticker                           │
      │                                                                    │
  [ingestor]     → SEC EDGAR + yfinance + RSS news → ChromaDB             │
      │                                                                    │
  ┌───┼───┐  (parallel via Send API)                                       │
  ▼   ▼   ▼                                                                │
[fin][risk][sent] → RAG retrieval + Gemini Flash                          │
  └───┴───┘                                                                │
      │                                                                    │
  [synthesis]    → final report via Gemini Flash                           │
      │                                                                    │
  SQLite + PDF ──────────────────────────────────────────────────────────►┘
```

## Tech Stack

| Layer | Library |
|---|---|
| LLM | `google-generativeai` (Gemini 1.5 Flash) |
| Embeddings | `text-embedding-004` |
| Agents | `langgraph` + `langchain-core` |
| Vector DB | `chromadb` (local persistent) |
| MCP Server | `fastmcp` |
| REST API | `fastapi` + `uvicorn` |
| Dashboard | `streamlit` + `plotly` |
| PDF | `weasyprint` + `jinja2` |
| Market data | `yfinance` |
| SEC filings | `httpx` (direct EDGAR REST API) |
| News | `feedparser` (Google News RSS) |
| Database | `sqlalchemy` + `aiosqlite` (SQLite) |
| Cache | `diskcache` |
| Scheduler | `apscheduler` |

## Prerequisites

- Python 3.11+
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/)

## Setup

### 1. Clone or create the project directory

```bash
cd "D:\RESUME PROJECTS\finsight_ai"
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for WeasyPrint on Windows:** You may need to install GTK libraries. See [WeasyPrint docs](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows).

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your **GEMINI_API_KEY**:

```env
GEMINI_API_KEY=your_actual_key_here
```

### 5. Initialize the database

```bash
python -c "from db.database import init_db; init_db()"
```

## Running the Application

Open **3 terminals** from the project root:

### Terminal 1 — FastAPI Server

```bash
uvicorn app.main:app --reload --port 8000
```

### Terminal 2 — MCP Server (optional, agents use in-process calls by default)

```bash
python mcp_server/server.py
```

### Terminal 3 — Streamlit Dashboard

```bash
streamlit run dashboard/Home.py
```

The dashboard will open at **http://localhost:8501**

## Usage

### Via curl (API)

```bash
# Start analysis
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "force_refresh": false}'

# Returns: {"run_id": "...", "status": "PENDING", "ticker": "AAPL"}

# Poll for status
curl http://localhost:8000/api/v1/runs/{run_id}

# Get report JSON
curl http://localhost:8000/api/v1/reports/{run_id}

# Download PDF
curl -o report.pdf http://localhost:8000/api/v1/reports/{run_id}/pdf
```

### Via Streamlit Dashboard

1. Navigate to http://localhost:8501
2. Enter a ticker (e.g., `AAPL`, `MSFT`, `GOOGL`)
3. Click **Analyze**
4. Track progress on the **Analyze** page
5. View results in the **Report Viewer** tab

## Running Tests

```bash
pytest tests/ -v
```

### Test Coverage

| Module | Tests |
|---|---|
| `analysis/financial_ratios.py` | Ratio computation, NaN handling, edge cases |
| `rag/chunker.py` | Section detection, chunk sizes, overlap |
| `rag/retriever.py` | BM25 scoring, hybrid retrieval, error handling |

## API Reference

### POST `/api/v1/analyze`

Launch a new financial analysis run.

```json
{
  "ticker": "AAPL",
  "force_refresh": false
}
```

Returns `202 Accepted` with `run_id`.

### GET `/api/v1/runs/{run_id}`

Get analysis run status. Statuses: `PENDING`, `RUNNING`, `COMPLETE`, `FAILED`.

### GET `/api/v1/reports/{run_id}`

Get the full FinSightReport JSON when status is `COMPLETE`.

### GET `/api/v1/reports/{run_id}/pdf`

Download the PDF report file.

### GET `/api/v1/reports/ticker/{ticker}`

List all completed reports for a ticker.

### Alerts API

```
POST   /api/v1/alerts/configs           Create alert config
GET    /api/v1/alerts/configs           List all configs
PATCH  /api/v1/alerts/configs/{id}/toggle  Enable/disable
DELETE /api/v1/alerts/configs/{id}     Delete config
GET    /api/v1/alerts/events            List alert events
PATCH  /api/v1/alerts/events/{id}/acknowledge  Acknowledge event
```

## Project Structure

```
finsight_ai/
├── app/                    # FastAPI application
│   ├── config.py           # Settings (pydantic-settings)
│   ├── main.py             # App factory + lifespan
│   ├── routers/            # API routes
│   └── schemas/            # Pydantic v2 models
├── agents/                 # LangGraph agent nodes
│   ├── graph.py            # StateGraph topology
│   ├── state.py            # FinSightState TypedDict
│   ├── orchestrator.py     # CIK resolution
│   ├── ingestor.py         # Data fetching + embedding
│   ├── financial.py        # Financial analysis agent
│   ├── risk.py             # Risk extraction agent
│   ├── sentiment.py        # Sentiment analysis agent
│   └── synthesis.py        # Report synthesis agent
├── mcp_server/             # FastMCP tools
│   ├── server.py           # MCP server
│   ├── client.py           # In-process client
│   └── tools/              # Tool implementations
├── rag/                    # RAG pipeline
│   ├── chunker.py          # Section-aware chunking
│   ├── embedder.py         # Gemini embeddings
│   ├── vectorstore.py      # ChromaDB wrapper
│   └── retriever.py        # Hybrid BM25+semantic
├── analysis/               # Pure analytics
│   ├── financial_ratios.py # Ratio computations
│   ├── sentiment_scoring.py # Sentiment aggregation
│   └── anomaly_detection.py # Statistical flagging
├── delivery/               # Output delivery
│   ├── pdf_renderer.py     # WeasyPrint PDF
│   ├── alert_engine.py     # APScheduler alerts
│   └── templates/          # Jinja2 HTML template
├── dashboard/              # Streamlit UI
│   ├── Home.py             # Main page
│   └── pages/              # Additional pages
├── db/                     # Database layer
│   ├── database.py         # SQLAlchemy engine
│   └── models.py           # ORM models
├── cache/                  # Caching layer
│   └── disk_cache.py       # diskcache wrapper
├── prompts/                # LLM prompt templates
└── tests/                  # pytest test suite
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | ✅ | — | Google AI Studio API key |
| `NEWSAPI_KEY` | ❌ | — | NewsAPI.org key (optional) |
| `FRED_API_KEY` | ❌ | — | FRED API key (optional) |
| `DATABASE_URL` | ❌ | `sqlite:///./finsight.db` | SQLite DB path |
| `CHROMA_PERSIST_DIR` | ❌ | `.chroma_data` | ChromaDB storage |
| `CACHE_DIR` | ❌ | `.cache_data` | diskcache directory |
| `PDF_OUTPUT_DIR` | ❌ | `.reports` | PDF output directory |
| `LOG_LEVEL` | ❌ | `INFO` | Logging level |

## Data Sources

- **SEC EDGAR** — 10-K, 10-Q, 8-K filings (free, no API key)
- **yfinance** — Market prices, financial statements (free)
- **Google News RSS** — Recent news articles (free)
- **Yahoo Finance RSS** — Ticker-specific news (free)
- **NewsAPI** — Broader news coverage (optional free tier)

## Limitations (MVP)

- Filing text is limited to first 100,000 characters per filing
- News is fetched for the last 60 days
- PDF generation requires WeasyPrint and GTK (see setup notes for Windows)
- Analysis takes 2-5 minutes depending on data availability and Gemini API rate limits

## License

MIT License — See LICENSE file for details.

---

Built with ❤️ using LangGraph, Gemini, and FastAPI.
