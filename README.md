# Constitution GPT

An open-source legal research assistant for exploring Nepal's Constitution and
related law through clear, source-grounded answers.

[Live application](https://constitution.subigyasubedi.com.np) ·
[API documentation](https://constitution-gpt-w91k.onrender.com/docs)

Constitution GPT is built for questions such as:

- How is the Prime Minister appointed?
- Which fundamental rights are guaranteed to citizens?
- How is the Federal Parliament structured?
- What constitutional duties do citizens have?
- How can a person apply for Nepali citizenship under current law?
- Does a parliamentary bill appear consistent with relevant constitutional provisions?
- What did a Nepalese court decide about a constitutional issue?

Instead of asking a language model to answer from memory, the application routes
each request through a bounded evidence pipeline. Constitutional questions use
the indexed constitutional text. Related legal, judicial, parliamentary, and
current-information questions use bounded web research with provider-returned
source links. Mixed questions keep the two evidence types visibly separate.

> Constitution GPT is an educational and research tool, not a substitute for
> the official Constitution or professional legal advice.

## What the project demonstrates

- Hierarchy-aware PDF ingestion that preserves Part, Article, Sub-article, and
  Clause metadata.
- Hybrid retrieval combining semantic search, lexical search, and exact
  constitutional citation matching.
- Reciprocal-rank fusion, deduplication, reranking, and parent-article expansion.
- Structured OpenAI responses with deterministic citation validation and a
  second groundedness check.
- Deterministic conversation and research routing without an autonomous agent loop.
- Follow-up resolution using persisted conversation context.
- Bounded live web research with structured source metadata.
- Multi-issue constitutional retrieval and completeness verification for legal hypotheticals.
- Intent-sized current-fact answers requiring concise official-source evidence.
- Prompt-injection defenses across user input, retrieved documents, and model
  output.
- A FastAPI backend with CORS validation, concurrency limits, timeouts, and
  liveness/readiness endpoints.
- A responsive Next.js chat interface with loading, error, copy, and new-chat
  states.

## How it works

```text
User
  │
  ▼
Next.js chat interface
  │  POST /api/chat
  ▼
FastAPI service
  │
  ├── handle basic conversation deterministically
  ├── resolve follow-ups into standalone questions
  ├── classify and sanitize the legal information need
  ├── constitutional route → retrieve and validate Chroma evidence
  ├── legal-research route → bounded web search with citations
  └── mixed route → present both evidence sections
  │
  ▼
Evidence-backed answer with conversation and source metadata
```

## Technology stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4 |
| API | FastAPI, Uvicorn, Pydantic |
| RAG orchestration | LangChain |
| Vector search | Chroma Cloud |
| Embeddings | OpenAI `text-embedding-3-small` |
| Answer generation | OpenAI `gpt-4o` |
| Live legal research | OpenAI Responses API web search |
| Conversation persistence | PostgreSQL |
| Source document | Constitution of Nepal (English PDF) |

## Run it locally

### Prerequisites

Install these before starting:

- Git
- Python 3.11 or newer
- Node.js 20 or newer with npm
- An OpenAI API key
- A Chroma Cloud account
- A PostgreSQL connection URL required by the current API runtime

### 1. Clone the repository

```bash
git clone https://github.com/subigya-js/constitution-gpt.git
cd constitution-gpt
```

All commands below assume that your terminal is in the repository root unless
the step explicitly says otherwise.

### 2. Create the Python environment

macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Using `python -m pip` ensures packages are installed into the same Python
interpreter that runs the API.

### 3. Configure the backend

Copy the example file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Open `.env` and replace every placeholder in the required section:

```env
# OpenAI creates embeddings and generates the final answer.
OPENAI_API_KEY=your_openai_api_key

# Only these browser origins may call the API.
FRONTEND_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Chroma Cloud stores and searches the indexed Constitution.
CHROMA_API_KEY=your_chroma_cloud_api_key
CHROMA_TENANT=your_chroma_tenant
CHROMA_DATABASE=your_chroma_database
CHROMA_COLLECTION=constitution_english

# PostgreSQL connection used by the current API runtime.
DATABASE_URL=postgresql://username:password@host:5432/database
```

What each required variable means:

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` | Secret key used by the backend for embeddings and answers |
| `FRONTEND_ORIGINS` | Comma-separated frontend origins allowed by browser CORS |
| `CHROMA_API_KEY` | Secret key from the Chroma Cloud dashboard |
| `CHROMA_TENANT` | Tenant identifier shown in Chroma Cloud |
| `CHROMA_DATABASE` | Chroma database containing the collection |
| `CHROMA_COLLECTION` | Collection name; defaults to `constitution_english` |
| `DATABASE_URL` | A PostgreSQL URL reachable from the machine running the API |

Do not commit `.env`. Never put OpenAI, Chroma, or PostgreSQL secrets in a
variable beginning with `NEXT_PUBLIC_`; those variables are exposed to the
browser.

The remaining values in `.env.example` are optional operational controls. Their
defaults are suitable for local development:

| Variable | Default | Purpose |
|---|---:|---|
| `OPENAI_REQUEST_TIMEOUT_SECONDS` | `45` | Timeout for an OpenAI SDK operation |
| `OPENAI_MAX_RETRIES` | `2` | Retry count for transient OpenAI failures |
| `RESEARCH_MODEL` | `gpt-4o` | Model used for bounded web research |
| `OPENAI_WEB_SEARCH_TOOL` | `web_search` | Responses API web-search tool name |
| `WEB_SEARCH_CONTEXT_SIZE` | `medium` | Web-search context depth |
| `RESEARCH_ALLOWED_DOMAINS` | unset | Optional comma-separated web domain allowlist |
| `MAX_CONCURRENT_RAG_REQUESTS` | `3` | Maximum RAG jobs per API process |
| `RAG_QUEUE_TIMEOUT_SECONDS` | `1` | Maximum wait for an available execution slot |
| `RAG_REQUEST_TIMEOUT_SECONDS` | `90` | Deadline for a complete chat request |
| `CHROMA_HEALTH_TIMEOUT_SECONDS` | `5` | Chroma readiness-check timeout |
| `CHROMA_HOST` | Chroma default | Optional custom Chroma Cloud host |

### 4. Index the Constitution

The source PDF is located at `rag/data/Constitution_English.pdf`. Upload its
hierarchy-aware chunks to your configured Chroma Cloud collection:

```bash
python rag/ingestion_pipeline.py
```

This step calls the OpenAI embeddings API and can incur usage charges. Run it
once for a new collection, and run it again only when the source document or
chunking logic changes. Stable chunk IDs make repeated uploads idempotent.

### 5. Configure the frontend

```bash
cd web
cp .env.example .env.local
npm install
```

The local frontend environment should contain:

```env
# Public origin of the FastAPI service. Do not append /api/chat.
NEXT_PUBLIC_API_URL=http://localhost:8000
```

This value is safe to expose because it is only an API address, not a secret.

### 6. Start both applications

Open two terminals.

Terminal 1 — backend, from the repository root:

```bash
source venv/bin/activate
python -m uvicorn api.main:app --reload --port 8000
```

Terminal 2 — frontend:

```bash
cd web
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) and ask a constitutional
question.

## Verify the setup

Check that the API process is running:

```bash
curl http://localhost:8000/health/live
```

Check that required external dependencies are reachable:

```bash
curl http://localhost:8000/health/ready
```

A ready response looks like:

```json
{
  "status": "ready",
  "service": "Constitution GPT API",
  "dependencies": {
    "chroma": "ok",
    "postgres": "ok"
  }
}
```

You can also open [http://localhost:8000/docs](http://localhost:8000/docs) to
try the API through FastAPI's interactive documentation.

## API

### `POST /api/chat`

Request:

```json
{
  "question": "How is the Prime Minister appointed in Nepal?",
  "conversation_id": null
}
```

Response:

```json
{
  "question": "How is the Prime Minister appointed in Nepal?",
  "answer": "The President appoints the Prime Minister under Article 76...",
  "conversation_id": "f8d53b9b-dbae-4477-b7a4-bf7c30c2b411",
  "resolved_question": "How is the Prime Minister appointed in Nepal?",
  "mode": "constitutional",
  "sources": []
}
```

Send the returned `conversation_id` with the next request so pronouns and omitted
subjects in follow-up questions can be resolved. `mode` is one of
`conversation`, `constitutional`, `legal_research`, `mixed`, or `boundary`.
For web-researched answers, `sources` contains the source title, URL, and type.

Other endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /` | API information |
| `GET /health/live` | Confirms that the API process is alive |
| `GET /health/ready` | Checks dependencies required to serve chat requests |
| `GET /docs` | Interactive OpenAPI documentation |

## Project structure

```text
constitution-gpt/
├── api/
│   ├── main.py                 # FastAPI routes and application lifecycle
│   ├── chat_repository.py      # Conversations, messages, and source persistence
│   └── execution_limits.py     # Concurrency and timeout protection
├── rag/
│   ├── data/                   # Source Constitution PDF
│   ├── ingestion_pipeline.py   # PDF parsing, chunking, and indexing
│   ├── hybrid_retrieval.py     # Semantic, lexical, and citation retrieval
│   ├── prompt_security.py      # Input, context, and output safeguards
│   ├── research_assistant.py   # Conversation, follow-up, and research orchestration
│   └── retrieval_pipeline.py   # Routing, generation, and verification
├── web/
│   ├── app/                    # Next.js application and chat interface
│   └── package.json
├── .env.example                # Backend environment template
├── requirements.txt            # Python dependencies
└── README.md
```

## Testing

Run the deterministic backend and RAG tests from the repository root:

```bash
python -m unittest \
  api.test_chat_repository \
  api.test_chat_persistence \
  api.test_execution_limits \
  rag.test_answer_formatting \
  rag.test_chroma_connection \
  rag.test_hybrid_retrieval \
  rag.test_prompt_security \
  rag.test_research_assistant \
  rag.test_legal_issue_research
```

Build and type-check the frontend:

```bash
cd web
npm run build
```

The prompt-security tests are deterministic regression checks. No prompt
defense guarantees protection from every future attack, so avoid placing
secrets in prompts or granting the answer-generation model privileged tools.

## Troubleshooting

### `ModuleNotFoundError`

The dependency was probably installed into a different Python interpreter:

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
python -c "import sys; print(sys.executable)"
```

### The frontend reports that it cannot reach the API

Confirm all three items:

1. FastAPI is running on `http://localhost:8000`.
2. `web/.env.local` contains `NEXT_PUBLIC_API_URL=http://localhost:8000`.
3. You restarted Next.js after changing `web/.env.local`.

### Browser CORS error

Add the exact frontend origin to `FRONTEND_ORIGINS`, without URL paths or a
trailing slash, and restart FastAPI.

### `/health/ready` returns `503`

At least one required dependency is unavailable. Inspect the backend terminal
for the server-side error, then verify your Chroma and PostgreSQL configuration.
The endpoint intentionally does not expose credentials or internal error
details to browsers.

## Roadmap

- [x] Hierarchical constitutional document ingestion
- [x] Hybrid semantic and lexical retrieval
- [x] Exact Article/Sub-article lookup
- [x] Structured, citation-validated answers
- [x] Prompt-injection regression suite
- [x] FastAPI service and Next.js chat interface
- [x] Basic conversation and multi-turn follow-up resolution
- [x] Bounded web research for related Nepalese legal questions
- [x] Structured web source links in API and UI
- [x] Cloud deployment
- [ ] Nepali-language Constitution and answers
- [ ] Support for additional constitutions and legal documents
- [ ] Authentication and personal conversation history
- [ ] Dedicated statutes, regulations, bills, and judgments corpus
- [ ] Autonomous multi-step research agent with explicit tool policies
- [ ] Automated retrieval-quality evaluation in CI
- [ ] Accessibility and end-to-end browser testing

## Contributing

Issues and pull requests are welcome. For substantial changes, open an issue
first so the implementation approach and test coverage can be discussed.

When contributing:

1. Create a focused branch.
2. Add or update tests with the change.
3. Run the backend tests and frontend build.
4. Submit a pull request explaining the problem, solution, and tradeoffs.

## Acknowledgements

- [The Constitution of Nepal](https://ag.gov.np/files/Constitution-of-Nepal_2072_Eng_www.moljpa.gov_.npDate-72_11_16.pdf)
- [OpenAI](https://openai.com/)
- [Chroma](https://www.trychroma.com/)
- The open-source LangChain, FastAPI, Next.js, and React communities
