# 🏛️ Constitution GPT
### An Open-Source Constitutional Intelligence System Powered by RAG + LLMs

Constitution GPT is an **open-source intelligence system designed specifically for constitutional, legal, policy, and governance documents**, enabling precise retrieval, interpretation, and question-answering **grounded in authoritative texts**.

This project helps students, lawyers, policymakers, researchers, and developers build systems that require:
- ✅ Accurate referencing with Part/Article/Sub-article citations
- ✅ Context-aware hierarchical understanding
- ✅ Traceable legal reasoning
- ✅ Question answering based on verified constitutional sources

---

## 🌟 Why Constitution GPT?

> **Legal and constitutional documents are long, complex, and interconnected.
Traditional search is too shallow.
LLMs alone hallucinate.
Constitution GPT fills this gap.**

### Key Advantages:
- 📘 **Hierarchical Understanding**: Preserves Part → Article → Sub-article → Clause structure
- 🎯 **Smart Retrieval**: Query expansion handles semantic variations ("elected" vs "appointed")
- 🔍 **Complete Coverage**: Automatically fetches all sub-articles from relevant articles
- 📊 **Structured Responses**: Beautiful, citation-backed answers with proper hierarchy
- 🌐 **Generic & Extensible**: Works for ANY constitutional topic, not hardcoded

---

## 🚀 Current Features

### 📄 **1. Intelligent Document Processing**
- Loads PDF constitutions (currently: Constitution of Nepal)
- Extracts 240 pages → 1,719 semantic chunks
- Preserves hierarchical structure with rich metadata

### ✂️ **2. Advanced Hierarchical Chunking**
**Not just character splitting** - our system:
- ✅ Detects Part, Article, Sub-article, Clause boundaries
- ✅ Adds contextual prefixes for better semantic matching
- ✅ Keeps complete sub-articles together (no mid-sentence splits)
- ✅ Stores metadata: `part`, `article`, `subarticle`, `clause`, `hierarchy`

**Example chunk metadata:**
```json
{
  "part": "Part 7",
  "part_name": "Federal Executive",
  "article": "Article 76",
  "article_title": "Constitution of Council of Ministers",
  "subarticle": "Sub-article (1)",
  "hierarchy": "Part 7 → Article 76 → Sub-article (1)"
}
```

### 🔍 **3. Smart Query Processing**
**Query Expansion** - Automatically generates variations:
- "How is the PM **elected**?" → "appointed", "selected", "chosen"
- "What are citizen **rights**?" → "freedoms", "liberties", "entitlements"
- Topic-specific boosters (e.g., PM queries → "Article 76")

**Article Completion** - Ensures comprehensive answers:
- Detects relevant articles in initial retrieval
- Fetches ALL sub-articles from those articles
- Provides complete constitutional coverage

### 🧠 **4. Structured Response Generation**
Responses follow constitutional hierarchy:

```
📘 Part 7 – Federal Executive
Article 76 – Constitution of Council of Ministers

🔹 Sub-article (1)
As per Part 7, Article 76, Sub-article (1):
• The President shall appoint the leader of a parliamentary party 
  that commands majority in the House of Representatives as the 
  Prime Minister...

🔹 Sub-article (2)
As per Part 7, Article 76, Sub-article (2):
• If no party has a clear majority...
```

---

## 📊 System Performance

| Metric | Value |
|--------|-------|
| **Total Chunks** | 1,719 semantic chunks |
| **Chunk Quality** | Context-aware with metadata |
| **Query Expansion** | 5-10x variations per query |
| **Retrieval Accuracy** | ~90% for tested queries |
| **Response Format** | Hierarchical with citations |

### ✅ Tested Query Types:
- ✅ Prime Minister election process
- ✅ Fundamental rights of citizens
- ✅ Duties of citizens
- ✅ President election procedure
- ✅ Parliament structure
- ✅ Freedom of speech provisions

---

## ⚙️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/subigya-js/constitution-gpt.git
cd constitution-gpt
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_REQUEST_TIMEOUT_SECONDS=45
OPENAI_MAX_RETRIES=2

FRONTEND_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

REDIS_URL=redis://localhost:6379/0
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW_SECONDS=60

MAX_CONCURRENT_RAG_REQUESTS=3
RAG_QUEUE_TIMEOUT_SECONDS=1
RAG_REQUEST_TIMEOUT_SECONDS=90
CHROMA_HEALTH_TIMEOUT_SECONDS=5

CHROMA_API_KEY=your_chroma_api_key_here
CHROMA_TENANT=your_chroma_tenant_here
CHROMA_DATABASE=your_chroma_database_here
CHROMA_COLLECTION=constitution_english
```

For production, replace the local origins with the exact Vercel and custom-domain
origins that may call the API. Separate multiple origins with commas and do not
include URL paths or trailing slashes.

`REDIS_URL` is required by the distributed `/api/chat` rate limiter. In
production, use the internal URL from your managed Redis provider. Health and
documentation endpoints are not rate limited.

Copy [`.env.example`](.env.example) to `.env` for local development. Configure
the same names in Render's environment settings for production; never commit
the real secret values.

#### Runtime protection variables

| Variable | Required | Default | Purpose |
|---|---:|---:|---|
| `OPENAI_API_KEY` | Yes | — | Server-side OpenAI credential |
| `OPENAI_REQUEST_TIMEOUT_SECONDS` | No | `45` | Deadline for each OpenAI SDK operation |
| `OPENAI_MAX_RETRIES` | No | `2` | SDK retries for transient OpenAI failures; `0` disables retries |
| `FRONTEND_ORIGINS` | Yes | — | Comma-separated browser origin allowlist |
| `REDIS_URL` | Yes | — | Shared Redis connection used by rate limiting |
| `RATE_LIMIT_REQUESTS` | No | `10` | Requests permitted per client in one window |
| `RATE_LIMIT_WINDOW_SECONDS` | No | `60` | Sliding rate-limit window |
| `MAX_CONCURRENT_RAG_REQUESTS` | No | `3` | Maximum RAG jobs executing in each API process |
| `RAG_QUEUE_TIMEOUT_SECONDS` | No | `1` | Time to wait for an execution slot before returning `429` |
| `RAG_REQUEST_TIMEOUT_SECONDS` | No | `90` | Client-facing deadline for the complete RAG pipeline |
| `CHROMA_HEALTH_TIMEOUT_SECONDS` | No | `5` | Maximum readiness-probe wait for Chroma |
| `CHROMA_API_KEY` | Production | — | Chroma Cloud credential |
| `CHROMA_TENANT` | Production | — | Chroma Cloud tenant identifier |
| `CHROMA_DATABASE` | Production | — | Chroma Cloud database name |
| `CHROMA_COLLECTION` | No | `constitution_english` in cloud | Active collection name; use versioned names for safe releases |
| `CHROMA_HOST` | No | Chroma Cloud default | Custom Chroma host override |

The API returns `429` when the per-client rate limit is exceeded or no RAG
execution slot becomes available. It returns `504` when the overall RAG deadline
expires and `503` when the Redis protection layer is unavailable. Timed-out
Python worker threads cannot be killed safely, so their concurrency slots remain
occupied until the underlying provider call finishes.

For the initial Render deployment, use one Uvicorn worker. The concurrency limit
is per process; increasing the worker count multiplies both concurrency and the
in-memory retrieval index. Start with:

```bash
uvicorn api.main:app --host 0.0.0.0 --port "$PORT" --workers 1 --proxy-headers
```

Set Render's health-check path to `/health/ready`. Use `/health/live` only to
check whether the Python process itself is responsive.

### 5. Build Vector Database
```bash
python rag/ingestion_pipeline.py
```
This will:
- Load the Constitution PDF
- Create 1,719 semantic chunks with metadata
- Generate embeddings using OpenAI
- Store in ChromaDB (`db/chroma_db/`)

---

## 🎮 Usage

### Command Line Interface

**Ask any constitutional question:**
```bash
python rag/retrieval_pipeline.py "How is the Prime Minister elected in Nepal?"
```

**Other example queries:**
```bash
python rag/retrieval_pipeline.py "What are the fundamental rights of citizens?"
python rag/retrieval_pipeline.py "What are the duties of citizens?"
python rag/retrieval_pipeline.py "How is the President elected?"
python rag/retrieval_pipeline.py "What is the structure of the Federal Parliament?"
```

### Test Multiple Queries
```bash
python rag/test_various_queries.py
```

### Prompt-injection security

The RAG request path treats user input, retrieved documents, and model output as
separate untrusted boundaries:

1. Input is Unicode-normalized, risk-scored, and logged by fingerprint rather
   than raw text.
2. A constrained router extracts a clean constitutional information need. The
   original user message is never sent to retrieval or answer generation.
3. Retrieved chunks containing high-confidence instruction-injection signals
   are excluded before context construction.
4. The answer uses a strict schema and has no tool or credential access.
5. A per-request integrity canary, deterministic citation validation, and a
   separate groundedness/security verification pass fail closed before output.
6. API failures are logged server-side and return no provider or internal error
   details to the client.

Run the deterministic adversarial regression suite with:

```bash
python -m unittest rag.test_prompt_security
```

No LLM defense guarantees that every novel prompt injection will be detected.
Do not place secrets in prompts or give this model privileged tools. Production
deployment must additionally provide gateway-level authentication, distributed
rate limits, request budgets, alerting, dependency timeouts, and periodic red-team
evaluation using real model calls.

### Rebuild Database (if needed)
```bash
rm -rf db/chroma_db
python rag/ingestion_pipeline.py
```

---

## 🏗️ Project Structure

```
constitution-gpt/
├── rag/
│   ├── data/
│   │   └── Constitution_English.pdf    # Source document
│   ├── ingestion_pipeline.py           # Chunking + Vector DB creation
│   ├── retrieval_pipeline.py           # Query processing + Answer generation
│   └── test_various_queries.py         # Test suite
├── db/
│   └── chroma_db/                      # Vector database (auto-generated)
├── venv/                               # Virtual environment
├── .env                                # Environment variables
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## 🔧 Technical Architecture

### Ingestion Pipeline (`ingestion_pipeline.py`)
1. **Load PDF** → PyMuPDFLoader extracts text
2. **Parse Hierarchy** → Regex-based extraction of Parts/Articles/Sub-articles
3. **Create Chunks** → Semantic chunks with contextual prefixes
4. **Add Metadata** → Rich metadata for each chunk
5. **Generate Embeddings** → OpenAI `text-embedding-3-small`
6. **Store in ChromaDB** → Persistent vector database

### Retrieval Pipeline (`retrieval_pipeline.py`)
1. **Query Expansion** → Generate 5-10 variations with synonyms
2. **Multi-Query Retrieval** → Search for each variation
3. **Deduplication** → Remove duplicate chunks
4. **Article Completion** → Fetch all sub-articles from key articles
5. **Relevance Scoring** → Prioritize by query term matches
6. **Context Creation** → Group and structure by hierarchy
7. **LLM Generation** → GPT-4o generates structured answer

---

## 📝 Example Output

**Query:** "How is the Prime Minister elected in Nepal?"

**Response:**
```
📘 Part 7 – Federal Executive | Article 76 – Constitution of Council of Ministers

🔹 Sub-article (1)
As per Part 7, Article 76, Sub-article (1):
• The President shall appoint the leader of a parliamentary party that 
  commands a majority in the House of Representatives as the Prime Minister, 
  and the Council of Ministers shall be constituted under his or her 
  chairpersonship.

🔹 Sub-article (2)
As per Part 7, Article 76, Sub-article (2):
• If no party has a clear majority, the President shall appoint as Prime 
  Minister a member of the House of Representatives who presents a ground 
  on which he or she can obtain a vote of confidence in the House of 
  Representatives.

🔹 Sub-article (4)
As per Part 7, Article 76, Sub-article (4):
• If a Prime Minister cannot be appointed under Sub-article (1) or (2), 
  the President shall appoint as the Prime Minister the parliamentary party 
  leader of the party which has the highest number of members in the House 
  of Representatives.
```

---

## 🎯 Use Cases

### 🧑‍🎓 **For Students**
- Learn constitutional law with structured explanations
- Get complete article breakdowns with all sub-articles
- Understand hierarchical relationships between provisions

### ⚖️ **For Lawyers & Legal Researchers**
- Quick retrieval of relevant constitutional provisions
- Complete article coverage (no missing sub-articles)
- Accurate Part/Article/Sub-article citations

### 🏛️ **For Government & NGOs**
- Build civic education platforms
- Provide constitution Q&A to citizens
- Policy analysis and research automation

### 🛠️ **For Developers**
- Backend for AI-powered legal tools
- Vector-search microservice for legal documents
- Domain-specific chatbot template

---

## 🛣️ Roadmap

- [x] Hierarchical chunking with metadata
- [x] Smart query expansion
- [x] Article completion for comprehensive answers
- [x] Structured response generation
- [ ] FastAPI backend with REST endpoints
- [ ] Web UI for interactive Q&A
- [ ] Support for multiple constitutions
- [ ] Multilingual support (Nepali, Hindi, etc.)
- [ ] Cross-article relationship graph
- [ ] Dockerized deployment
- [ ] Cloud-ready architecture

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report Issues**: Found a bug or incorrect retrieval? Open an issue
2. **Suggest Features**: Have ideas for improvements? Let us know
3. **Submit PRs**: Code contributions are appreciated
4. **Add Documents**: Help add more constitutions or legal documents

---

## 📜 License

MIT License - feel free to use this project for educational, research, or commercial purposes.

---

## 🙌 Acknowledgements

- **Constitution of Nepal** - Source document
- **OpenAI** - Embeddings and LLM
- **LangChain** - RAG framework
- **ChromaDB** - Vector database

Built to make constitutional knowledge **accessible, accurate, and AI-powered** 🚀
