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
```

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
