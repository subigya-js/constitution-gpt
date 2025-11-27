# 🏛️ Constitution GPT
### An Open-Source Constitutional Intelligence System Powered by RAG + LLMs

Constitution GPT is not just another RAG pipeline.
It is an **open-source intelligence system designed specifically for constitutional, legal, policy, and governance documents**, enabling precise retrieval, interpretation, and question-answering **grounded in authoritative texts**.

This project aims to help students, lawyers, policymakers, researchers, and developers build systems that require:
- Accurate referencing
- Context-aware summarization
- Traceable legal reasoning
- Question answering based on verified constitutional sources

---

## 🌟 Why Constitution GPT?

The main goal is to solve a real-world problem:

> **Legal and constitutional documents are long, complex, and interconnected.
Traditional search is too shallow.
LLMs alone hallucinate.
Constitution GPT fills this gap.**

It provides:
- Reliable answers grounded *only* in uploaded legal documents
- Retrieval across extremely large PDFs (full constitutions, amendments, legal codes)
- Explanations and breakdowns of legal concepts
- Transparent citations
- Extensibility for any domain (tax law, policies, HR manuals, contracts, etc.)

You can use Constitution GPT as:
- A study assistant for constitutional law
- A legal research tool
- A chatbot for citizens to learn about rights and duties
- A backend engine for civic education apps
- A document-analysis microservice in your own applications

---

## 🚀 Features

### 📄 **1. PDF → Knowledge Engine**
Upload any constitution or legal document:
- National Constitutions
- Amendments
- Acts & Regulations
- Policy drafts
- Academic legal papers

The system converts them into **structured, retrievable knowledge**.

### ✂️ **2. Smart Chunking (Not Just Character Splitting)**
Supports:
- Chunking Methods
- Rule-based legal breakpoints
- Section-article-auto-detection
- Hierarchical chunk structure

Designed for **legal text hierarchy**, not random chunk boundaries.

### 🔍 **3. Vector Retrieval Optimized for Law**
Uses OpenAI embeddings + vector database (Chroma by default) to retrieve:
- The most relevant articles
- Related clauses
- Cross-referenced sections
- Definitions and exceptions

### 🧠 **4. Constitutional Q&A Engine**
Example queries:
- “What are the fundamental rights outlined in Article 17?”
- “Explain the separation of powers in simple words.”
- “What duties do citizens have according to the constitution?”
- “Summaries of the constitutional amendments so far.”

Output is grounded, citation-backed, and easy to understand.

### ⚡ **5. FastAPI Backend for Developers**
Provides clean API endpoints so you can:
- Build apps
- Integrate into Go or Node backends
- Use it in mobile apps
- Connect it to your front-end (React/Next.js)

---

## 📁 Repository Structure

```
constitution-gpt/
│── backend/
│   ├── main.py              # FastAPI app
│   ├── ingest.py            # PDF loading
│   ├── chunking.py          # Semantic + rule-based chunkers
│   ├── embeddings.py        # Embedding generation
│   ├── vector.py            # Vector database interface
│   ├── retrieval.py         # Retrieval logic
│   ├── qa_pipeline.py       # Final answer generation
│   └── data/                # PDF files
│
│── frontend/                # (Optional) Web UI
│── README.md
│── requirements.txt
│── .env.example
```

---

## 🧩 How It Works (Conceptual Flow)

```
PDF → Extract Text → Smart Chunking → Embeddings → Vector DB
          ↑                               ↓
      User Query  ← Retrieval ← LLM reasoning ← Context
```

### In real usage:
- User asks:
  **“What are the powers of the Supreme Court?”**
- System retrieves Articles 126, 127, 128, related clauses
- LLM analyzes them
- Answer is grounded *only* on actual constitutional text
- Final response is clear and citation-backed

---

## ⚙️ Installation

### Clone & Setup

```bash
git clone https://github.com/subigya-js/constitution-gpt.git
cd constitution-gpt/backend
```

### Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Environment Variables

Create `.env`:

```
OPENAI_API_KEY=your_api_key
```

---

## ▶️ Running the API

```bash
uvicorn main:app --reload
```

---

## 🎯 Example Use Cases

### 🧑‍🎓 For Students
- Learn constitutional law with simplified explanations
- Ask “why” questions, not just definitions
- Revise articles with summaries

### ⚖️ For Lawyers
- Quick retrieval of relevant clauses
- Interpretation assistant (non-legal-advice)
- Cross-reference articles instantly

### 🏛️ For Government / NGOs
- Build civic education platforms
- Provide constitution Q&A to citizens
- Policy analysis automation

### 🛠️ For Developers
- Backend for AI-powered legal tools
- Vector-search microservice
- Domain-specific chatbot starter template

---

## 🛣️ Roadmap

- [ ] UI for uploading new constitutions
- [ ] Multilingual support (Nepali, Hindi, etc.)
- [ ] Context graph for cross-article interpretations
- [ ] Citations mode
- [ ] Dockerized deployment
- [ ] Option for Go backend
- [ ] Cloud-ready architecture

---

## 🤝 Contributing

PRs, issues, and feature suggestions are welcome!

---

## 📜 License

MIT License.

---

## 🙌 Acknowledgements

Built as part of the **Constitution GPT** initiative
to make constitutional knowledge accessible, accurate, and AI-powered.
