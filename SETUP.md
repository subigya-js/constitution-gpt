# 🎉 Constitution GPT - Complete Setup Guide

This guide will help you set up and run the complete Constitution GPT system with both the FastAPI backend and Next.js frontend.

## 📋 Prerequisites

- **Python 3.8+** with virtual environment
- **Node.js 20+** with npm
- **OpenAI API Key**

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Install Python Dependencies

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 3. Create Vector Database

```bash
# This creates the ChromaDB database from the Constitution PDF
python rag/ingestion_pipeline.py
```

This will:
- Load the Constitution PDF (240 pages)
- Create 1,719 semantic chunks with metadata
- Generate embeddings using OpenAI
- Store in `db/chroma_db/`

### 4. Install Frontend Dependencies

```bash
cd web
npm install
cd ..
```

## 🎮 Running the Application

You need to run **both** the API server and the frontend. Open two terminal windows:

### Terminal 1: Start the API Server

```bash
# From project root
python -m uvicorn api.main:app --reload --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

### Terminal 2: Start the Frontend

```bash
# From project root
cd web
npm run dev
```

The frontend will be available at:
- **App**: http://localhost:3000

## ✅ Verify Everything Works

1. **Check API Health**:
   ```bash
   curl http://localhost:8000/health
   ```
   Should return: `{"status":"healthy","service":"Constitution GPT API"}`

2. **Open Frontend**:
   Navigate to http://localhost:3000

3. **Ask a Question**:
   - Click on a suggested question, or
   - Type your own question about the Constitution of Nepal
   - Wait for the AI response (2-4 seconds)

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User's Browser                          │
│                  http://localhost:3000                      │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Next.js Frontend (React 19)                  │  │
│  │  - Beautiful chat interface                          │  │
│  │  - Suggested questions                               │  │
│  │  - Message history                                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ HTTP POST /api/chat
                           │ { "question": "..." }
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend                            │
│               http://localhost:8000                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         API Endpoints                                │  │
│  │  - POST /api/chat                                    │  │
│  │  - GET /health                                       │  │
│  │  - GET /docs                                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ retrieve_and_answer(query)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   RAG Pipeline                              │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Query Expansion                                  │  │
│  │     - Generate variations                            │  │
│  │     - Add synonyms                                   │  │
│  │                                                       │  │
│  │  2. Vector Search (ChromaDB)                         │  │
│  │     - Retrieve relevant chunks                       │  │
│  │     - Fetch complete articles                        │  │
│  │                                                       │  │
│  │  3. Context Creation                                 │  │
│  │     - Group by article                               │  │
│  │     - Sort by hierarchy                              │  │
│  │                                                       │  │
│  │  4. LLM Generation (GPT-4o)                          │  │
│  │     - Structured response                            │  │
│  │     - Proper citations                               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
constitution-gpt/
├── api/
│   ├── main.py              # FastAPI server
│   └── README.md            # API documentation
├── rag/
│   ├── data/
│   │   └── Constitution_English.pdf
│   ├── ingestion_pipeline.py
│   ├── retrieval_pipeline.py
│   └── test_various_queries.py
├── web/
│   ├── app/
│   │   ├── components/
│   │   │   └── ChatInterface.tsx
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── package.json
│   └── README.md
├── db/
│   └── chroma_db/          # Vector database (auto-generated)
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
└── README.md              # Main documentation
```

## 🐛 Troubleshooting

### API Server Won't Start

**Error**: `OpenAI API key not found`
- **Solution**: Make sure `.env` file exists in project root with `OPENAI_API_KEY=...`

**Error**: `ModuleNotFoundError: No module named 'langchain_chroma'`
- **Solution**: Run `pip install -r requirements.txt`

**Error**: `ChromaDB not found`
- **Solution**: Run `python rag/ingestion_pipeline.py` to create the database

### Frontend Issues

**Error**: Hydration errors
- **Solution**: Already fixed! The page uses `'use client'` and dynamic imports

**Error**: Cannot connect to API
- **Solution**: Make sure API server is running at http://localhost:8000

**Error**: CORS errors
- **Solution**: API is configured for `localhost:3000`. If using different port, update `api/main.py`

### Port Conflicts

**API Port 8000 in use**:
```bash
python -m uvicorn api.main:app --reload --port 8001
```
Update frontend to use port 8001 in `ChatInterface.tsx`

**Frontend Port 3000 in use**:
```bash
PORT=3001 npm run dev
```
Update API CORS settings to include port 3001

## 🎯 Example Queries

Try these questions:
- "How is the Prime Minister elected in Nepal?"
- "What are the fundamental rights of citizens?"
- "What are the duties of citizens?"
- "How is the President elected?"
- "What is the structure of the Federal Parliament?"
- "What are the provisions for freedom of speech?"

## 📊 Performance Metrics

- **Average Response Time**: 2-4 seconds
- **Vector Database**: 1,719 chunks
- **Retrieval Accuracy**: ~90% for tested queries
- **Supported Queries**: Any question about Nepal's Constitution

## 🚀 Next Steps

1. **Add Authentication**: Implement user login
2. **Message Persistence**: Store chat history in database
3. **Export Functionality**: Export conversations as PDF
4. **Voice Input**: Add speech-to-text
5. **Multiple Documents**: Support other constitutions
6. **Deployment**: Deploy to production (Vercel + Railway/Render)

## 📝 License

MIT License

---

Built with ❤️ for Constitution GPT
