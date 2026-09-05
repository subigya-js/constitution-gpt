# 🚀 Constitution GPT API

FastAPI backend server for Constitution GPT that provides RESTful API endpoints for querying the Constitution of Nepal using RAG (Retrieval-Augmented Generation).

## ✨ Features

- **RESTful API**: Clean and well-documented API endpoints
- **RAG Integration**: Connects to the existing retrieval pipeline
- **CORS Support**: Configured for Next.js frontend
- **Interactive Docs**: Auto-generated Swagger UI documentation
- **Health Checks**: Monitor API status
- **Error Handling**: Comprehensive error responses

## 📋 Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- OpenAI API key
- ChromaDB vector database (created by running `rag/ingestion_pipeline.py`)

## 🚀 Installation

### 1. Install Dependencies

Make sure you're in the project root directory and have activated your virtual environment:

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install FastAPI and Uvicorn (if not already installed)
pip install fastapi uvicorn
```

### 2. Verify Environment Variables

Ensure your `.env` file in the project root contains:

```env
OPENAI_API_KEY=your_openai_api_key_here
FRONTEND_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

Use a comma-separated list of exact origins. On Render, replace the local values
with the Vercel production URL and any custom domain; omit URL paths and trailing
slashes.

### 3. Ensure Vector Database Exists

The API requires the ChromaDB vector database. If you haven't created it yet:

```bash
python rag/ingestion_pipeline.py
```

This will create the `db/chroma_db/` directory with the indexed constitution.

## 🎮 Running the API

### Development Mode

From the project root directory:

```bash
# Method 1: Using uvicorn directly
uvicorn api.main:app --reload --port 8000

# Method 2: Using Python
python -m uvicorn api.main:app --reload --port 8000

# Method 3: Running the script directly
python api/main.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Production Mode

For production deployment:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 --proxy-headers
```

## 📖 API Endpoints

### GET `/`
Root endpoint with API information.

**Response:**
```json
{
  "message": "Welcome to Constitution GPT API",
  "version": "1.0.0",
  "endpoints": {
    "/": "API information",
    "/health/live": "Liveness check",
    "/health/ready": "Dependency readiness check",
    "/api/chat": "Query the Constitution (POST)",
    "/docs": "Interactive API documentation"
  }
}
```

### GET `/health/live`
Process liveness endpoint. It does not call external dependencies.

**Response:**
```json
{
  "status": "alive",
  "service": "Constitution GPT API"
}
```

### GET `/health/ready`
Checks Chroma. It returns HTTP 503 when Chroma is unavailable. The legacy
`/health` path has the same readiness behavior.

### POST `/api/chat`
Query the Constitution of Nepal.

**Request Body:**
```json
{
  "question": "How is the Prime Minister elected in Nepal?"
}
```

**Response:**
```json
{
  "question": "How is the Prime Minister elected in Nepal?",
  "answer": "📘 Part 7 – Federal Executive\nArticle 76 – Constitution of Council of Ministers\n\n🔹 Sub-article (1)\nAs per Part 7, Article 76, Sub-article (1):\n• The President shall appoint the leader of a parliamentary party that commands a majority in the House of Representatives as the Prime Minister..."
}
```

**Error Response (400):**
```json
{
  "detail": "Question cannot be empty"
}
```

**Error Response (500):**
```json
{
  "detail": "Error processing query: [error message]"
}
```

## 🧪 Testing the API

### Using cURL

```bash
# Health check
curl http://localhost:8000/health/ready

# Query the constitution
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the fundamental rights of citizens?"}'
```

### Using Python Requests

```python
import requests

# Query the API
response = requests.post(
    "http://localhost:8000/api/chat",
    json={"question": "How is the President elected?"}
)

print(response.json()["answer"])
```

### Using the Interactive Docs

1. Open http://localhost:8000/docs in your browser
2. Click on the `/api/chat` endpoint
3. Click "Try it out"
4. Enter your question in the request body
5. Click "Execute"

## 🔌 Connecting to the Frontend

Update the `ChatInterface.tsx` component to use the API:

```typescript
const handleSend = async (question?: string) => {
  const messageContent = question || input.trim();
  if (!messageContent || isLoading) return;

  const userMessage: Message = {
    id: Date.now().toString(),
    role: 'user',
    content: messageContent,
    timestamp: new Date(),
  };

  setMessages(prev => [...prev, userMessage]);
  setInput('');
  setIsLoading(true);

  try {
    const response = await fetch('http://localhost:8000/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question: messageContent }),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch response');
    }

    const data = await response.json();

    const aiMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: data.answer,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, aiMessage]);
  } catch (error) {
    console.error('Error fetching response:', error);
    const errorMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: 'Sorry, I encountered an error. Please try again.',
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, errorMessage]);
  } finally {
    setIsLoading(false);
  }
};
```

## 🐛 Troubleshooting

### Port Already in Use

If port 8000 is already in use:
```bash
uvicorn api.main:app --reload --port 8001
```

Don't forget to update the frontend to use the new port!

### CORS Errors

If you get CORS errors from the frontend:
1. Check that the frontend's exact origin is present in `FRONTEND_ORIGINS`
2. Restart the API server after making changes

### Module Import Errors

If you get import errors:
```bash
# Make sure you're running from the project root
cd /path/to/constitution-gpt
python -m uvicorn api.main:app --reload
```

### ChromaDB Not Found

If you get database errors:
```bash
# Create the vector database first
python rag/ingestion_pipeline.py
```

## 📊 API Performance

- **Average Response Time**: 2-4 seconds (depends on query complexity)
- **Concurrent Requests**: Supports multiple simultaneous requests
- **Capacity Protection**: Concurrent RAG execution is bounded per API process

## 🔒 Security Considerations

For production deployment:
1. **Add Authentication**: Implement API key or JWT authentication
2. **Abuse Prevention**: Add gateway rate limiting before opening anonymous access at scale
3. **HTTPS**: Use HTTPS in production
4. **Environment Variables**: Never commit `.env` file
5. **CORS**: Set `FRONTEND_ORIGINS` to only your production frontend domains

## 📝 License

MIT License - Same as the parent Constitution GPT project

---

Built with ❤️ for Constitution GPT
