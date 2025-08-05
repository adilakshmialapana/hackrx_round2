# HackRX 6.0 - RAG API Server

A high-performance RAG (Retrieval-Augmented Generation) system using Gemini 2.5 Flash for accurate document question answering.

## Features

- **Fast PDF Processing**: Uses PyMuPDF for efficient text extraction
- **Advanced RAG Pipeline**: FAISS vector store with Gemini embeddings
- **High-Speed LLM**: Gemini 2.5 Flash for fast and accurate responses
- **Secure API**: Bearer token authentication
- **Production Ready**: Optimized for deployment on Render

## API Endpoint

```
POST /api/v1/hackrx/run
```

### Request Format

```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the definition of 'Accident' in this policy?",
    "Are AYUSH treatments covered under this policy?"
  ]
}
```

### Headers

```
Authorization: Bearer f072e58e6d9a51de69f3f1d1a0e267f663a545d4c3b4edda40dba2e631f1ee73
Content-Type: application/json
```

## Environment Variables

Create a `.env` file with:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Deployment on Render

1. **Connect your repository** to Render
2. **Set environment variables**:
   - `GEMINI_API_KEY`: Your Gemini API key
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Local Development

```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Project Structure

```
├── main.py              # Main FastAPI application
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Performance

- **Response Time**: < 15 seconds for 10 questions
- **Accuracy**: High-quality answers using Gemini 2.5 Flash
- **Scalability**: Optimized for concurrent requests 