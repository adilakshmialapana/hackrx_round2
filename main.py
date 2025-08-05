import os
import io
import json
import requests
import asyncio
import traceback
from typing import List
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

# RAG-specific imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
import fitz  # PyMuPDF is imported as fitz

# Load environment variables
load_dotenv()

# Set API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment variables")




# --- FastAPI App Instance ---
app = FastAPI(
    title="HackRX 6.0 - The Final Winning RAG Solution",
    description="A streamlined approach using Gemini 2.5 Flash for guaranteed performance.",
    version="1.0.0"
)

# Security scheme
security = HTTPBearer()

# --- Pydantic Data Models ---
class DocumentInput(BaseModel):
    documents: HttpUrl
    questions: List[str]

class Answers(BaseModel):
    answers: List[str]

# --- RAG Components ---
vector_store = None
llm = None
retriever = None

# --- Initialize RAG System ---
def initialize_rag_system(document_url: str):
    global vector_store, llm, retriever
    
    try:
        response = requests.get(str(document_url))
        response.raise_for_status()
        
        pdf_content = io.BytesIO(response.content)
        
        documents = []
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Using text extraction for better completeness
            text = page.get_text("text")
            if text.strip():
                documents.append(Document(page_content=text, metadata={"page": page_num + 1}))
        
        if not documents:
            raise ValueError("No text could be extracted from the PDF.")
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=500,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GEMINI_API_KEY,
            task_type="retrieval_document"
        )
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.2,
            max_output_tokens=1024,
            convert_system_message_to_human=True
        )
        
        # Increase top-k to 15 for richer context retrieval
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15}
        )
        
        print(f"RAG system initialized with document from: {document_url}")

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize RAG system: {str(e)}"
        )

# --- Generate Answers with Chunk Logging ---
async def get_answer_with_gemini_flash(question: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        try:
            docs = retriever.get_relevant_documents(question)
            # Log retrieved document pages and short snippet for debugging
            print(f"Question: {question}")
            for idx, doc in enumerate(docs):
                snippet = doc.page_content[:300].replace("\n", " ")  # first 300 chars snippet
                print(f"Retrieved chunk {idx+1} (page {doc.metadata.get('page', 'N/A')}): {snippet}...")
            
            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = f"""You are an expert system designed to provide accurate and factual answers based ONLY on the provided context.

IMPORTANT RULES:
1. Provide a direct and complete answer based ONLY on the provided context.
2. If the context contains insufficient or partial information, provide the best possible answer based on what is available.
3. If the context has no relevant information, answer: "Information not available in the document."
4. Do NOT use lists, bullet points, or formatting. Provide a clear paragraph or sentence.

Context:
{context}

Question: {question}

Answer:"""
            
            response = await llm.ainvoke(prompt)
            processed_response = (
                response.content.strip()
                .replace('*', '')
                .replace('\n', ' ')
                .replace('#', '')
                .strip()
            )
            return processed_response
        except Exception as e:
            return f"Error processing question with Gemini Flash: {str(e)}"

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=Answers, status_code=status.HTTP_200_OK)
async def run_submissions(data: DocumentInput, credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "f072e58e6d9a51de69f3f1d1a0e267f663a545d4c3b4edda40dba2e631f1ee73":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token."
        )
    
    try:
        if data.documents:
            initialize_rag_system(data.documents)
        
        if not retriever:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="RAG system not initialized. A document URL is required."
            )

        semaphore = asyncio.Semaphore(5)
        
        tasks = [get_answer_with_gemini_flash(question, semaphore) for question in data.questions]
        all_answers = await asyncio.gather(*tasks)
        
        return Answers(answers=all_answers)
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in processing: {e}")
        print(f"Full traceback: {error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in processing: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
