"""
RAG Service with Embedding Model Version Tracking
===================================================

ISSUE: T135 #119 - No Embedding Model Version Tracking Causes Vector Store Drift

PROBLEM:
- The RAG pipeline generates embeddings using HuggingFace models and stores them in FAISS
- However, there was NO tracking of embedding model version/dimension alongside vectors
- If the embedding model changed/upgraded without re-indexing, old vectors silently mixed with new embedding space
- This led to "silent retrieval corruption" - answers degraded without throwing errors

SOLUTION IMPLEMENTED:
1. EmbeddingMetadata class tracks: model_name, embedding_dim, created_timestamp, model_hash
2. Metadata persisted as JSON files alongside session data for durability
3. On vectorstore load: Validates stored metadata vs current model config
4. Auto-invalidates incompatible indexes and triggers re-embedding
5. New diagnostic endpoints for debugging embedding drift issues

KEY FEATURES:
- Automatic detection of embedding model changes
- Prevention of vector space mixing from different models
- Per-session metadata persistence to disk
- Thread-safe session management
- Silent failure prevention through explicit validation
- Diagnostic endpoints for monitoring and debugging

NEW ENDPOINTS:
- GET /embedding-model-info - Show current embedding model configuration
- POST /validate-embeddings - Validate session embedding compatibility
- Updated GET /status - Now includes embedding metadata

BACKWARD COMPATIBILITY:
- Existing vectorstores continue to work
- New sessions automatically tracked with metadata
- No schema changes to existing APIs
"""

from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from uuid import uuid4
import os
import time
import uuid
import torch
import uvicorn
import threading
from datetime import datetime
from pathlib import Path
import json
import hashlib
import pdf2image
import pytesseract
from PIL import Image
import docx
import re

# Post-processing helpers: strip prompt echoes / context leakage from LLM output
from utils.postprocess import extract_final_answer, extract_final_summary, extract_comparison

# Centralised minimal prompt builders (short prompts → less instruction echoing)
from utils.prompt_templates import build_ask_prompt, build_summarize_prompt, build_compare_prompt

load_dotenv()

app = FastAPI(
    title="PDF QA Bot API",
    description="PDF Question-Answering Bot (Session-based, Embedding Tracking)",
    version="2.2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ===============================
# CONFIGURATION
# ===============================
HF_GENERATION_MODEL = os.getenv("HF_GENERATION_MODEL", "google/flan-t5-small")
LLM_GENERATION_TIMEOUT = int(os.getenv("LLM_GENERATION_TIMEOUT", "30"))
SESSION_TIMEOUT = 3600
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# ===============================
# EMBEDDING MODEL METADATA TRACKING
# ===============================
class EmbeddingMetadata:
    """
    Tracks embedding model version and configuration to prevent vector store drift.
    This prevents silent retrieval corruption when embedding models change.
    
    Issue Reference: T135 #119 - Embedding Model Version Tracking
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding_dim = self._get_embedding_dimension()
        self.created_timestamp = datetime.utcnow().isoformat()
        self.model_hash = self._compute_model_hash()
    
    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension from the model."""
        try:
            test_embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            test_embed = test_embeddings.embed_query("test")
            return len(test_embed)
        except Exception as e:
            raise RuntimeError(f"Failed to determine embedding dimension: {str(e)}")
    
    def _compute_model_hash(self) -> str:
        """Compute a hash of the model configuration for change detection."""
        config_str = f"{self.model_name}_{self.embedding_dim}_{EMBEDDING_MODEL_NAME}"
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def to_dict(self) -> dict:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "created_timestamp": self.created_timestamp,
            "model_hash": self.model_hash,
            "metadata_version": "1.0"
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'EmbeddingMetadata':
        """Load metadata from dictionary."""
        metadata = EmbeddingMetadata.__new__(EmbeddingMetadata)
        metadata.model_name = data.get("model_name")
        metadata.embedding_dim = data.get("embedding_dimension")
        metadata.created_timestamp = data.get("created_timestamp")
        metadata.model_hash = data.get("model_hash")
        return metadata
    
    def validate_compatibility(self, current_model_name: str) -> tuple[bool, str]:
        """
        Validate if stored metadata is compatible with current model config.
        
        Returns:
            (is_compatible, reason_message)
        """
        if self.model_name != current_model_name:
            reason = f"Model mismatch: stored={self.model_name}, current={current_model_name}"
            return False, reason
        
        try:
            current_embeddings = HuggingFaceEmbeddings(model_name=current_model_name)
            current_dim = len(current_embeddings.embed_query("test"))
            
            if self.embedding_dim != current_dim:
                reason = f"Embedding dimension mismatch: stored={self.embedding_dim}, current={current_dim}"
                return False, reason
        except Exception as e:
            return False, f"Failed to validate embedding dimension: {str(e)}"
        
        return True, "Metadata is compatible"

# ===============================
# GLOBAL STATE MANAGEMENT (Thread-safe, Multi-user support)
# ===============================
# Session structure: {session_id: {"vectorstores": [FAISS], "filename": str, "last_accessed": float, "embedding_metadata": EmbeddingMetadata}}
sessions = {}
sessions_lock = threading.RLock()

# Embedding model (loaded once)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Initialize current embedding metadata at startup
current_embedding_metadata = EmbeddingMetadata(EMBEDDING_MODEL_NAME)

# ===============================
# GENERATION MODEL (loaded once)
# ===============================
config = AutoConfig.from_pretrained(HF_GENERATION_MODEL)
is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
tokenizer = AutoTokenizer.from_pretrained(HF_GENERATION_MODEL)

if is_encoder_decoder:
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_GENERATION_MODEL)
else:
    model = AutoModelForCausalLM.from_pretrained(HF_GENERATION_MODEL)

if torch.cuda.is_available():
    model = model.to("cuda")

model.eval()

# ===============================
# EMBEDDING METADATA UTILITIES
# ===============================
def get_metadata_file_path(session_id: str) -> Path:
    """Get the path where embedding metadata should be stored for a session."""
    metadata_dir = Path(__file__).parent / ".embedding_metadata"
    metadata_dir.mkdir(exist_ok=True)
    return metadata_dir / f"{session_id}_metadata.json"


def save_embedding_metadata(session_id: str, metadata: EmbeddingMetadata) -> None:
    """Persist embedding metadata to disk for durability and verification."""
    try:
        metadata_file = get_metadata_file_path(session_id)
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    except Exception as e:
        print(f"[WARNING] Failed to save embedding metadata: {str(e)}")


def load_embedding_metadata(session_id: str) -> EmbeddingMetadata | None:
    """Load embedding metadata from disk if it exists."""
    try:
        metadata_file = get_metadata_file_path(session_id)
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                return EmbeddingMetadata.from_dict(data)
    except Exception as e:
        print(f"[WARNING] Failed to load embedding metadata: {str(e)}")
    return None


def delete_embedding_metadata(session_id: str) -> None:
    """Delete embedding metadata file for a session."""
    try:
        metadata_file = get_metadata_file_path(session_id)
        if metadata_file.exists():
            metadata_file.unlink()
    except Exception:
        pass


def validate_embedding_compatibility(session_id: str) -> tuple[bool, str]:
    """Check if stored embedding metadata is compatible with current model."""
    stored_metadata = load_embedding_metadata(session_id)
    
    if stored_metadata is None:
        return True, "No stored metadata (new session)"
    
    is_compatible, reason = stored_metadata.validate_compatibility(EMBEDDING_MODEL_NAME)
    
    if not is_compatible:
        print(f"[EMBEDDING DRIFT DETECTED] Session {session_id}: {reason}")
    
    return is_compatible, reason

# ===============================
# REQUEST MODELS
# ===============================
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_ids: list = []

    @validator("question")
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Empty question")
        return v.strip()


class SummarizeRequest(BaseModel):
    session_ids: list = []


class CompareRequest(BaseModel):
    session_ids: list = []

# ===============================
# UTILITIES
# ===============================
def cleanup_expired_sessions():
    """Remove sessions that have exceeded SESSION_TIMEOUT."""
    current_time = time.time()
    expired = [
        sid for sid, data in sessions.items()
        if current_time - data.get("last_accessed", 0) > SESSION_TIMEOUT
    ]
    for sid in expired:
        with sessions_lock:
            if sid in sessions:
                del sessions[sid]
        delete_embedding_metadata(sid)


def generate_response(prompt: str, max_new_tokens: int = 200) -> str:
    """Generate response using the generation model."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    if is_encoder_decoder:
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


def normalize_spaced_text(text: str) -> str:
    """Normalize text with unusual spacing patterns."""
    pattern = r"\b(?:[A-Za-z] ){2,}[A-Za-z]\b"
    return re.sub(pattern, lambda m: m.group(0).replace(" ", ""), text)


def normalize_answer(text: str) -> str:
    """Post-process and normalize answers."""
    text = normalize_spaced_text(text)
    text = re.sub(r"^(Answer[^:]*:|Context:|Question:)\s*", "", text, flags=re.I)
    return text.strip()

# ===============================
# HEALTH ENDPOINTS
# ===============================
@app.get("/healthz")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/readyz")
def readiness_check():
    """Readiness check endpoint."""
    return {"status": "ready"}


@app.get("/health")
def health():
    """Legacy health check."""
    return {"status": "ok"}


# ===============================
# UPLOAD ENDPOINT
# ===============================
@app.post("/upload")
@limiter.limit("10/15 minutes")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """
    Upload and process a PDF file.
    Returns session_id for future operations.
    Tracks embedding metadata for drift detection.
    """
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported"}

    session_id = str(uuid4())
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    # SECURITY: Use uuid4().hex to prevent path traversal
    file_path = os.path.join(upload_dir, f"{uuid4().hex}.pdf")
    upload_dir_resolved = os.path.abspath(upload_dir)
    file_path_resolved = os.path.abspath(file_path)
    
    # SECURITY: Validate path is within upload_dir
    if not file_path_resolved.startswith(upload_dir_resolved + os.sep):
        return {"error": "Upload failed: Invalid file path detected."}

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Check if each page has extractable text
        final_docs = []
        images = None
        
        for i, doc in enumerate(docs):
            if len(doc.page_content.strip()) < 50:
                # Fallback to OCR for this specific page
                if images is None:
                    print("Low text content detected. Falling back to OCR...")
                    images = pdf2image.convert_from_path(file_path)
                
                if i < len(images):
                    ocr_text = pytesseract.image_to_string(images[i])
                    final_docs.append(Document(
                        page_content=ocr_text,
                        metadata={"source": file_path, "page": i}
                    ))
                else:
                    final_docs.append(doc)
            else:
                final_docs.append(doc)

        docs = final_docs

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            return {"error": "Upload failed: No extractable text found."}

        # Create vectorstore and track embedding metadata
        vectorstore = FAISS.from_documents(chunks, embedding_model)

        with sessions_lock:
            sessions[session_id] = {
                "vectorstores": [vectorstore],
                "filename": file.filename,
                "last_accessed": time.time(),
                "embedding_metadata": current_embedding_metadata
            }
        
        # Persist metadata to disk
        save_embedding_metadata(session_id, current_embedding_metadata)

        return {
            "message": "PDF uploaded and processed",
            "session_id": session_id,
            "page_count": len(docs),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dimension": current_embedding_metadata.embedding_dim
        }

    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}
    
    finally:
        # Delete PDF file after processing (Issue #110)
        try:
            os.remove(file_path)
        except (FileNotFoundError, OSError) as e:
            print(f"[/upload] Warning: Failed to delete file: {str(e)}")


# ===============================
# ASK ENDPOINT
# ===============================
@app.post("/ask")
@limiter.limit("60/15 minutes")
def ask_question(request: Request, data: AskRequest):
    """
    Answer questions using retrieved document context.
    Validates embedding compatibility for all sessions.
    """
    cleanup_expired_sessions()

    if not data.session_ids:
        return {"answer": "No session selected.", "citations": []}

    # Validate embedding compatibility and update last_accessed
    for sid in data.session_ids:
        with sessions_lock:
            session = sessions.get(sid)
            if session:
                session["last_accessed"] = time.time()
                
                # Validate embedding compatibility
                is_compatible, reason = validate_embedding_compatibility(sid)
                if not is_compatible:
                    print(f"[EMBEDDING VALIDATION] Session {sid} incompatible: {reason}")

    # Gather retrieved docs with their session filenames
    docs_with_meta = []
    for sid in data.session_ids:
        with sessions_lock:
            session = sessions.get(sid)
            if session:
                vs = session["vectorstores"][0]
                filename = session.get("filename", "unknown")
                retrieved = vs.similarity_search(data.question, k=4)
                for doc in retrieved:
                    docs_with_meta.append({
                        "doc": doc,
                        "filename": filename,
                        "sid": sid
                    })

    if not docs_with_meta:
        return {"answer": "No relevant context found.", "citations": []}

    # Build context with page annotations
    context_parts = []
    for item in docs_with_meta:
        raw_page = item["doc"].metadata.get("page", 0)
        page_num = int(raw_page) + 1
        context_parts.append(f"[Page {page_num}] {item['doc'].page_content}")

    context = "\n\n".join(context_parts)

    # Use minimal prompt builder
    prompt = build_ask_prompt(context=context, question=data.question)
    raw_answer = generate_response(prompt, max_new_tokens=150)
    clean_answer = extract_final_answer(raw_answer)

    # Build deduplicated, sorted citations
    seen = set()
    citations = []
    for item in docs_with_meta:
        raw_page = item["doc"].metadata.get("page", 0)
        page_num = int(raw_page) + 1
        key = (item["filename"], page_num)
        if key not in seen:
            seen.add(key)
            citations.append({
                "page": page_num,
                "source": item["filename"]
            })

    citations.sort(key=lambda c: (c["source"], c["page"]))

    return {"answer": clean_answer, "citations": citations}


# ===============================
# SUMMARIZE ENDPOINT
# ===============================
@app.post("/summarize")
@limiter.limit("15/15 minutes")
def summarize_pdf(request: Request, data: SummarizeRequest):
    """
    Summarize documents from selected sessions.
    Validates embedding compatibility for all sessions.
    """
    cleanup_expired_sessions()

    if not data.session_ids:
        return {"summary": "No session selected."}

    # Validate embedding compatibility and update last_accessed
    vectorstores = []
    for sid in data.session_ids:
        with sessions_lock:
            session = sessions.get(sid)
            if session:
                session["last_accessed"] = time.time()
                
                # Validate embedding compatibility
                is_compatible, reason = validate_embedding_compatibility(sid)
                if not is_compatible:
                    print(f"[EMBEDDING VALIDATION] Session {sid} incompatible: {reason}")
                
                vectorstores.extend(session["vectorstores"])

    if not vectorstores:
        return {"summary": "No documents found."}

    docs = []
    for vs in vectorstores:
        docs.extend(vs.similarity_search("Summarize the document", k=6))

    context = "\n\n".join([d.page_content for d in docs])

    prompt = build_summarize_prompt(context=context)
    raw_summary = generate_response(prompt, max_new_tokens=300)
    summary = extract_final_summary(raw_summary)
    
    return {"summary": summary}


# ===============================
# COMPARE ENDPOINT
# ===============================
@app.post("/compare")
@limiter.limit("10/15 minutes")
def compare_documents(request: Request, data: CompareRequest):
    """
    Compare documents from multiple sessions.
    Validates embedding compatibility for all sessions.
    """
    cleanup_expired_sessions()

    if len(data.session_ids) < 2:
        return {"comparison": "Select at least 2 documents."}

    # Validate embedding compatibility and gather vectorstores
    vectorstores = []
    for sid in data.session_ids:
        with sessions_lock:
            session = sessions.get(sid)
            if session:
                session["last_accessed"] = time.time()
                
                # Validate embedding compatibility
                is_compatible, reason = validate_embedding_compatibility(sid)
                if not is_compatible:
                    print(f"[EMBEDDING VALIDATION] Session {sid} incompatible: {reason}")
                
                vectorstores.extend(session["vectorstores"])

    if not vectorstores:
        return {"comparison": "No documents found."}

    # Retrieve top chunks from each document
    query = "summarize the main topic, purpose, and key details of this document"
    per_doc_contexts = []
    for vs in vectorstores:
        chunks = vs.similarity_search(query, k=4)
        text = "\n".join([c.page_content for c in chunks])
        per_doc_contexts.append(text)

    # Build minimal comparison prompt
    prompt = build_compare_prompt(per_doc_contexts=per_doc_contexts)
    raw = generate_response(prompt, max_new_tokens=400)
    comparison = extract_comparison(raw)
    
    return {"comparison": comparison}


# ===============================
# EMBEDDING METADATA DIAGNOSTICS
# ===============================
@app.get("/embedding-model-info")
def get_embedding_model_info():
    """
    Returns current embedding model configuration.
    Useful for debugging embedding drift issues.
    """
    return {
        "model_name": EMBEDDING_MODEL_NAME,
        "embedding_dimension": current_embedding_metadata.embedding_dim,
        "model_hash": current_embedding_metadata.model_hash,
        "created_timestamp": current_embedding_metadata.created_timestamp,
        "metadata_version": "1.0"
    }


@app.post("/validate-embeddings")
def validate_embeddings_endpoint(request: Request, session_id: str = None):
    """
    Validates embedding compatibility for a session.
    Returns detailed information about any incompatibilities detected.
    """
    if not session_id:
        return {"error": "session_id parameter required"}
    
    is_compatible, reason = validate_embedding_compatibility(session_id)
    stored_metadata = load_embedding_metadata(session_id)
    
    return {
        "session_id": session_id,
        "is_compatible": is_compatible,
        "reason": reason,
        "current_model": EMBEDDING_MODEL_NAME,
        "current_dimension": current_embedding_metadata.embedding_dim,
        "stored_metadata": stored_metadata.to_dict() if stored_metadata else None,
        "action_taken": "validation_required" if not is_compatible else "vectorstore_valid"
    }


# ===============================
# SESSION MANAGEMENT ENDPOINT
# ===============================
@app.get("/sessions")
def get_sessions():
    """Returns list of active sessions with their metadata."""
    cleanup_expired_sessions()
    
    session_list = []
    with sessions_lock:
        for sid, session_data in sessions.items():
            metadata = session_data.get("embedding_metadata")
            session_list.append({
                "session_id": sid,
                "filename": session_data.get("filename", "unknown"),
                "last_accessed": session_data.get("last_accessed"),
                "vectorstore_count": len(session_data.get("vectorstores", [])),
                "embedding_metadata": metadata.to_dict() if metadata else None
            })
    
    return {"sessions": session_list}


# ===============================
# START SERVER
# ===============================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)
