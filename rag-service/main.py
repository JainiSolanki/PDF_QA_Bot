from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
import pdf2image
import pytesseract
from PIL import Image

# ---------------------------------------------------------------------------
# Utils: post-processing, prompt building, numeric answer disambiguation
# ---------------------------------------------------------------------------

# Post-processing: strip prompt echoes / context leakage from every LLM response.
from utils.postprocess import extract_final_answer, extract_final_summary, extract_comparison

# Minimal prompt builders (short prompts → less instruction echoing by the model).
from utils.prompt_templates import build_ask_prompt, build_summarize_prompt, build_compare_prompt

# Query expansion + answer-type chunk re-ranking + typed-answer validation.
from utils.query_utils import expand_query, rerank_docs, extract_typed_answer, get_answer_type_hint

load_dotenv()

app = FastAPI(
    title="PDF QA Bot API",
    description="PDF Question-Answering Bot (Session-based, No Auth)",
    version="2.1.0"
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
# SESSION STORAGE (REQUIRED: keep sessionId)
# ===============================
# Format: { session_id: { "vectorstores": [FAISS], "last_accessed": float } }
sessions = {}
SESSION_TIMEOUT = 3600  # 1 hour

# Embedding model (loaded once)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ===============================
# LOAD GENERATION MODEL ONCE
# ===============================
HF_GENERATION_MODEL = os.getenv("HF_GENERATION_MODEL", "google/flan-t5-small")

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



# ---------------------------------------------------------------------------
# MODEL GENERATION
# ---------------------------------------------------------------------------

def generate_response(prompt: str, max_new_tokens: int) -> str:
    """Run inference with the globally loaded model (no reload per request)."""
    global generation_tokenizer, generation_model, generation_is_encoder_decoder
    tokenizer = generation_tokenizer
    model = generation_model
    is_encoder_decoder = generation_is_encoder_decoder
    model_device = next(model.parameters()).device

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

    if is_encoder_decoder:
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return text.strip()

    input_len = encoded["input_ids"].shape[1]
    new_tokens = generated_ids[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


# ---------------------------------------------------------------------------
# ===============================
# REQUEST MODELS
# ===============================
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_ids: list = []
    doc_ids: list = []
    history: list = []


class SummarizeRequest(BaseModel):
    session_ids: list = []


class CompareRequest(BaseModel):
    session_ids: list = []


# ===============================
# UTILITIES
# ===============================
def cleanup_expired_sessions():
    current_time = time.time()
    expired = [
        sid for sid, data in sessions.items()
        if current_time - data["last_accessed"] > SESSION_TIMEOUT
    ]
    for sid in expired:
        del sessions[sid]


def generate_response(prompt: str, max_new_tokens: int = 200) -> str:
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


# ===============================
# HEALTH ENDPOINTS (kept from enhancement branch)
# ===============================
@app.get("/healthz")
def health_check():
    return {"status": "healthy"}


@app.get("/readyz")
def readiness_check():
    return {"status": "ready"}


# ===============================
# UPLOAD (NO AUTH, RETURNS session_id)
# ===============================
@app.post("/upload")
@limiter.limit("10/15 minutes")
async def upload_file(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported"}

    session_id = str(uuid4())
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    # SECURITY: Use only uuid4().hex to prevent path traversal from client filename
    file_path = os.path.join(upload_dir, f"{uuid4().hex}.pdf")
    upload_dir_resolved = os.path.abspath(upload_dir)
    file_path_resolved = os.path.abspath(file_path)
    
    # SECURITY: Validate that file_path is within upload_dir (prevent path traversal)
    if not file_path_resolved.startswith(upload_dir_resolved + os.sep):
        return {"error": "Upload failed: Invalid file path detected."}

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        try:
            from utils.layout_extractor import extract_layout_aware_text
            docs = extract_layout_aware_text(file_path)
            
            # Since layout extractor doesn't explicitly return original page count matching strictly OCR,
            # we infer page count from metadata.
            page_count = max([doc.metadata.get("page", 0) for doc in docs]) + 1 if docs else 0
        except Exception as layout_e:
            print(f"Layout extractor failed, falling back to PyPDFLoader: {layout_e}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            page_count = len(docs)
    
            # Check if each page has extractable text
            final_docs = []
            images = None
            
            for i, doc in enumerate(docs):
                if len(doc.page_content.strip()) < 50:
                    # Fallback to OCR for this specific page
                    if images is None:
                        print("Low text content detected on one or more pages. Falling back to OCR...")
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

        # Check if each page has extractable text
        final_docs = []
        images = None
        
        for i, doc in enumerate(docs):
            if len(doc.page_content.strip()) < 50:
                # Fallback to OCR for this specific page
                if images is None:
                    print("Low text content detected on one or more pages. Falling back to OCR...")
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
            return {"error": "Upload failed: No extractable text found in the document (OCR yielded nothing)."}

        vectorstore = FAISS.from_documents(chunks, embedding_model)

        sessions[session_id] = {
            "vectorstores": [vectorstore],
            "filename": file.filename,
            "last_accessed": time.time()
        }

        return {
            "message": "PDF uploaded and processed",
            "session_id": session_id,
            "page_count": page_count
        }

    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}
    
    finally:
        # FIX: Delete PDF file after processing to prevent disk space exhaustion (Issue #110)
        # This ensures the physical file is deleted even if OCR or embedding fails
        try:
            os.remove(file_path)
        except FileNotFoundError:
            # File already deleted or never created; nothing to clean up
            pass
        except OSError as delete_err:
            # Log other errors but don't crash
            print(f"[/upload] Warning: Failed to delete file: {str(delete_err)}")



# ===============================
# ASK (USES session_ids — matches fixed App.js)
# ===============================
@app.post("/ask")
@limiter.limit("60/15 minutes")
def ask_question(request: Request, data: AskRequest):
    cleanup_expired_sessions()

    # Get the first session_id from the list
    if not data.session_ids or len(data.session_ids) == 0:
        return {"answer": "No session provided. Please upload a PDF first."}

    session_id = data.session_ids[0]
    session_data = sessions.get(session_id)
    if not session_data:
        return {"answer": "Session expired or no PDF uploaded for this session."}

    session_data["last_accessed"] = time.time()

    vectorstores = get_session_docs(session_id, data.doc_ids)
    if not vectorstores:
        return {"answer": "No documents found for the selected session."}

    question = data.question
    history  = data.history

    # Build conversation context (last 5 turns max)
    conversation_context = ""
    for msg in history[-5:]:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        conversation_context += f"{role}: {content}\n"

    # ── Step 1: Query expansion — cast a wider net for numeric/typed answers ──
    # e.g. "What is the percentage?" → appends "percentage % score marks grade"
    expanded_query = expand_query(question)

    # Retrieve a larger candidate pool (k=8) so re-ranking has more to work with
    docs = merged_similarity_search(vectorstores, expanded_query, k=8)
    if not docs:
        return {"answer": "No relevant context found in the selected documents."}

    # ── Step 2: Re-rank chunks by answer-type relevance ───────────────────────
    # Promotes chunks whose content FORMAT matches what the question asks for
    # (e.g. chunk with "69%" ranked above chunk with "45/75" for a % question).
    docs = rerank_docs(docs, question, top_k=4)

    context = "\n\n".join([doc.page_content for doc in docs])

    # ── Step 3: Build minimal prompt (short prompt → less instruction echoing) ──
    # Note: NO format hint injected into the prompt.
    # flan-t5-base treats format hints literally and outputs just the symbol
    # (e.g. bare "%"). Numeric disambiguation is handled in Step 5 below.
    prompt = build_ask_prompt(
        context=context,
        question=question,
        conversation_context=conversation_context,
    )

    raw_answer   = generate_response(prompt, max_new_tokens=150)

    # ── Step 4: Post-process — strip all prompt echoes / context leakage ───────
    clean_answer = extract_final_answer(raw_answer)

    # ── Step 5: Typed-answer validation / context-extraction fallback ─────────
    # If the model returned garbage (e.g. bare "%", single char, empty string),
    # extract the correct value directly from the retrieved context using regex.
    # Example: question asks for "%" → LLM outputs "%" → we find "69%" in context.
    clean_answer = extract_typed_answer(clean_answer, question, context)

    return {"answer": clean_answer}


    # Build deduplicated, sorted citations
    seen = set()
    citations = []
    # Filter citations to only matched docs that survived reranking
    final_docs_with_meta = [item for item in docs_with_meta if item["doc"] in docs]
    for item in final_docs_with_meta:
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
# SUMMARIZE
# ===============================
@app.post("/summarize")
@limiter.limit("15/15 minutes")
def summarize_pdf(request: Request, data: SummarizeRequest):
    cleanup_expired_sessions()

    if not data.session_ids:
        return {"summary": "No session selected."}

    vectorstores = []
    for sid in data.session_ids:
        session = sessions.get(sid)
        if session:
            vectorstores.extend(session["vectorstores"])

    if not vectorstores:
        return {"summary": "No documents found."}

    docs = []
    for vs in vectorstores:
        docs.extend(vs.similarity_search("Summarize the document", k=6))

    context = "\n\n".join([d.page_content for d in docs])

    # Minimal summarization prompt (no bullet-rule echoing)
    prompt = build_summarize_prompt(context=context)

    raw_summary = generate_response(prompt, max_new_tokens=300)
    summary     = extract_final_summary(raw_summary)
    return {"summary": summary}


@app.post("/suggest-questions")
def suggest_questions():
    global vectorstore, qa_chain
    
    if not qa_chain:
        return {"suggestions": []}
    
    try:
        # Get representative chunks from different parts of document
        docs = vectorstore.similarity_search("main topics key concepts summary", k=4)
        context = "\n\n".join([doc.page_content[:500] for doc in docs])
        
        prompt = (
            "Based on this document excerpt, generate 4 specific, useful questions "
            "that a reader would want answered. Make them clear and concise.\n\n"
            f"Document content:\n{context}\n\n"
            "Generate exactly 4 questions (one per line, no numbering):"
        )
        
        response = generate_response(prompt, max_new_tokens=120)
        
        # Parse and clean questions
        questions = [
            q.strip().lstrip('0123456789.-) ') 
            for q in response.split('\n') 
            if q.strip() and len(q.strip()) > 10
        ][:4]
        
        # Return suggestions or fallback questions
        if questions:
            return {"suggestions": questions}
        else:
            return {"suggestions": [
                "What are the main topics covered?",
                "Can you summarize the key points?",
                "What are the most important findings?",
                "What conclusions does this document present?"
            ]}
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return {"suggestions": [
            "What are the main topics covered?",
            "Can you summarize the key points?",
            "What are the most important findings?",
            "What conclusions does this document present?"
        ]}
# ===============================
# COMPARE
# ===============================
@app.post("/compare")
@limiter.limit("10/15 minutes")
def compare_documents(request: Request, data: CompareRequest):
    cleanup_expired_sessions()

    if len(data.session_ids) < 2:
        return {"comparison": "Select at least 2 documents."}

    contexts = []
    for sid in data.session_ids:
        session = sessions.get(sid)
        if session:
            vs = session["vectorstores"][0]
            chunks = vs.similarity_search("main topics", k=4)
            text = "\n".join([c.page_content for c in chunks])
            contexts.append(text)

    # Retrieve top chunks from each document separately for fair comparison
    query = "summarize the main topic, purpose, and key details of this document"
    per_doc_contexts = []
    for i, vs in enumerate(vectorstores):
        chunks = vs.similarity_search(query, k=4)
        text   = "\n".join([c.page_content for c in chunks])
        per_doc_contexts.append(text)

    # Minimal comparison prompt (no numbered-rule echoing)
    prompt = build_compare_prompt(per_doc_contexts=per_doc_contexts)

    raw        = generate_response(prompt, max_new_tokens=400)
    comparison = extract_comparison(raw)
    return {"comparison": comparison}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)