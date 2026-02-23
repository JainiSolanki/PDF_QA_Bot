from fastapi import FastAPI
from fastapi import Request
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from groq import Groq
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
import os
import re
import uvicorn
import time

load_dotenv()

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# ---------------------------------------------------------------------------
# GROQ CLIENT SETUP
# ---------------------------------------------------------------------------

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Please add it to your .env file.\n"
        "Get a free key at https://console.groq.com"
    )

groq_client = Groq(api_key=GROQ_API_KEY)

# Session management for isolated vectorstores
sessions = {}
SESSION_TIMEOUT = 3600  # 1 hour

# Load local embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# TEXT NORMALIZATION UTILITIES
# ---------------------------------------------------------------------------

def normalize_spaced_text(text: str) -> str:
    """
    Fixes character-level spaced text produced by PyPDFLoader on certain
    vector-based PDFs (e.g. NPTEL / IBM Coursera certificates).
    """
    def fix_spaced_word(match):
        return match.group(0).replace(" ", "")

    # Pattern: 3+ single alpha chars each separated by exactly one space
    pattern = r'\b(?:[A-Za-z] ){2,}[A-Za-z]\b'
    return re.sub(pattern, fix_spaced_word, text)


def normalize_answer(text: str) -> str:
    """
    Post-processes the LLM-generated answer.
    """
    text = normalize_spaced_text(text)
    # Strip only clear prompt-echo artefacts at the very start
    text = re.sub(r'^(Final Answer:|Context:|Question:)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# CCC PROMPT TEMPLATE (Connect – Content – Continue)
# ---------------------------------------------------------------------------

_CCC_SYSTEM = (
    "You are an intelligent PDF Question Answering Assistant.\n"
    "Your task is to answer the user's question strictly using the provided context "
    "retrieved from uploaded documents.\n\n"
    "Follow the CCC communication structure in a SINGLE response:\n\n"
    "1. Context Connection — Start with a short professional greeting (e.g. \"Hello,\").\n"
    "2. Content Explanation — Rewrite the retrieved context in clear, meaningful, "
    "grammatically correct sentences. Do NOT copy text directly.\n"
    "3. Core Answer — Present the main explanation in a structured, readable format.\n"
    "4. Call to Action — End with a relevant follow-up question to encourage further exploration.\n\n"
    "Rules:\n"
    "- Use ONLY the provided context. Do NOT add external knowledge.\n"
    "- If the answer is not in the context, say: "
    "\"The uploaded document does not contain sufficient information to answer this question.\"\n"
    "- Maintain a clear, professional tone.\n"
    "- Produce one continuous response — no bullet headers like '1.' or '2.'."
)

_CCC_USER_TEMPLATE = """\
Context from the uploaded PDF:
{context}

Question:
{question}

Final Answer:"""

CCC_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=_CCC_USER_TEMPLATE,
)


# ---------------------------------------------------------------------------
# GROQ GENERATION
# ---------------------------------------------------------------------------

def generate_response(system_prompt: str, user_prompt: str, max_tokens: int = 600) -> str:
    """
    Calls the Groq chat-completions API with a system + user message pair.
    """
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,     # low temp = factual, grounded answers
    )
    return completion.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------------------------

class PDFPath(BaseModel):
    filePath: str
    session_id: str

class AskRequest(BaseModel):
    question: str
    session_id: str
    history: list = []

class SummarizeRequest(BaseModel):
    session_id: str
    pdf: str | None = None


def cleanup_expired_sessions():
    """Removes sessions that haven't been accessed for over SESSION_TIMEOUT seconds."""
    current_time = time.time()
    expired = [sid for sid, data in sessions.items() if current_time - data["last_accessed"] > SESSION_TIMEOUT]
    for sid in expired:
        del sessions[sid]


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.post("/process-pdf")
@limiter.limit("15/15 minutes")
def process_pdf(request: Request, data: PDFPath):
    cleanup_expired_sessions()

    loader   = PyPDFLoader(data.filePath)
    raw_docs = loader.load()

    # Layer 1: normalize at ingestion so embeddings are on real words
    cleaned_docs = []
    for doc in raw_docs:
        cleaned_content = normalize_spaced_text(doc.page_content)
        cleaned_docs.append(Document(page_content=cleaned_content, metadata=doc.metadata))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks   = splitter.split_documents(cleaned_docs)
    if not chunks:
        return {"error": "No text chunks generated from the PDF. Please check your file."}

    # Store a FRESH vectorstore for this session (overwrites any previous one)
    sessions[data.session_id] = {
        "vectorstore": FAISS.from_documents(chunks, embedding_model),
        "last_accessed": time.time()
    }

    return {"message": "PDF processed successfully"}


@app.post("/ask")
@limiter.limit("60/15 minutes")
def ask_question(request: Request, data: AskRequest):
    cleanup_expired_sessions()

    session_data = sessions.get(data.session_id)
    if not session_data:
        return {"answer": "Session expired or no PDF uploaded for this session!"}

    session_data["last_accessed"] = time.time()
    vectorstore = session_data["vectorstore"]  # Retrieve session-specific store

    question = data.question
    history  = data.history

    # Build conversation context from last 5 turns
    conversation_context = ""
    for msg in history[-5:]:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        conversation_context += f"{role}: {content}\n"

    # Retrieve relevant chunks from the SESSION vectorstore
    docs = vectorstore.similarity_search(question, k=4)
    if not docs:
        return {"answer": "No relevant context found in the uploaded document."}

    # Context is already clean (normalized at ingestion)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Merge conversation history into the question slot
    question_with_history = question
    if conversation_context.strip():
        question_with_history = (
            f"Conversation so far:\n{conversation_context.strip()}\n\n"
            f"Current Question: {question}"
        )

    # Build user turn from CCC template
    user_prompt = CCC_PROMPT.format(
        context=context,
        question=question_with_history,
    )

    raw_answer = generate_response(
        system_prompt=_CCC_SYSTEM,
        user_prompt=user_prompt,
        max_tokens=600,
    )

    # Post-process the answer itself
    answer = normalize_answer(raw_answer)
    return {"answer": answer}


@app.post("/summarize")
@limiter.limit("15/15 minutes")
def summarize_pdf(request: Request, data: SummarizeRequest):
    cleanup_expired_sessions()

    session_data = sessions.get(data.session_id)
    if not session_data:
        return {"summary": "Session expired or no PDF uploaded for this session!"}

    session_data["last_accessed"] = time.time()
    vectorstore = session_data["vectorstore"]  # Retrieve session-specific store

    # Retrieve chunks from the SESSION vectorstore
    docs = vectorstore.similarity_search("Give a concise summary of the document.", k=6)
    if not docs:
        return {"summary": "No document context available to summarize."}

    context = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = (
        "You are a document summarization assistant.\n"
        "Rules:\n"
        "1. Summarize in 6-8 concise bullet points.\n"
        "2. Clearly state: who received the certificate/document, what it is for, "
        "which organization issued it, who authorized it, and the date.\n"
        "3. Use proper Title Case for names. Return clean, readable text.\n"
        "4. Use ONLY the information in the provided context."
    )

    user_prompt = f"Context:\n{context}\n\nSummary (bullet points):"

    raw_summary = generate_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=512
    )
    summary = normalize_answer(raw_summary)
    return {"summary": summary}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
