import os
import math
import time
from typing import List, Optional, Tuple

import pdfplumber
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_objectbox.vectorstores import ObjectBox
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from PIL import Image
import pytesseract

from utils import groq_llm, huggingface_instruct_embedding, get_reranker
from config import load_config

load_config()

app = FastAPI(title="DocChat API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    content: str
    source: Optional[str] = None
    page: Optional[int] = None
    relevance: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    response_time: float
    sources: List[SourceChunk] = []
    confidence: Optional[str] = None


VECTOR_STORE = None

SYSTEM_PROMPT = (
    "You are DocChat, a precise AI assistant. "
    "Answer the user's question using ONLY the numbered document excerpts provided in the conversation. "
    "Rules:\n"
    "1. Base your answer strictly on information in the excerpts. Never use outside knowledge.\n"
    "2. When possible, reference which excerpt supports your claim (e.g. \"According to Excerpt 2…\").\n"
    "3. Use exact numbers, dates, and names from the excerpts — do not approximate or guess.\n"
    "4. If the excerpts do not contain enough information, reply: "
    "\"The provided documents don't contain enough information to answer this question.\"\n"
    "5. If excerpts seem contradictory, point out the discrepancy.\n"
    "6. Be concise. Do not repeat the question or add filler."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",
     "Document excerpts:\n\n{context}\n\n---\n\nQuestion: {input}\n\nAnswer:"),
])


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def deduplicate_chunks(
    results: List[Tuple[Document, float]], threshold: float = 0.80
) -> List[Tuple[Document, float]]:
    if not results:
        return results

    unique: List[Tuple[Document, float]] = [results[0]]
    for doc, score in results[1:]:
        words_a = set(doc.page_content.lower().split())
        is_dup = False
        for existing_doc, _ in unique:
            words_b = set(existing_doc.page_content.lower().split())
            if not words_a or not words_b:
                continue
            overlap = len(words_a & words_b) / len(words_a | words_b)
            if overlap > threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append((doc, score))
    return unique


def process_pdf(uploaded_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path=uploaded_path)
    documents = loader.load()

    table_docs: List[Document] = []
    try:
        with pdfplumber.open(uploaded_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                tables = page.extract_tables() or []
                for t_idx, table in enumerate(tables):
                    if not table:
                        continue
                    rows = [" | ".join(cell or "" for cell in row) for row in table]
                    table_text = "\n".join(rows)
                    if not table_text.strip():
                        continue
                    table_docs.append(
                        Document(
                            page_content=table_text,
                            metadata={"page": page_index, "is_table": True},
                        )
                    )
    except Exception:
        pass

    return documents + table_docs


def process_image(uploaded_path: str, filename: str) -> List[Document]:
    image = Image.open(uploaded_path)
    text = pytesseract.image_to_string(image)
    return [Document(page_content=text, metadata={"source": filename})]


def process_text_file(uploaded_path: str, filename: str) -> List[Document]:
    with open(uploaded_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [Document(page_content=text, metadata={"source": filename})]


def build_vector_store(files: List[UploadFile]):
    global VECTOR_STORE

    db_root = os.path.join(os.path.dirname(__file__), "project", "objectbox")
    os.makedirs(db_root, exist_ok=True)
    db_directory = os.path.join(db_root, f"session_{int(time.time())}")
    os.makedirs(db_directory, exist_ok=True)

    embeddings = huggingface_instruct_embedding()
    docs: List[Document] = []

    for uploaded_file in files:
        if not uploaded_file.filename:
            continue

        suffix = os.path.splitext(uploaded_file.filename)[1].lower()
        temp_path = os.path.join(
            "/tmp", f"docchat_{int(time.time() * 1000)}_{uploaded_file.filename}"
        )

        with open(temp_path, "wb") as temp_file:
            temp_file.write(uploaded_file.file.read())

        try:
            if uploaded_file.content_type == "application/pdf" or suffix == ".pdf":
                documents = process_pdf(temp_path)
            elif uploaded_file.content_type in (
                "image/jpeg",
                "image/png",
            ) or suffix in (".jpg", ".jpeg", ".png"):
                documents = process_image(temp_path, uploaded_file.filename)
            elif suffix in (".txt", ".md"):
                documents = process_text_file(temp_path, uploaded_file.filename)
            else:
                continue

            for d in documents:
                meta = dict(d.metadata or {})
                meta["source"] = uploaded_file.filename
                if "page" not in meta:
                    meta["page"] = None
                d.metadata = meta

            docs.extend(documents)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if not docs:
        raise HTTPException(
            status_code=400, detail="No supported documents were uploaded."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    final_documents = text_splitter.split_documents(docs)

    VECTOR_STORE = ObjectBox.from_documents(
        final_documents,
        embeddings,
        embedding_dimensions=768,
        db_directory=db_directory,
    )

    return len(final_documents)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed")
async def embed_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    try:
        doc_count = build_vector_store(files)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error embedding documents: {e}"
        )

    return {
        "message": "Documents embedded successfully.",
        "document_chunks": doc_count,
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest):
    if VECTOR_STORE is None:
        raise HTTPException(
            status_code=400,
            detail="No vector store available. Please upload and embed documents first.",
        )

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        results: List[Tuple[Document, float]] = (
            VECTOR_STORE.similarity_search_with_score(payload.question, k=15)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during retrieval: {e}")

    if not results:
        return QueryResponse(
            answer="I couldn't find any relevant content in your documents.",
            response_time=0.0,
            sources=[],
            confidence="none",
        )

    results = deduplicate_chunks(results)

    reranker = get_reranker()
    pairs = [(payload.question, doc.page_content) for doc, _ in results]
    rerank_scores = reranker.predict(pairs).tolist()

    ranked = sorted(
        zip(results, rerank_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    top_n = min(5, len(ranked))
    top_items = ranked[:top_n]
    top_results = [(doc, orig) for (doc, orig), _ in top_items]
    top_rerank = [rs for _, rs in top_items]

    best = sigmoid(top_rerank[0])
    avg3 = sum(sigmoid(s) for s in top_rerank[: min(3, len(top_rerank))]) / min(
        3, len(top_rerank)
    )

    if best >= 0.85 or avg3 >= 0.75:
        confidence = "high"
    elif best >= 0.55 or avg3 >= 0.45:
        confidence = "medium"
    else:
        confidence = "low"

    context_parts: List[str] = []
    for i, (doc, _) in enumerate(top_results, start=1):
        meta = doc.metadata or {}
        src = meta.get("source", "Document")
        page = meta.get("page")
        label = f"[Excerpt {i} — {src}"
        if page is not None:
            label += f", Page {page + 1}"
        label += "]"
        context_parts.append(f"{label}\n{doc.page_content}")

    context_str = "\n\n---\n\n".join(context_parts)

    start = time.perf_counter()

    try:
        llm = groq_llm()
        messages = prompt.format_messages(context=context_str, input=payload.question)
        response = llm.invoke(messages)
        answer = response.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during generation: {e}")

    elapsed = time.perf_counter() - start

    sources: List[SourceChunk] = []
    for (doc, _), rs in top_items:
        metadata = doc.metadata or {}
        sources.append(
            SourceChunk(
                content=doc.page_content,
                source=metadata.get("source", ""),
                page=metadata.get("page"),
                relevance=round(sigmoid(rs), 3),
            )
        )

    if not answer:
        answer = "No answer could be generated from the current document context."

    return QueryResponse(
        answer=answer,
        response_time=elapsed,
        sources=sources,
        confidence=confidence,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
