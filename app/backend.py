import os
import time
import uuid
from typing import List, Optional, Tuple

import pdfplumber
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_objectbox.vectorstores import ObjectBox
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from PIL import Image
import pytesseract

from utils import groq_llm, huggingface_instruct_embedding
from config import load_config


# Ensure environment variables are loaded before anything else
load_config()


app = FastAPI(title="DocChat API", version="1.0.0")

# Allow common frontend origins; adjust as needed in deployment
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


class QueryResponse(BaseModel):
    answer: str
    response_time: float
    sources: List[SourceChunk] = []


# Global in-memory handle to the current vector store
VECTOR_STORE = None


prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant that answers questions strictly using the provided context.

    Guidelines:
    - Only use information that appears in the <context>.
    - If the answer is not clearly supported by the context, say
      "I don't know based on the provided documents."
    - Prefer short, direct answers over long speculation.

    <context>
    {context}
    </context>

    Question: {input}
    Answer:
    """
)


def process_pdf(uploaded_path: str) -> List[Document]:
    """
    Load a PDF into page-level documents, plus additional table documents.
    Tables are converted into a simple markdown-like representation so they
    can participate in text-based retrieval and Q&A.
    """
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
                    # Convert table to a markdown-style string
                    rows = [" | ".join(cell or "" for cell in row) for row in table]
                    table_text = "\n".join(rows)
                    if not table_text.strip():
                        continue
                    meta = {
                        "source": os.path.basename(uploaded_path),
                        "source_name": os.path.basename(uploaded_path),
                        "page": page_index,
                        "table_index": t_idx,
                        "is_table": True,
                    }
                    table_docs.append(Document(page_content=table_text, metadata=meta))
    except Exception:
        # If table extraction fails, just fall back to text-only pages.
        pass

    base_docs = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in documents]
    return base_docs + table_docs


def process_image(uploaded_path: str, filename: str) -> List[Document]:
    image = Image.open(uploaded_path)
    text = pytesseract.image_to_string(image)
    return [Document(page_content=text, metadata={"source": filename})]


def build_vector_store(files: List[UploadFile]):
    """
    Build an ObjectBox vector store from the uploaded files and keep
    a global reference so subsequent queries can use it.
    """
    global VECTOR_STORE

    # Use a unique directory per embedding session to avoid ObjectBox
    # "another store is still open using the same path" errors when
    # rebuilding the vector store multiple times in a single process.
    db_root = os.path.join(os.path.dirname(__file__), "project", "objectbox")
    os.makedirs(db_root, exist_ok=True)
    db_directory = os.path.join(db_root, f"session_{int(time.time())}")
    os.makedirs(db_directory, exist_ok=True)

    embeddings = huggingface_instruct_embedding()

    docs: List[Document] = []

    # Persist uploads temporarily to disk so loaders/OCR can read them
    for uploaded_file in files:
        if not uploaded_file.filename:
            continue

        suffix = os.path.splitext(uploaded_file.filename)[1].lower()
        temp_path = os.path.join("/tmp", f"docchat_{int(time.time() * 1000)}_{uploaded_file.filename}")

        # Basic identifiers for later filtering / multi-user support
        document_id = str(uuid.uuid4())
        user_id = "local-user"

        with open(temp_path, "wb") as temp_file:
            temp_file.write(uploaded_file.file.read())

        try:
            if uploaded_file.content_type == "application/pdf" or suffix == ".pdf":
                documents = process_pdf(temp_path)
            elif uploaded_file.content_type in ("image/jpeg", "image/png") or suffix in (".jpg", ".jpeg", ".png"):
                documents = process_image(temp_path, uploaded_file.filename)
            else:
                # Skip unsupported types but do not fail the entire request
                continue

            # Enrich metadata for each document page / image
            for d in documents:
                meta = dict(d.metadata or {})
                meta.setdefault("source", uploaded_file.filename)
                meta.setdefault("source_name", uploaded_file.filename)
                meta.setdefault("document_id", document_id)
                meta.setdefault("user_id", user_id)
                # For PDFs, page is usually present; for others, leave as None
                if "page" not in meta:
                    meta["page"] = 0 if suffix == ".pdf" else None
                d.metadata = meta

            docs.extend(documents)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if not docs:
        raise HTTPException(status_code=400, detail="No supported documents were uploaded.")

    # Use smaller, overlapping chunks to preserve context and improve retrieval quality
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    final_documents = text_splitter.split_documents(docs)

    # Derive lightweight section titles / heading hints per chunk
    def infer_section_title(text: str) -> Optional[str]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None
        first = lines[0]
        return first if len(first) <= 120 else first[:117] + "..."

    for d in final_documents:
        meta = dict(d.metadata or {})
        if not meta.get("section_title"):
            meta["section_title"] = infer_section_title(d.page_content)
        if not meta.get("heading_hierarchy"):
            title = meta.get("section_title")
            meta["heading_hierarchy"] = [title] if title else []
        d.metadata = meta

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
    """
    Upload one or more PDF or image files and build the vector database.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    try:
        doc_count = build_vector_store(files)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error embedding documents: {e}")

    return {"message": "Documents embedded successfully.", "document_chunks": doc_count}


@app.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest):
    """
    Ask a question based on the previously uploaded and embedded documents.
    """
    if VECTOR_STORE is None:
        raise HTTPException(status_code=400, detail="No vector store available. Please upload and embed documents first.")

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    document_chain = create_stuff_documents_chain(groq_llm(), prompt)

    # First: manual similarity search so we can compute confidence scores.
    try:
        results: List[Tuple[Document, float]] = VECTOR_STORE.similarity_search_with_score(
            payload.question, k=8
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during retrieval: {e}")

    if not results:
        return QueryResponse(
            answer="I couldn't find a reliable answer in your documents.",
            response_time=0.0,
            sources=[],
        )

    # Many vector stores return distance where lower is better.
    # Convert to a rough similarity in [0, 1] using 1 / (1 + distance).
    sims = [1.0 / (1.0 + float(score)) for _, score in results]
    max_sim = max(sims)
    top3_sim = sims[:3]
    avg_top3 = sum(top3_sim) / len(top3_sim)

    high_confidence = max_sim >= 0.75 or avg_top3 >= 0.7

    docs = [doc for doc, _ in results]

    start = time.process_time()

    if high_confidence:
        # High-confidence path: let the LLM synthesize an answer from the top chunks.
        try:
            response = document_chain.invoke({"input": payload.question, "context": docs})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during retrieval: {e}")

        answer = (response.get("answer") or "").strip()
    else:
        # Low-confidence path: don't hallucinate, just explain and show snippets.
        answer = (
            "I couldn't find a reliable answer in your documents. "
            "Here are the most relevant snippets I found; you may want to review these sections."
        )

    elapsed = time.process_time() - start

    sources: List[SourceChunk] = []
    for doc, _score in results:
        metadata = getattr(doc, "metadata", {}) or {}
        sources.append(
            SourceChunk(
                content=getattr(doc, "page_content", ""),
                source=str(
                    metadata.get("source_name")
                    or metadata.get("source")
                    or metadata.get("file_path")
                    or ""
                ),
                page=metadata.get("page", None),
            )
        )

    if not answer:
        answer = "No answer could be generated from the current document context."

    return QueryResponse(answer=answer, response_time=elapsed, sources=sources)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)



