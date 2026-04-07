from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import CrossEncoder
from config import load_config, get_groq_api

load_config()

_reranker = None


def groq_llm():
    return ChatGroq(
        groq_api_key=get_groq_api(),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024,
    )


def huggingface_instruct_embedding():
    return HuggingFaceBgeEmbeddings(
        model_name='BAAI/bge-base-en-v1.5',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            max_length=512,
        )
    return _reranker
