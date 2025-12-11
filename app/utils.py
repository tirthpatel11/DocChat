from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from config import load_config, get_groq_api

load_config()


def groq_llm():
    """
    Return a Groq chat model instance.

    The previous model `llama3-8b-8192` has been decommissioned, so we switch
    to a currently supported Llama 3.1 model. If Groq changes model names again,
    you can update the `model_name` below.
    """
    llm = ChatGroq(
        groq_api_key=get_groq_api(),
        model_name="llama-3.1-8b-instant",
        temperature=0.1,  # low but non-zero for slightly richer answers
        max_tokens=512,
    )
    return llm

def huggingface_instruct_embedding():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name='BAAI/bge-small-en-v1.5',  
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings