import time
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_core.prompts import ChatPromptTemplate
from utils import groq_llm, huggingface_instruct_embedding
import pytesseract
from langchain.docstore.document import Document
from PIL import Image
import os

st.set_page_config(layout='wide', page_title="DocChat")
st.title("DocChat")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def process_pdf(uploaded_file):
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    loader = PyPDFLoader(file_path=temp_file_path)
    documents = loader.load()
    os.remove(temp_file_path)
    return [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in documents]

def process_image(uploaded_file):
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)
    return [Document(page_content=text, metadata={"source": uploaded_file.name})]

def vector_embedding(uploaded_files):
    db_directory = 'project/objectbox'
    if not os.path.exists(db_directory):
        os.makedirs(db_directory)

    if 'vectors' not in st.session_state:
        st.session_state.embeddings = huggingface_instruct_embedding()

        st.session_state.docs = []
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                documents = process_pdf(uploaded_file)
            elif uploaded_file.type in ["image/jpeg", "image/png"]:
                documents = process_image(uploaded_file)
            else:
                st.warning(f"unsupported file type: {uploaded_file.type}")
                continue
            st.session_state.docs.extend(documents)

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        st.write(f"Documents loaded: {len(st.session_state.final_documents)}")
        st.write(f"Embeddings: {st.session_state.embeddings}")

        try:
            st.session_state.vectors = ObjectBox.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings,
                embedding_dimensions=768,
                db_directory=db_directory
            )
            st.write('ObjectBox Database is ready.')
        except Exception as e:
            st.error(f"Error creating ObjectBox database: {e}")
            return
        
uploaded_files = st.file_uploader(
    "upload your PDF or image documents (JPEG, PNG)",
    type=["pdf", "jpeg", "jpg", "png"],
    accept_multiple_files=True
)

if st.button('Embedd Documents') and uploaded_files:
    vector_embedding(uploaded_files)
    st.write("database is ready. You can now enter your Question")

user_input = st.text_input("Enter your question from documents")

if user_input:
    document_chain = create_stuff_documents_chain(groq_llm(), prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()

    try:
        response = retrieval_chain.invoke({'input': user_input})
        st.write(response['answer'])
        st.write(f'response time: {(time.process_time() - start):.2f} secs')
    except Exception as e:
        st.error(f"Error during retrieval: {e}")

    with st.expander("Document similarity search"):
        if 'context' in response:
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("-------------------------------")
        else:
            st.write("no similar documents found.")

