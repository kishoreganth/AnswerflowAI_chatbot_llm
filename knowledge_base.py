from PyPDF2 import PdfReader
from langchain.text_splitter import  RecursiveCharacterTextSplitter, CharacterTextSplitter
from config import *
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import WebBaseLoader,DirectoryLoader, PyPDFLoader
import os
import glob
from config import *
import streamlit as st


final_loader_data = []


def get_vector():
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.load_local(VECTOR_DB_PATH, embeddings)
    return vector_db



# This will call only if new files is added to the vector
@st.cache_resource()
def get_knowledge_from_pdfs(pdf_docs, URLs):
    if pdf_docs:
        # Save PDFs to the folder
        for pdf in pdf_docs:
            file_path = os.path.join("./res/PDFs", pdf.name)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(pdf.getbuffer())
            else:
                return(get_vector())

        loader_pdf = DirectoryLoader("./res/PDFs", glob="./*.pdf", loader_cls=PyPDFLoader)
        files = glob.glob("*.pdf")

        pdf_data = loader_pdf.load()
        final_loader_data.extend(pdf_data)

    # print(pdf_data)

    ### URLS LoADER
    if URLs:
        loader_web = WebBaseLoader(URLs)
        web_data = loader_web.load()
        # print(web_data)
        final_loader_data.extend(web_data)

## IF FINAL LOADER THEN
    if final_loader_data:
        # TextSplitter for creating chunks -------------------
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=700,
            chunk_overlap=100
            # length_function=len
        )
        chunks = text_splitter.split_documents(final_loader_data)
        print(dir(chunks))
        print(type(chunks))
        # Embeddings
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
        # vectorstore.save_local(VECTOR_DB_PATH)
        return vectorstore

    # web_chunks = text_splitter.split_documents(data)
    # print(dir(web_chunks))
    # print(type(web_chunks))
    # total_chunks = []
    # total_chunks.append(chunks)
    # total_chunks.append(web_chunks)



