# ingest.py

import os
import re
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/"
VECTOR_STORE_PATH = "vector_store/"

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    return text.strip()

def create_vector_store(pdf_path=None):
    print("Iniciando o carregamento dos documentos PDF...")
    
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    
    if pdf_path:
        print(f"Novo PDF adicionado: {pdf_path}")
    print(f"Processando todos os PDFs da pasta data...")
    
    if not documents:
        print("Nenhum documento PDF encontrado na pasta data.")
        return False

    print(f"{len(documents)} página(s) carregada(s).")

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        if 'source' in doc.metadata:
            doc.metadata['filename'] = os.path.basename(doc.metadata['source'])

    print("Dividindo os documentos em chunks otimizados...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=200, 
        separators=["\n\n", "\n", ".", "!", "?", ";", ":", " "],
        length_function=len,
    )
    docs = text_splitter.split_documents(documents)
    
    docs = [doc for doc in docs if len(doc.page_content.strip()) > 50]
    print(f"Total de {len(docs)} chunks de texto criados.")

    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)
        print("Vector Store antigo removido.")
    
    print("Gerando embeddings e criando novo Vector Store FAISS...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    db = FAISS.from_documents(docs, embeddings)
    print("Novo Vector Store criado do zero.")

    db.save_local(VECTOR_STORE_PATH)
    print(f"Vector Store salvo com sucesso em '{VECTOR_STORE_PATH}'.")
    return True

if __name__ == "__main__":
    create_vector_store()