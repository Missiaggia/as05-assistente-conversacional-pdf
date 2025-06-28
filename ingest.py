# ingest.py

import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Constantes
DATA_PATH = "data/"
VECTOR_STORE_PATH = "vector_store/"

def create_vector_store():
    """
    Função para carregar documentos PDF, dividi-los em chunks,
    gerar embeddings e salvar em um Vector Store FAISS.
    """
    print("Iniciando o carregamento dos documentos PDF...")
    # Carrega todos os PDFs do diretório especificado
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    if not documents:
        print("Nenhum documento PDF encontrado no diretório 'data'.")
        return

    print(f"{len(documents)} documento(s) carregado(s).")

    print("Dividindo os documentos em chunks...")
    # Divide os documentos em chunks menores para processamento
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    print(f"Total de {len(docs)} chunks de texto criados.")

    print("Gerando embeddings e criando o Vector Store FAISS...")
    # Usa um modelo de embedding open-source da Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Cria o banco de dados vetorial com os chunks e seus embeddings
    db = FAISS.from_documents(docs, embeddings)

    # Salva o banco de dados localmente
    db.save_local(VECTOR_STORE_PATH)
    print(f"Vector Store criado e salvo com sucesso em '{VECTOR_STORE_PATH}'.")

if __name__ == "__main__":
    create_vector_store()