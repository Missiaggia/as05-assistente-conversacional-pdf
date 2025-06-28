# app.py (versão 3 - final e correta)

import os
import streamlit as st
from dotenv import load_dotenv

# Importações necessárias de LangChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint # Importação do Endpoint
from langchain_huggingface import ChatHuggingFace     # Importação do Adaptador de Chat

# Carrega as variáveis de ambiente (sua chave da Hugging Face)
load_dotenv()

# --- Constantes ---
VECTOR_STORE_PATH = "vector_store/"
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct" 

# --- Funções ---

def create_rag_chain():
    """
    Cria a cadeia de RAG (Retrieval-Augmented Generation) completa.
    """
    # Carrega o banco de dados vetorial salvo localmente
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    try:
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Erro ao carregar o Vector Store: {e}")
        st.error("Verifique se você executou 'ingest.py' e se a pasta 'vector_store' não está vazia.")
        return None

    retriever = vector_store.as_retriever()

    # PASSO 1: Criar o Endpoint base que se conecta ao Hugging Face.
    base_llm = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        max_new_tokens=512,
        temperature=0.7,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

    # PASSO 2: Envolver o endpoint no adaptador ChatHuggingFace.
    # O argumento obrigatório 'llm' recebe o nosso endpoint base.
    llm = ChatHuggingFace(llm=base_llm)

    # Template de prompt otimizado para modelos de chat
    system_prompt = """Você é um assistente de IA focado em responder perguntas com base em um contexto fornecido.
    Use os trechos de texto do contexto para responder à pergunta do usuário.
    Se a resposta não estiver no contexto, diga claramente que você não encontrou a informação nos documentos.
    Não tente inventar uma resposta. Seja direto e útil.

    Contexto:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    # Função auxiliar para formatar os documentos recuperados em uma única string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Construindo a cadeia com a LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- Interface Gráfica com Streamlit ---

st.set_page_config(page_title="Assistente de Documentos PDF", page_icon="🤖")

st.title("🤖 Assistente Conversacional para Documentos PDF")
st.markdown("""
Esta aplicação permite que você converse com seus documentos PDF.
1.  Coloque seus arquivos PDF na pasta `data`.
2.  Execute o script `ingest.py` para processá-los.
3.  Faça suas perguntas abaixo e obtenha respostas baseadas nos documentos.
""")

# Verifica se o Vector Store existe antes de iniciar o chat
if not os.path.exists(VECTOR_STORE_PATH) or not os.listdir(VECTOR_STORE_PATH):
    st.error("O diretório do Vector Store não foi encontrado ou está vazio. Por favor, execute o script 'ingest.py' primeiro.")
else:
    # Cria a cadeia de RAG
    rag_chain = create_rag_chain()

    if rag_chain:
        # Campo de entrada para a pergunta do usuário
        user_question = st.text_input("Faça sua pergunta:")

        if user_question:
            with st.spinner("Buscando a resposta..."):
                response = rag_chain.invoke(user_question)
                
                st.subheader("Resposta:")
                st.write(response)