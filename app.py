import os
import streamlit as st
from dotenv import load_dotenv

from ingest import create_vector_store

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace

load_dotenv()

os.makedirs("data", exist_ok=True)

# --- Constantes ---
VECTOR_STORE_PATH = "vector_store/"
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct" 

# --- Funções ---
def create_rag_chain():
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs
    )
    try:
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Erro ao carregar o Vector Store: {e}")
        st.error("Verifique se você executou 'ingest.py' e se a pasta 'vector_store' não está vazia.")
        return None

    retriever = vector_store.as_retriever()

    base_llm = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        max_new_tokens=512,
        temperature=0.7,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

    llm = ChatHuggingFace(llm=base_llm)

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

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

st.set_page_config(page_title="Assistente de Documentos PDF", page_icon="🤖")

st.title("🤖 Assistente Conversacional para Documentos PDF")
st.markdown("""
Esta aplicação permite que você converse com seus documentos PDF.
Faça upload de um PDF ou use os documentos já processados.
""")

st.header("📁 Gerenciar Documentos PDF")

pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
if pdf_files:
    st.subheader("Documentos salvos:")
    for pdf_file in pdf_files:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.text(pdf_file)
        with col2:
            if st.button("❌", key=f"delete_{pdf_file}"):
                os.remove(os.path.join("data", pdf_file))
                st.success(f"Arquivo {pdf_file} excluído!")
                st.rerun()

uploaded_file = st.file_uploader("Escolha um arquivo PDF", type="pdf")

if uploaded_file is not None:
    if st.button("Salvar e Processar PDF"):
        with st.spinner("Salvando e processando o PDF..."):
            pdf_path = os.path.join("data", uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            success = create_vector_store(pdf_path)
            
            if success:
                st.success(f"PDF {uploaded_file.name} salvo e processado com sucesso!")
                st.rerun()
            else:
                st.error("Erro ao processar o PDF. Verifique se o arquivo é válido.")
                os.remove(pdf_path)

if pdf_files:
    if st.button("Reprocessar Todos os PDFs"):
        with st.spinner("Reprocessando todos os documentos..."):
            success = create_vector_store()
            if success:
                st.success("Todos os documentos foram reprocessados!")
                st.rerun()
            else:
                st.error("Erro ao reprocessar os documentos.")

st.divider()

st.header("💬 Chat com Documentos")

if not os.path.exists(VECTOR_STORE_PATH) or not os.listdir(VECTOR_STORE_PATH):
    st.warning("Nenhum documento foi processado ainda. Faça upload de um PDF acima para começar.")
else:
    rag_chain = create_rag_chain()

    if rag_chain:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Faça sua pergunta sobre o documento..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Buscando a resposta..."):
                    response = rag_chain.invoke(prompt)
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

        if st.button("Limpar Conversa"):
            st.session_state.messages = []
            st.rerun()