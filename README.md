# **Assistente Conversacional para Documentos PDF (Tarefa AS05)**

Este projeto implementa um assistente conversacional (chatbot) capaz de responder perguntas sobre o conteúdo de uma coleção de documentos PDF. Utilizando um modelo de linguagem grande (LLM) de última geração e a arquitetura de **Geração Aumentada por Recuperação (RAG)**, a aplicação indexa o texto dos documentos e fornece respostas contextuais através de uma interface web interativa.

Este repositório contém o código-fonte e as instruções para a tarefa **AS05: Implementação de Assistente Conversacional Baseado em LLM**.

## **Funcionalidades Principais**

-   **Carregamento de Múltiplos PDFs:** Processa todos os arquivos `.pdf` localizados na pasta `/data`.
-   **Indexação Vetorial:** Divide os textos em pedaços, gera *embeddings* (vetores numéricos) para cada um e os armazena em um banco de dados vetorial local usando **FAISS**.
-   **Interface de Chat Interativa:** Utiliza **Streamlit** para criar uma interface web amigável onde o usuário pode fazer perguntas em linguagem natural.
-   **Geração de Respostas com LLMs:** Conecta-se a modelos de linguagem poderosos (como Llama 3 ou Mixtral) hospedados na **Hugging Face** para gerar respostas baseadas no contexto recuperado.
-   **Depuração de Contexto:** Permite visualizar os trechos de texto exatos que o LLM utilizou para formular a resposta, garantindo transparência e auxiliando na avaliação da relevância.

## **Tecnologias Utilizadas**

-   **Linguagem:** Python 3.9+
-   **Framework Principal:** LangChain
-   **Interface Web:** Streamlit
-   **Modelos de Linguagem (LLMs):** Hugging Face Hub (Llama 3, Mixtral, etc.)
-   **Embeddings e Vetorização:** Sentence Transformers, FAISS (Facebook AI Similarity Search)
-   **Manipulação de PDFs:** PyPDF

## **Estrutura do Projeto**
/
├── data/                  # Pasta para colocar seus arquivos PDF.

├── vector_store/          # Criada automaticamente para salvar o índice FAISS.

├── .env                   # Arquivo local para sua chave de API (NÃO ENVIAR PARA O GIT).

├── .gitignore             # Especifica arquivos a serem ignorados pelo Git.

├── app.py                 # Código da aplicação web Streamlit.

├── ingest.py              # Script para processar os PDFs e criar o índice vetorial.

├── requirements.txt       # Lista de dependências Python.

└── README.md              # Este arquivo.
## **Instruções de Instalação e Configuração**

Siga estes passos para configurar e executar o projeto em seu ambiente local.

### **1. Clonar o Repositório**

git clone https://github.com/Missiaggia/as05-assistente-conversacional-pdf.git

cd as05-assistente-conversacional-pdf

### **2. Instalar as Dependências**

pip install -r requirements.txt

### **3. Configurar a Chave de API da Hugging Face**

A aplicação precisa de uma chave de API para acessar os modelos no Hugging Face Hub.

Crie uma conta em huggingface.co.

Acesse seu Perfil -> Settings -> Access Tokens.

Crie um novo token com permissão de read.

Na raiz do projeto, crie um arquivo chamado .env (pode usar o .env.exemple como base).

Adicione sua chave a este arquivo da seguinte forma:

HUGGINGFACEHUB_API_TOKEN="hf_sua_chave_secreta_aqui"

### **4.  Execução da Aplicação**

rode o comando: streamlit run app.py

Seu navegador padrão deverá abrir automaticamente no endereço http://localhost:8501 com a aplicação em funcionamento(provavelmente se for sua primeira vez usando o streamlit ele vai pedir para fazer login, porem não é obrigatorio, so apertar enter com os espaços em branco que o projeto ira rodar).

Com isso é so fazer upload de seus arquivos pedir para serem processados e apos isso so fazer perguntas sobre ele!