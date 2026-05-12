import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import time
import glob

# LangChain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# =========================================================
# AZURE OPENAI CLIENT
# =========================================================

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version="2024-02-15-preview"
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HelixPoint AI",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.stChatMessage {
    border-radius: 14px;
    padding: 12px;
}

.block-container {
    padding-top: 2rem;
}

.big-title {
    font-size: 46px;
    font-weight: 700;
    color: white;
}

.subtitle {
    color: #a0a0a0;
    margin-bottom: 25px;
}

.source-box {
    background-color: #1e1e1e;
    padding: 10px;
    border-radius: 10px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DOCUMENTS
# =========================================================

@st.cache_resource
def load_vectorstore():

    documents = []

    for file in glob.glob("documents/*.md"):

        loader = TextLoader(file, encoding="utf-8")
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

vectorstore = load_vectorstore()

# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are HelixPoint AI Assistant.

You are a professional enterprise onboarding and workflow assistant.

Your job is to:
- Answer employee questions
- Explain onboarding processes
- Explain HR and IT workflows
- Assist with internal company procedures
- Recommend escalation when appropriate

Rules:
- Be concise
- Be professional
- Use bullet points when useful
- Never invent policies
- If unsure, say you are unsure
- Use retrieved company knowledge when available
"""

# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🧠 HelixPoint AI")

    st.markdown("""
### Enterprise Workflow Assistant

Powered By:
- Azure OpenAI
- GPT-4.1
- LangChain
- FAISS
- Streamlit
""")

    st.divider()

    st.markdown("### Example Questions")

    examples = [
        "How do I request leave?",
        "How do I reset my VPN password?",
        "Explain the onboarding process.",
        "Who handles payroll issues?",
        "What is the escalation policy?",
        "How do I report a security incident?"
    ]

    for example in examples:
        st.markdown(f"- {example}")

    st.divider()

    if st.button("🗑 Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="big-title">🧠 HelixPoint AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Enterprise onboarding and workflow assistant powered by Azure OpenAI + RAG</div>',
    unsafe_allow_html=True
)

# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# =========================================================
# USER INPUT
# =========================================================

user_input = st.chat_input("Ask a workflow or onboarding question...")

# =========================================================
# HANDLE USER INPUT
# =========================================================

if user_input:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # =====================================================
    # RETRIEVE DOCUMENTS
    # =====================================================

    retrieved_docs = vectorstore.similarity_search(
        user_input,
        k=4
    )

    retrieved_context = "\n\n".join([
        doc.page_content for doc in retrieved_docs
    ])

    # =====================================================
    # BUILD PROMPT
    # =====================================================

    final_system_prompt = f"""
{SYSTEM_PROMPT}

RELEVANT COMPANY KNOWLEDGE:
{retrieved_context}
"""

    api_messages = [
        {
            "role": "system",
            "content": final_system_prompt
        }
    ]

    api_messages.extend(st.session_state.messages)

    # =====================================================
    # ASSISTANT RESPONSE
    # =====================================================

    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        full_response = ""

        try:

            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=api_messages,
                temperature=0.3,
                max_tokens=700
            )

            assistant_response = response.choices[0].message.content

            # Typing effect
            for chunk in assistant_response.split():

                full_response += chunk + " "

                time.sleep(0.015)

                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # Sources
            with st.expander("📚 Retrieved Knowledge"):

                for i, doc in enumerate(retrieved_docs):

                    st.markdown(f"""
<div class="source-box">
<b>Source {i+1}</b><br>
{doc.page_content[:500]}
</div>
""", unsafe_allow_html=True)

            # Save assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

        except Exception as e:

            st.error("Error communicating with Azure OpenAI")

            st.exception(e)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "HelixPoint AI Assistant • Azure OpenAI • LangChain • FAISS • GPT-4.1"
)
