import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import sqlite3
import pandas as pd
import plotly.express as px
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ============================================================
# AGENT ROUTING
# ============================================================

AGENTS = {

    "HR Agent": {
        "keywords": [
            "leave",
            "pto",
            "payroll",
            "benefits",
            "onboarding",
            "vacation",
            "employee",
            "hr"
        ]
    },

    "IT Agent": {
        "keywords": [
            "password",
            "reset",
            "microsoft teams",
            "locked out",
            "outlook",
            "vpn",
            "wifi",
            "access"
        ]
    },

    "Security Agent": {
        "keywords": [
            "security",
            "breach",
            "suspicious",
            "phishing",
            "unauthorized",
            "hack",
            "malware",
            "another country",
            "incident"
        ]
    },

    "Analytics Agent": {
        "keywords": [
            "dashboard",
            "analytics",
            "metrics",
            "report",
            "kpi",
            "data",
            "trend",
            "performance"
        ]
    },

    "Enterprise Operations Agent": {
        "keywords": [
            "workflow",
            "operations",
            "process",
            "escalation",
            "support",
            "ticket",
            "system"
        ]
    }

}

# =========================================================
# ENV VARIABLES
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
    page_title="HelixPoint Enterprise AI",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stChatMessage {
    border-radius: 16px;
    padding: 14px;
}

.big-title {
    font-size: 52px;
    font-weight: 800;
}

.subtitle {
    color: #9ca3af;
    font-size: 18px;
    margin-bottom: 20px;
}

.metric-card {
background: rgba(120,120,120,0.08);
padding: 16px;
border-radius: 12px;
border: 1px solid rgba(128,128,128,0.2);
}

.workflow-card {
background: rgba(120,120,120,0.08);
padding: 18px;
border-radius: 14px;
border-left: 5px solid #4F46E5;
margin-top: 14px;
border: 1px solid rgba(128,128,128,0.2);
}

.agent-card {
background: rgba(120,120,120,0.08);
padding: 12px;
border-radius: 10px;
margin-bottom: 10px;
border: 1px solid rgba(128,128,128,0.2);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# DATABASE
# =========================================================

conn = sqlite3.connect("helixpoint.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT,
    priority TEXT,
    issue TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

# =========================================================
# VECTOR STORE
# =========================================================

@st.cache_resource
def load_vectorstore():

    documents = []

    if os.path.exists("documents"):

        for file in os.listdir("documents"):

            if file.endswith(".md"):

                loader = TextLoader(
                    os.path.join("documents", file),
                    encoding="utf-8"
                )

                documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

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
You are HelixPoint Enterprise AI.

You are an elite enterprise operations copilot.

You help with:
- IT Operations
- HR Workflows
- Security Escalations
- Incident Management
- Employee Support
- Workflow Automation
- Meeting Intelligence

Rules:
- Be concise
- Be highly professional
- Think like an enterprise operations platform
- Recommend automation opportunities
- Use bullet points
- Escalate security concerns immediately
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

    st.markdown("### Enterprise Modules")

    st.markdown("""
    - HR Operations
    - IT Service Desk
    - Security Operations
    - Workflow Automation
    - Analytics Dashboard
    - Knowledge Retrieval
    """)

    st.divider()

    st.markdown("### System Status")

    st.success("AI Systems Online")
    st.success("Security Monitoring Active")
    st.warning("2 Pending Escalations")

    st.divider()

    st.markdown("### Active AI Agents")

    st.markdown("""
    <div class="agent-card">👨‍💼 HR Agent</div>
    <div class="agent-card">💻 IT Agent</div>
    <div class="agent-card">🛡️ Security Agent</div>
    <div class="agent-card">📊 Analytics Agent</div>
    """, unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="big-title">🧠 HelixPoint Enterprise AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Enterprise Operations Copilot powered by Azure OpenAI + Multi-Agent Workflow Automation</div>',
    unsafe_allow_html=True
)

# =========================================================
# DASHBOARD METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Tickets Resolved", "142", "+18%")

with c2:
    st.metric("Security Alerts", "8", "-12%")

with c3:
    st.metric("Onboarding Progress", "93%", "+6%")

with c4:
    st.metric("AI Automations", "58", "+31%")

# =========================================================
# ANALYTICS DASHBOARD
# =========================================================

st.divider()

analytics_df = pd.DataFrame({
    "Category": [
        "IT",
        "HR",
        "Security",
        "Onboarding",
        "Compliance"
    ],
    "Tickets": [
        52,
        31,
        14,
        25,
        11
    ]
})

chart = px.bar(
    analytics_df,
    x="Category",
    y="Tickets",
    title="Enterprise Ticket Distribution"
)

st.plotly_chart(chart, use_container_width=True)

# =========================================================
# CHAT HISTORY
# =========================================================

st.divider()

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# =========================================================
# USER INPUT
# =========================================================

user_input = st.chat_input(
    "Ask about workflows, onboarding, incidents, IT support, security..."
)

# =========================================================
# MAIN AI WORKFLOW
# =========================================================

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    retrieved_docs = vectorstore.similarity_search(
        user_input,
        k=4
    )

    retrieved_context = "\\n\\n".join([
        doc.page_content for doc in retrieved_docs
    ])

    final_prompt = f"""
    {SYSTEM_PROMPT}

    ENTERPRISE KNOWLEDGE:
    {retrieved_context}
    """

    api_messages = [
        {
            "role": "system",
            "content": final_prompt
        }
    ]

    api_messages.extend(st.session_state.messages)

    workflow_output = ""

    query = user_input.lower()

    # =====================================================
    # AGENT ROUTING
    # =====================================================

    active_agent = "Enterprise Operations Agent"

    for agent_name, agent_data in AGENTS.items():

        if any(keyword in query.lower() for keyword in agent_data["keywords"]):

            active_agent = agent_name
            break

        if active_agent == "HR Agent":

            workflow_output = """
            <div class="workflow-card">

            ## 👨‍💼 HR Workflow Automation

            ### PTO Request Generated
            - Status: Pending Manager Approval
            - HR Workflow Initiated
            - Calendar Conflict Check Completed

            ### Recommended Actions
            - Confirm leave dates
            - Assign backup owner
            - Notify reporting manager

            </div>
            """

            cursor.execute("""
            INSERT INTO tickets(category, priority, issue)
            VALUES (?, ?, ?)
            """, ("HR", "Medium", user_input))

            conn.commit()

        elif active_agent == "IT Agent":

            workflow_output = """
            <div class="workflow-card">

            ## 💻 IT Incident Workflow

            ### Ticket Created
            - Priority: Medium
            - Category: Access Management
            - IT Queue Updated

            ### Automated Actions
            - MFA verification initiated
            - Credential reset workflow started
            - Escalation policy attached

            </div>
            """

            cursor.execute("""
            INSERT INTO tickets(category, priority, issue)
            VALUES (?, ?, ?)
            """, ("IT", "High", user_input))

            conn.commit()

        elif active_agent == "Security Agent":

            workflow_output = """
            <div class="workflow-card">

            ## 🛡️ Security Escalation Workflow

            ### Severity: HIGH

            ### Automated Response
            - Security Operations alerted
            - Device isolation recommended
            - Incident report generated
            - Compliance workflow initiated

            ### Immediate Next Steps
            - Reset credentials
            - Review access logs
            - Notify SOC team

            </div>
            """

            cursor.execute("""
            INSERT INTO tickets(category, priority, issue)
            VALUES (?, ?, ?)
            """, ("Security", "Critical", user_input))

            conn.commit()

    # =====================================================
    # ASSISTANT RESPONSE
    # =====================================================

    with st.chat_message("assistant"):

        st.markdown(f"### 🤖 Active Agent: {active_agent}")

        message_placeholder = st.empty()

        full_response = ""

        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=api_messages,
            temperature=0.3,
            max_tokens=700
        )

        assistant_response = response.choices[0].message.content

        for chunk in assistant_response.split():

            full_response += chunk + " "

            time.sleep(0.01)

            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

        if workflow_output:
            st.markdown(
                workflow_output,
                unsafe_allow_html=True
            )

        with st.expander("📚 Enterprise Knowledge Sources"):

            for i, doc in enumerate(retrieved_docs):

                st.markdown(f"""
<div class="workflow-card">

### Source {i+1}

{doc.page_content[:500]}

</div>
""", unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "HelixPoint Enterprise AI • Multi-Agent Operations Platform • Azure OpenAI • LangChain • FAISS • SQLite"
)

