# 🏗 HelixPoint Enterprise AI Architecture

## System Overview

HelixPoint Enterprise AI is a multi-agent enterprise operations platform designed to simulate real-world AI workflow orchestration inside modern organizations.

The platform combines:
- Azure OpenAI
- Retrieval-Augmented Generation (RAG)
- Multi-Agent Routing
- Enterprise Workflow Automation
- Operational Analytics
- Persistent Ticket Logging

---

# 🔄 High-Level Workflow

User Query
↓
Intent Classification & Agent Routing
↓
Specialized Enterprise Agent Selection
↓
Enterprise Knowledge Retrieval (FAISS)
↓
Azure OpenAI Response Generation
↓
Workflow Automation + Ticket Logging
↓
Analytics & Operational Insights

---

# 🤖 Multi-Agent Architecture

## 👨‍💼 HR Agent
Handles:
- PTO requests
- onboarding support
- employee policy guidance
- HR workflow automation

---

## 💻 IT Support Agent
Handles:
- password resets
- VPN troubleshooting
- Microsoft Teams access
- enterprise IT escalation workflows

---

## 🔒 Security Agent
Handles:
- suspicious login detection
- phishing escalation
- incident response workflows
- security alert automation

---

## 📊 Analytics Agent
Handles:
- KPI reporting
- operational trend analysis
- dashboard metrics
- enterprise analytics insights

---

## 🏢 Enterprise Operations Agent
Fallback enterprise assistant responsible for:
- workflow orchestration
- operational support
- general enterprise assistance

---

# 🧠 Retrieval-Augmented Generation (RAG)

The platform uses:
- Sentence Transformers embeddings
- FAISS vector similarity search
- contextual enterprise knowledge retrieval

to generate grounded and enterprise-aware responses.

---

# 🗄 Persistent Storage

SQLite is used for:
- ticket logging
- operational persistence
- workflow tracking
- escalation history

---

# 📊 Analytics Layer

Plotly dashboards visualize:
- enterprise ticket distribution
- workflow activity
- onboarding metrics
- operational KPIs

---

# ☁ Deployment

Frontend:
- Streamlit Cloud

Backend Stack:
- Python
- Azure OpenAI
- LangChain
- FAISS
- SQLite