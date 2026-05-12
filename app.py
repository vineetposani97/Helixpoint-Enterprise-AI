import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# ----------------------------
# Azure OpenAI Configuration
# ----------------------------

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version="2024-02-15-preview"
)

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="HelixPoint AI Workflow Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 HelixPoint AI Workflow Assistant")

st.markdown("""
Enterprise onboarding and workflow assistant powered by Azure OpenAI.

This assistant helps employees:
- navigate onboarding
- understand workflows
- receive IT and HR guidance
- escalate sensitive requests responsibly
""")

# -----------------------------
# User Input
# -----------------------------

user_input = st.text_input(
    "Ask a workflow or onboarding question:"
)

# -----------------------------
# AI Prompt
# -----------------------------

SYSTEM_PROMPT = """
You are an enterprise AI workflow assistant for HelixPoint Technologies.

Your role:
- support onboarding workflows
- answer HR and IT process questions
- provide professional enterprise guidance
- follow responsible AI principles

Rules:
- Never approve PTO, payroll, promotions, or confidential actions.
- Escalate sensitive decisions to HR, IT, or management.
- Respond professionally and clearly.
- Keep responses concise but useful.
"""

# -----------------------------
# Generate Response
# -----------------------------

if st.button("Generate Response"):

    if user_input.strip():

        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {
                    
                    "role": "system",
                    "content": """
                    You are an enterprise onboarding and workflow assistant.
                    Help employees with:
                    - onboarding
                    - HR questions
                    - IT workflows
                    - escalation procedures
                    - company policies
                    Be professional, concise, and structured.
                    """
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            temperature=0.7
        )

        answer = response.choices[0].message.content

        st.subheader("Assistant Response")
        st.write(response.choices[0].message.content)

    else:
        st.warning("Please enter a question.")
