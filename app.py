import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import time

# ==================================================
# LOAD ENV VARIABLES
# ==================================================

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# ==================================================
# AZURE OPENAI CLIENT
# ==================================================

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version="2024-02-15-preview"
)

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="HelixPoint AI Assistant",
    page_icon=":robot_face:",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.main {
    background-color: #000000;
}

.stChatMessage {
    border-radius: 12px;
    padding: 10px;
}

.block-container {
    padding-top: 2rem;
}

.big-title {
    font-size: 42px;
    font-weight: 700;
    color: #ffffff;
}

.subtitle {
    color: #a0a0a0;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD COMPANY KNOWLEDGE
# ==================================================

def load_file_content(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""

company_overview = load_file_content("company_overview.md")
onboarding_handbook = load_file_content("onboarding_handbook.md")
sample_interactions = load_file_content("sample_interactions.md")

# ==================================================
# SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = f"""
You are HelixPoint AI Workflow Assistant.

You are an enterprise onboarding and workflow assistant.

Your job is to help employees:
- navigate onboarding
- understand workflows
- answer HR and IT questions
- explain company processes
- escalate sensitive issues responsibly

Behavior Rules:
- Be professional, helpful, and concise
- Use bullet points where appropriate
- Never invent policies
- If unsure, say you are unsure
- Recommend escalation for legal/security/payroll issues
- Maintain a friendly enterprise tone

COMPANY OVERVIEW:
{company_overview}

ONBOARDING HANDBOOK:
{onboarding_handbook}

SAMPLE INTERACTIONS:
{sample_interactions}
"""

# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🧠 HelixPoint AI")

    st.markdown("""
### Enterprise Workflow Assistant
Powered by:
- Azure OpenAI
- GPT-4.1
- Streamlit
""")

    st.divider()

    st.markdown("### Example Questions")

    examples = [
        "How do I request leave?",
        "How do I reset my VPN password?",
        "Who do I contact for payroll issues?",
        "Explain the onboarding process.",
        "How do I escalate a security incident?"
    ]

    for example in examples:
        st.markdown(f"- {example}")

    st.divider()

    if st.button("🗑 Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# ==================================================
# MAIN HEADER
# ==================================================

st.markdown(
    '<div class="big-title">🧠 HelixPoint AI Workflow Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">Enterprise onboarding and workflow assistant powered by Azure OpenAI.</div>',
unsafe_allow_html=True
)

# ==================================================
# DISPLAY CHAT HISTORY
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ==================================================
# USER INPUT
# ==================================================

user_input = st.chat_input("Ask a workflow or onboarding question...")

# ==================================================
# HANDLE USER MESSAGE
# ==================================================

if user_input:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Build messages for API
    api_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
    ]

    # Add conversation history
    api_messages.extend(st.session_state.messages)

    # Assistant response UI
    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        full_response = ""

        try:

            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=api_messages,
                temperature=0.7,
                max_tokens=800
            )

            assistant_response = response.choices[0].message.content

            # Fake streaming effect
            for chunk in assistant_response.split():

                full_response += chunk + " "

                time.sleep(0.02)

                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

        except Exception as e:

            st.error("⚠️ Error communicating with Azure OpenAI")

            st.exception(e)

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption("HelixPoint AI Workflow Assistant • Azure OpenAI • GPT-4.1")