import streamlit as st
import requests

# Page Configuration
st.set_page_config(page_title="AI Agent Chat", layout="wide")

# Header
st.title("🤖 AI Chatbot Agent")
st.caption("Interact with powerful AI models via LangGraph agents")

# Sidebar Setup
with st.sidebar:
    st.header("🛠️ Agent Configuration")

    provider = st.radio("Select Provider", ["Groq"], index=0)
    model_options = ["llama-3.3-70b-versatile", "llama3-70b-8192"]
    sel_model = st.selectbox("Choose Model", model_options)

    system_prompt = st.text_area("🧠 System Prompt", height=100,
                                 placeholder="Define your AI Agent behavior...")
    
    alw_websearch = st.checkbox("🌐 Enable Web Search", value=False)

# Query Section
st.subheader("💬 Ask your question")
with st.form("chat_form", clear_on_submit=False):
    query = st.text_area("Your Query", height=100, placeholder="Type your question here...")
    submit = st.form_submit_button("🚀 Ask Agent")

# Backend URL
API_URL = "http://127.0.0.1:8000/chat"

# Trigger API on submit
if submit:
    if not query.strip():
        st.warning("Please enter a query before submitting.")
    else:
        payload = {
            "model_name": sel_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "prompt": [query],  # Fixed: should be a list!
            "allow_search": alw_websearch,
        }

        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                response_data = response.json()
                if isinstance(response_data, dict) and "error" in response_data:
                    st.error(f"❌ Error: {response_data['error']}")
                else:
                    st.success("✅ Response received!")
                    st.markdown("### 🤖 Agent's Response")
                    st.markdown(f"```markdown\n{response_data}\n```")
            else:
                st.error(f"⚠️ Server error: {response.status_code} - {response.text}")
        except Exception as e:
            st.exception(f"Connection failed: {e}")
