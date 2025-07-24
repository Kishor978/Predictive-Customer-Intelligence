import streamlit as st
import os

# Import the new LangGraph agent
from src.langgraph_agent import LangGraphAgent

# --- Streamlit UI Setup ---
st.set_page_config(page_title="PCI Query Flow Chatbot", page_icon="📈")
st.title("📈 PCI Query Flow Chatbot (Task 2)")
st.caption("This chatbot uses LangGraph to analyze customer queries and provide mock PCI suggestions.")

# --- API Key Validation and Setting Environment Variables ---
# Retrieve API keys from st.secrets
google_api_key = st.secrets.get("GOOGLE_API_KEY")
openai_api_key = st.secrets.get("OPENAI_API_KEY")

use_gemini_api = False
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    use_gemini_api = True
    st.info("Using Google Gemini API based on st.secrets.")
elif openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    use_gemini_api = False # Ensure this is false if using OpenAI
    st.info("Using OpenAI API based on st.secrets.")
else:
    st.error("Error: No GOOGLE_API_KEY or OPENAI_API_KEY found in `.streamlit/secrets.toml`.")
    st.error("Please add your API key to `.streamlit/secrets.toml` to continue.")
    st.stop() # Stop the app if no API key is found

# Initialize chatbot (LangGraphAgent) in Streamlit's session state
if "agent" not in st.session_state:
    try:
        st.session_state.agent = LangGraphAgent(
            use_gemini=use_gemini_api,
            memory_type="buffer" # "buffer" or "summary"
        )
        st.success("LangGraph Agent initialized successfully!")
    except Exception as e:
        st.error(f"An error occurred during LangGraph Agent initialization: {e}")
        st.error("Please check your model name in `langgraph_agent.py` or your API key permissions.")
        st.stop() # Stop the app if initialization fails

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I help you with your query today?"})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing your query..."):
            try:
                response = st.session_state.agent.run_agent(prompt)
                st.markdown(response)
            except Exception as e:
                st.error(f"Error getting agent response: {e}")
                response = "I'm sorry, I encountered an error. Please try again."

        st.session_state.messages.append({"role": "assistant", "content": response})