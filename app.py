import streamlit as st
import requests
import json
import os

# Set a basic page config
st.set_page_config(page_title="Major Incident Manager")

# Function to read and serve the HTML content
def get_html_content(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# --- Backend API Endpoint for Chat ---
def handle_chat_request(user_input, conversation_state, cmdb_data, simulated_logs):
    """
    Handles the chat request by making a secure call to the OpenAI API.
    """
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {openai_api_key}'
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": f"""You are a Major Incident Manager bot. Your goal is to guide the user through a structured incident response workflow. Your responses must be concise, specific, and formatted for a chat interface.
                        * **Agent 1 (Triage):** Greets the user and asks for the application name.
                        * **Agent 2 (CMDB):** Performs a lookup based on the user's input. If found, it provides a message and a JSON object with associated CIs.
                        * **Agent 3 (Log Analysis):** Summarizes the provided logs and sends the findings to Agent 4 for RCA. It also provides the full logs.
                        * **Agent 4 (RCA):** Provides the root cause, fix, and preventative measures.
                        * **Agent 5 (Helper):** Provides guidance if the user enters an unexpected command.
                    
                    Your responses should always begin with the format "Agent X:" where X is the agent number. This is critical for the UI to display the correct agent name.
                    
                    Use the following data as your context:
                    
                    CMDB Data:
                    {json.dumps(cmdb_data, indent=2)}
                    
                    Simulated Logs:
                    {simulated_logs}
                    
                    Current conversation state: {conversation_state}"""
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data['choices'][0]['message']['content']
        
    except Exception as e:
        return f"An error occurred: {e}"

# --- Main Streamlit App Logic ---
st.markdown(
    get_html_content("index.html"),
    unsafe_allow_html=True
)

# Use st.session_state to manage the conversation
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = "initial"
if "app_data" not in st.session_state:
    st.session_state.app_data = {
        "cmdb": json.loads(open("cmdb.json").read()),
        "logs": open("logs.txt").read()
    }

# This is the endpoint that the JavaScript will call
if st.experimental_get_query_params().get("chat"):
    user_input = st.experimental_get_query_params()["chat"][0]
    conversation_state = st.experimental_get_query_params()["state"][0]
    
    response_text = handle_chat_request(
        user_input, 
        conversation_state,
        st.session_state.app_data["cmdb"],
        st.session_state.app_data["logs"]
    )
    st.json({"response": response_text})
    
