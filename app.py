import streamlit as st
import pandas as pd
import graphviz
from openai import OpenAI
import base64
from streamlit_audiorecorder import audiorecorder

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Incident Manager",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Styling ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
    .stSpinner {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
    }
</style>
""", unsafe_allow_html=True)

# --- Data Simulation (from original JS) ---
# This data is used to ground the AI model and for UI displays.
CMDB_DF = pd.DataFrame([
    {'id': 'app-a', 'type': 'Application', 'name': 'Web Storefront', 'associated_cis': ['lb-a', 'web-s-1', 'web-s-2', 'pg-db-a', 'data-int-svc', 'pay-api']},
    {'id': 'app-b', 'type': 'Application', 'name': 'SAP S/4HANA', 'associated_cis': ['sap-as-1', 'hana-db']},
    {'id': 'app-c', 'type': 'Application', 'name': 'Salesforce CRM', 'associated_cis': ['sf-int-s', 'data-int-svc']},
    {'id': 'app-d', 'type': 'Application', 'name': 'Legacy Mainframe', 'associated_cis': ['mf-z']},
    {'id': 'app-e', 'type': 'Application', 'name': 'Data Integration Service', 'associated_cis': ['sap-sf-if', 'sf-int-s', 'web-s-1']},
    {'id': 'app-f', 'type': 'Application', 'name': 'Billing Microservice', 'associated_cis': ['bill-host', 'mysql-db']},
    {'id': 'app-g', 'type': 'Application', 'name': 'Reporting Dashboard', 'associated_cis': ['report-host', 'pg-db-a']},
    {'id': 'lb-a', 'type': 'Load Balancer', 'name': 'NGINX Load Balancer', 'associated_cis': []},
    {'id': 'web-s-1', 'type': 'Server', 'name': 'Web Server 1', 'associated_cis': ['app-a']},
    {'id': 'web-s-2', 'type': 'Server', 'name': 'Web Server 2', 'associated_cis': ['app-a']},
    {'id': 'pg-db-a', 'type': 'Database', 'name': 'PostgreSQL DB A', 'associated_cis': ['web-s-1', 'web-s-2']},
    {'id': 'hana-db', 'type': 'Database', 'name': 'SAP HANA DB', 'associated_cis': ['sap-as-1']},
    {'id': 'sap-as-1', 'type': 'Server', 'name': 'SAP Application Server', 'associated_cis': ['app-b', 'hana-db']},
    {'id': 'sf-int-s', 'type': 'Server', 'name': 'Salesforce Integration Server', 'associated_cis': ['app-c', 'app-e']},
    {'id': 'sap-sf-if', 'type': 'Interface', 'name': 'SAP-Salesforce Interface', 'associated_cis': ['app-e']},
])

SIMULATED_LOGS = """
2025-09-03 22:15:01 [ERROR] [Web Storefront] - Failed to submit order, dependency timeout.
2025-09-03 22:15:02 [ERROR] [Data Integration Service] - Connection to SAP system failed.
2025-09-03 22:15:03 [WARN] [SAP HANA DB] - High volume of failed login attempts from 'Data Integration Service'.
2025-09-03 22:15:06 [ERROR] [SAP-Salesforce Interface] - SSL Handshake failed, certificate expired.
2025-09-03 22:15:07 [ERROR] [Data Integration Service] - Unable to submit data to SAP.
"""

# --- OpenAI Client ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("OpenAI API key not found. Please add it to your Streamlit secrets.", icon="🚨")
    st.stop()

# --- Session State Initialization ---
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "stage" not in st.session_state:
        st.session_state.stage = "app_selection"
    if "selected_app" not in st.session_state:
        st.session_state.selected_app = None
    if "log_summary" not in st.session_state:
        st.session_state.log_summary = None
    if "rca_report" not in st.session_state:
        st.session_state.rca_report = None
    if "first_run" not in st.session_state:
        st.session_state.first_run = True

init_session_state()

# --- AI & Voice Helper Functions ---
def get_ai_response(system_prompt, user_prompt, model="gpt-4o-mini"):
    """Generates a response from the OpenAI API."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}", icon="🚨")
        return "Sorry, I encountered an error. Please check the API key and configuration."

def text_to_speech(text):
    """Converts text to speech using OpenAI TTS and returns audio data."""
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
        )
        return response.content
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {e}", icon="🚨")
        return None

def speech_to_text(audio_bytes):
    """Transcribes audio to text using OpenAI Whisper."""
    try:
        # Whisper API expects a file-like object
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)
        with open("temp_audio.wav", "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        return transcript.text
    except Exception as e:
        st.error(f"Error in speech-to-text conversion: {e}", icon="🚨")
        return ""

# --- Agent Logic ---
def add_message(agent_name, text, play_audio=True):
    """Adds a message to the chat and optionally plays audio."""
    st.session_state.messages.append({"role": agent_name, "content": text})
    if play_audio:
        audio_data = text_to_speech(text)
        if audio_data:
            # Use a unique key for the audio player to avoid conflicts
            audio_key = f"audio_{len(st.session_state.messages)}"
            # Base64 encode the audio data to embed it in HTML for autoplay
            audio_b64 = base64.b64encode(audio_data).decode()
            audio_html = f"""
                <audio autoplay id="{audio_key}">
                    <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)


def agent_1_triage():
    """Initial agent to welcome the user."""
    system_prompt = "You are Agent 1, the incident triage manager. Welcome the user to the Major Incident bridge and ask them to specify which application is having issues by name from the CMDB list."
    user_prompt = "The user has just joined the call. Please provide a welcome message."
    with st.spinner("Agent 1 is preparing the bridge..."):
        response = get_ai_response(system_prompt, user_prompt)
        add_message("Agent 1", response)
    st.session_state.first_run = False

def agent_2_cmdb_lookup(app_name):
    """Finds the app in CMDB and visualizes dependencies."""
    app_ci = CMDB_DF[CMDB_DF['name'].str.lower() == app_name.lower()]
    if app_ci.empty:
        add_message("Agent 1", f"I'm sorry, I couldn't find '{app_name}' in our CMDB. Please select a valid application from the list on the right.")
        return

    st.session_state.selected_app = app_ci.iloc[0]
    st.session_state.stage = "bridge_joined"
    
    system_prompt = "You are Agent 2, a CMDB analyst. Confirm you've identified the application and its dependencies. Hand over to Agent 3 for log extraction. Inform the user they can now join the bridge call."
    user_prompt = f"The user has identified the application as '{app_name}'. Confirm this and explain the next step."
    with st.spinner("Agent 2 is analyzing CMDB..."):
        response = get_ai_response(system_prompt, user_prompt)
        add_message("Agent 2", response)
    st.rerun()

def agent_3_log_analysis():
    """Analyzes logs and prepares for RCA."""
    st.session_state.stage = "rca_generation"
    system_prompt = "You are Agent 3, a log analysis specialist. You've received the following logs. Briefly summarize the key errors (dependency timeout, connection failure, SSL handshake failed) and state you are passing this summary to Agent 4 for root cause analysis."
    user_prompt = f"Here are the logs:\n{SIMULATED_LOGS}"
    with st.spinner("Agent 3 is analyzing logs..."):
        response = get_ai_response(system_prompt, user_prompt)
        st.session_state.log_summary = response
        add_message("Agent 3", response)
    st.rerun()

def agent_4_rca_and_fix():
    """Generates the final report."""
    st.session_state.stage = "incident_resolved"
    system_prompt = "You are Agent 4, a Root Cause Analysis specialist. Based on the log summary, generate a final incident report. The root cause is an expired SSL certificate on the 'SAP-Salesforce Interface'. Your report must have three sections: 'Root Cause Analysis', 'Recommended Fix', and 'Preventative Measures'. Be clear and concise."
    user_prompt = f"Log summary: {st.session_state.log_summary}"
    with st.spinner("Agent 4 is performing RCA..."):
        response = get_ai_response(system_prompt, user_prompt)
        st.session_state.rca_report = response
        add_message("Agent 4", "I have completed the analysis and generated the final report, which you can see on the right. This incident bridge can now be closed.")
    st.rerun()

def agent_5_qa(query):
    """Answers general questions based on available context."""
    context = f"""
    CMDB Data: {CMDB_DF.to_string()}
    Simulated Logs: {SIMULATED_LOGS if st.session_state.log_summary else "Not available yet."}
    Log Summary: {st.session_state.log_summary if st.session_state.log_summary else "Not available yet."}
    RCA Report: {st.session_state.rca_report if st.session_state.rca_report else "Not available yet."}
    """
    system_prompt = "You are Agent 5, a helpful Q&A assistant. Answer the user's question based ONLY on the provided context. If the information is not in the context, say that you cannot answer that question at this time."
    with st.spinner("Agent 5 is checking the records..."):
        response = get_ai_response(system_prompt, f"Context:\n{context}\n\nUser Question: {query}")
        add_message("Agent 5", response)


# --- UI Drawing Functions ---
def draw_knowledge_graph():
    """Displays the dependency graph for the selected application."""
    if st.session_state.selected_app is not None:
        st.subheader("Knowledge Graph: Associated CIs")
        app_info = st.session_state.selected_app
        dot = graphviz.Digraph(comment=f'Dependencies for {app_info["name"]}')
        dot.node(app_info['id'], app_info['name'], shape='ellipse', style='filled', fillcolor='skyblue')

        associated_ids = app_info['associated_cis']
        associated_cis = CMDB_DF[CMDB_DF['id'].isin(associated_ids)]

        for _, ci in associated_cis.iterrows():
            dot.node(ci['id'], ci['name'], shape='box', style='filled', fillcolor='lightgray')
            dot.edge(app_info['id'], ci['id'])
        
        st.graphviz_chart(dot)

def draw_data_panel():
    """Renders the right-hand panel with contextual data."""
    with st.container(border=True):
        if st.session_state.stage == "app_selection":
            st.subheader("CMDB: Applications")
            st.dataframe(CMDB_DF[CMDB_DF['type'] == 'Application'][['id', 'name']], use_container_width=True)
        
        if st.session_state.selected_app is not None:
            draw_knowledge_graph()

        if st.session_state.log_summary:
            with st.expander("Log Analysis by Agent 3", expanded=True):
                st.code(SIMULATED_LOGS, language="log")
                st.info(st.session_state.log_summary)

        if st.session_state.rca_report:
            st.subheader("Final Incident Report by Agent 4")
            st.markdown(st.session_state.rca_report)


# --- Main App Layout ---
st.title("🗣️ AI Major Incident Manager")

# Main columns
col1, col2 = st.columns([2, 1])

with col1:
    # Chat container
    with st.container(height=600):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

with col2:
    # Data and controls panel
    draw_data_panel()

    # Dynamic control buttons based on stage
    if st.session_state.stage == "bridge_joined":
        if st.button("▶️ Run Log Analysis", type="primary"):
            agent_3_log_analysis()

    if st.session_state.stage == "rca_generation":
        if st.button("🔎 Generate RCA & Fix", type="primary"):
            agent_4_rca_and_fix()

    if st.session_state.stage == "incident_resolved":
        st.success("Incident Resolved. You can restart the process by refreshing the page.")
        
# --- Input Handling ---
# Voice recorder UI
if st.session_state.stage != "incident_resolved":
    st.write("---")
    st.markdown("🎤 **Voice Input**")
    audio_bytes = audiorecorder("Click to record", "Recording...")
    if audio_bytes:
        with st.spinner("Transcribing your voice..."):
            user_input = speech_to_text(audio_bytes)
            # This will trigger processing in the chat_input handler logic below
            st.session_state.user_input_from_voice = user_input
            st.rerun()


# Text input handling (unified for text and voice)
prompt = st.chat_input("Or type your response here...", disabled=(st.session_state.stage == "incident_resolved"))

# Check for voice input first
if "user_input_from_voice" in st.session_state and st.session_state.user_input_from_voice:
    prompt = st.session_state.user_input_from_voice
    # Clear it so it doesn't re-run
    st.session_state.user_input_from_voice = ""

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent logic based on conversation stage
    if st.session_state.stage == "app_selection":
        agent_2_cmdb_lookup(prompt)
    else:
        # If not in the initial stage, it could be a Q&A
        agent_5_qa(prompt)
        st.rerun()

# Initial welcome message on first run
if st.session_state.first_run:
    agent_1_triage()
    st.rerun()
