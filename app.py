import streamlit as st
import pandas as pd
import graphviz
from openai import OpenAI
import base64
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from pydub import AudioSegment
import io
import av

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
</style>
""", unsafe_allow_html=True)

# --- Data Simulation (from original JS) ---
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

# --- OpenAI Client & Voice Handling ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("OpenAI API key not found. Please add it to your Streamlit secrets.", icon="🚨")
    st.stop()
    
# Initialize audio buffer
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = queue.Queue()

class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert audio frame to pydub AudioSegment
        sound = AudioSegment(
            data=frame.to_ndarray().tobytes(),
            sample_width=frame.format.bytes,
            frame_rate=frame.sample_rate,
            channels=len(frame.layout.channels),
        )
        st.session_state.audio_buffer.put(sound)
        return frame

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
    if "user_input" not in st.session_state:
        st.session_state.user_input = None
    if "transcribe_clicked" not in st.session_state:
        st.session_state.transcribe_clicked = False

init_session_state()

# --- AI & Helper Functions ---
def get_ai_response(system_prompt, user_prompt, model="gpt-4o-mini"):
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
        return "Sorry, I encountered an error."

def text_to_speech(text):
    try:
        response = client.audio.speech.create(model="tts-1", voice="alloy", input=text)
        return response.content
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {e}", icon="🚨")
        return None

def speech_to_text(audio_bytes):
    try:
        with io.BytesIO() as buffer:
            # Write audio bytes to buffer
            buffer.write(audio_bytes)
            buffer.seek(0)
            # Create a temporary file in memory
            audio_file = io.BytesIO(buffer.read())
            audio_file.name = "audio.wav"
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        st.error(f"Error in speech-to-text conversion: {e}", icon="🚨")
        return ""

def add_message(agent_name, text, play_audio=True):
    st.session_state.messages.append({"role": agent_name, "content": text})
    if play_audio:
        audio_data = text_to_speech(text)
        if audio_data:
            audio_b64 = base64.b64encode(audio_data).decode()
            audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'
            st.markdown(audio_html, unsafe_allow_html=True)

# --- Agent Logic ---
def agent_1_triage():
    system_prompt = "You are Agent 1, the incident triage manager. Welcome the user to the Major Incident bridge and ask them to specify which application is having issues by name from the CMDB list."
    user_prompt = "The user has just joined the call. Please provide a welcome message."
    with st.spinner("Agent 1 is preparing the bridge..."):
        response = get_ai_response(system_prompt, user_prompt)
        add_message("Agent 1", response)
    st.session_state.first_run = False

def agent_2_cmdb_lookup(app_name):
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

def agent_3_log_analysis():
    st.session_state.stage = "rca_generation"
    system_prompt = "You are Agent 3, a log analysis specialist. You've received the following logs. Briefly summarize the key errors and state you are passing this summary to Agent 4 for root cause analysis."
    user_prompt = f"Here are the logs:\n{SIMULATED_LOGS}"
    with st.spinner("Agent 3 is analyzing logs..."):
        response = get_ai_response(system_prompt, user_prompt)
        st.session_state.log_summary = response
        add_message("Agent 3", response)

def agent_4_rca_and_fix():
    st.session_state.stage = "incident_resolved"
    system_prompt = "You are Agent 4, a Root Cause Analysis specialist. Based on the log summary, generate a final incident report. The root cause is an expired SSL certificate on the 'SAP-Salesforce Interface'. Your report must have three sections: 'Root Cause Analysis', 'Recommended Fix', and 'Preventative Measures'."
    user_prompt = f"Log summary: {st.session_state.log_summary}"
    with st.spinner("Agent 4 is performing RCA..."):
        response = get_ai_response(system_prompt, user_prompt)
        st.session_state.rca_report = response
        add_message("Agent 4", "I have completed the analysis and generated the final report. This incident bridge can now be closed.")

def agent_5_qa(query):
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
    if st.session_state.selected_app is not None:
        st.subheader("Knowledge Graph: Associated CIs")
        app_info = st.session_state.selected_app
        dot = graphviz.Digraph()
        dot.node(app_info['id'], app_info['name'], shape='ellipse', style='filled', fillcolor='skyblue')
        associated_ids = app_info['associated_cis']
        associated_cis = CMDB_DF[CMDB_DF['id'].isin(associated_ids)]
        for _, ci in associated_cis.iterrows():
            dot.node(ci['id'], ci['name'], shape='box', style='filled', fillcolor='lightgray')
            dot.edge(app_info['id'], ci['id'])
        st.graphviz_chart(dot)

def draw_data_panel():
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

# --- Main App & Input Handling ---
def process_user_input(prompt):
    if not prompt:
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    if st.session_state.stage == "app_selection":
        agent_2_cmdb_lookup(prompt)
    else:
        agent_5_qa(prompt)

st.title("🗣️ AI Major Incident Manager")
col1, col2 = st.columns([2, 1])

with col1:
    with st.container(height=600):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

with col2:
    draw_data_panel()
    if st.session_state.stage == "bridge_joined":
        if st.button("▶️ Run Log Analysis", type="primary"):
            agent_3_log_analysis()
    if st.session_state.stage == "rca_generation":
        if st.button("🔎 Generate RCA & Fix", type="primary"):
            agent_4_rca_and_fix()
    if st.session_state.stage == "incident_resolved":
        st.success("Incident Resolved.")

# Initial welcome message
if st.session_state.first_run:
    agent_1_triage()

# --- Voice and Text Input Section ---
if st.session_state.stage != "incident_resolved":
    st.write("---")
    st.markdown("🎤 **Voice Input**")
    
    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    if st.button("Transcribe Last Spoken Words"):
        st.session_state.transcribe_clicked = True
        
    if st.session_state.transcribe_clicked:
        if not st.session_state.audio_buffer.empty():
            combined_audio = AudioSegment.empty()
            while not st.session_state.audio_buffer.empty():
                try:
                    combined_audio += st.session_state.audio_buffer.get_nowait()
                except queue.Empty:
                    break
            
            if len(combined_audio) > 0:
                buffer = io.BytesIO()
                combined_audio.export(buffer, format="wav")
                buffer.seek(0)
                with st.spinner("Transcribing your voice..."):
                    st.session_state.user_input = speech_to_text(buffer.getvalue())
                    st.session_state.transcribe_clicked = False
        else:
            st.warning("Audio buffer is empty. Speak into the microphone first.")
            st.session_state.transcribe_clicked = False

    # Process text from either voice transcription or text input
    text_prompt = st.chat_input("Or type your response here...")
    
    final_prompt = None
    if text_prompt:
        final_prompt = text_prompt
    elif st.session_state.get('user_input'):
        final_prompt = st.session_state.user_input
        st.session_state.user_input = None  # Clear after processing

    if final_prompt:
        process_user_input(final_prompt)
        st.rerun()
