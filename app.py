# app.py
import streamlit as st
import openai
import speech_recognition as sr
import threading
import time
from datetime import datetime
import re
import os
from dotenv import load_dotenv
import base64
import tempfile
import queue
import json

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Major Incident Manager AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #B0BEC5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        background-color: #1E1E1E;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid #424242;
    }
    .stButton button {
        width: 100%;
        background-color: #1565C0;
        color: white;
    }
    .stButton button:hover {
        background-color: #0D47A1;
        color: white;
    }
    .analysis-box {
        background-color: #1E1E1E;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-top: 1.5rem;
        border: 1px solid #424242;
    }
    .log-area {
        background-color: #0D0D0D;
        color: #E0E0E0;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .conversation-user {
        background-color: #37474F;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: right;
    }
    .conversation-ai {
        background-color: #0D47A1;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .pulse-animation {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
</style>
""", unsafe_allow_html=True)

# JavaScript for browser TTS
tts_js = """
<script>
function speakText(text) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(utterance);
        return true;
    }
    return false;
}
</script>
"""

class IncidentManagerAI:
    def __init__(self, openai_api_key):
        # Initialize OpenAI API
        openai.api_key = openai_api_key
        self.model = "gpt-4o-mini"
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = None
        
        # Try to initialize microphone (may not work in all environments)
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
        except:
            st.warning("Microphone not available. Voice input will be disabled.")
            
        # Conversation state
        self.conversation_active = False
        self.logs_provided = False
        self.logs = ""
        self.conversation_history = []
        
        # System prompt for the AI agent
        self.system_prompt = """You are an expert "Analyzer Agent" for IT incidents. Your name is IncidentBot. 
You are having a conversational interaction with a Subject Matter Expert (SME) to help them analyze and resolve a major incident.

Your task is to:
1. Greet the SME professionally and explain your role
2. Guide them through providing application, web, or database logs
3. Analyze the logs to identify the root cause
4. Provide actionable fixes
5. Suggest preventative measures

Be conversational, empathetic, and professional. Ask clarifying questions if needed. 
Structure your analysis with:
- Root Cause Analysis (RCA)
- Proposed Fixes
- Preventative SOPs

Speak clearly and concisely. Remember you're in a voice conversation."""

    def speak(self, text):
        """Use browser TTS via JavaScript"""
        # Inject JavaScript for TTS
        js_code = f"""
        <script>
            if ('speechSynthesis' in window) {{
                const utterance = new SpeechSynthesisUtterance("{text.replace('"', '\\"')}");
                window.speechSynthesis.speak(utterance);
            }}
        </script>
        """
        st.components.v1.html(js_code, height=0)
        
    def listen(self, timeout=10, phrase_time_limit=5):
        """Listen for speech and convert to text"""
        if not self.microphone:
            st.session_state.status = "Microphone not available"
            return ""
            
        try:
            with self.microphone as source:
                st.session_state.status = "Listening..."
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                text = self.recognizer.recognize_google(audio)
                st.session_state.status = f"Heard: {text}"
                return text.lower()
        except sr.WaitTimeoutError:
            st.session_state.status = "Listening timeout"
            return ""
        except sr.UnknownValueError:
            st.session_state.status = "Could not understand audio"
            return ""
        except sr.RequestError as e:
            st.session_state.status = f"Speech recognition error: {e}"
            return ""
        except Exception as e:
            st.session_state.status = f"Error: {e}"
            return ""
    
    def get_ai_response(self, user_input):
        """Get response from OpenAI GPT model"""
        # Prepare messages with conversation history
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history,
            {"role": "user", "content": user_input}
        ]
        
        try:
            # Use the updated OpenAI API
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            ai_response = response.choices[0].message.content
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            return ai_response
        except Exception as e:
            return f"I apologize, but I'm experiencing technical difficulties: {str(e)}"

    def start_conversation(self):
        """Start the incident management conversation"""
        self.conversation_active = True
        self.conversation_history = []
        
        # Initial greeting
        greeting = """Welcome to the Major Incident bridge. I am your AI Incident Manager, IncidentBot. 
I'm here to help you analyze and resolve this major incident. 
Please describe the issue or provide any application, web, or database logs you have available."""
        
        st.session_state.conversation.append({"role": "AI", "message": greeting})
        self.speak(greeting)
        
    def process_user_input(self, user_input):
        """Process user input and generate AI response"""
        if not user_input:
            return
            
        # Check for exit command
        if any(word in user_input for word in ["exit", "end", "stop", "quit", "goodbye"]):
            closing = "Thank you for using the Major Incident Manager. The incident call is now concluding."
            st.session_state.conversation.append({"role": "AI", "message": closing})
            self.speak(closing)
            self.conversation_active = False
            st.session_state.conversation_active = False
            return
            
        # Check if user is providing logs
        if not self.logs_provided and ("log" in user_input or "error" in user_input or "issue" in user_input):
            self.logs_provided = True
            
        # Get AI response
        ai_response = self.get_ai_response(user_input)
        
        # Add to conversation
        st.session_state.conversation.append({"role": "AI", "message": ai_response})
        
        # Speak the response
        self.speak(ai_response)
        
        # Check if analysis is complete
        if "root cause" in ai_response.lower() and ("fix" in ai_response.lower() or "solution" in ai_response.lower()):
            st.session_state.analysis_complete = True

    def extract_main_points(self, analysis_text):
        """Extract the main points from analysis for speech"""
        lines = analysis_text.split('\n')
        main_points = []
        
        for line in lines:
            if line.strip() and not line.startswith('#') and not line.startswith('*') and not line.startswith('-'):
                main_points.append(line.strip())
                if len(main_points) >= 3:
                    break
        
        if main_points:
            return "Based on my analysis: " + ". ".join(main_points)
        else:
            return "I've completed my analysis. The detailed findings are available in the text output."

def main():
    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'status' not in st.session_state:
        st.session_state.status = "Ready to start"
    if 'conversation_active' not in st.session_state:
        st.session_state.conversation_active = False
    if 'logs_provided' not in st.session_state:
        st.session_state.logs_provided = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'incident_ai' not in st.session_state:
        # Get API key from environment or user input
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = st.secrets.get("OPENAI_API_KEY", "")
        
        if api_key:
            st.session_state.incident_ai = IncidentManagerAI(api_key)
        else:
            st.session_state.incident_ai = None

    # Header
    st.markdown('<h1 class="main-header">Major Incident Manager AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Voice-driven incident analysis and resolution</p>', unsafe_allow_html=True)
    
    # Status indicator
    with st.container():
        status_color = "#9E9E9E"  # Default gray
        if "Listening" in st.session_state.status:
            status_color = "#FFC107"  # Yellow
        elif "Heard" in st.session_state.status:
            status_color = "#4CAF50"  # Green
        elif "Error" in st.session_state.status:
            status_color = "#F44336"  # Red
            
        st.markdown(f"""
        <div class="status-box">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div class="pulse-animation" style="width: 12px; height: 12px; background-color: {status_color}; border-radius: 50%;"></div>
                <span style="font-size: 1.1rem;">{st.session_state.status}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # API key input (if not set)
        if st.session_state.incident_ai is None:
            st.info("Please enter your OpenAI API key to continue")
            api_key = st.text_input("OpenAI API Key", type="password")
            if api_key:
                st.session_state.incident_ai = IncidentManagerAI(api_key)
                st.rerun()
            st.stop()
        
        # Start conversation button
        if not st.session_state.conversation_active:
            if st.button("🎤 Start Incident Call", use_container_width=True):
                st.session_state.conversation_active = True
                st.session_state.incident_ai.start_conversation()
                st.rerun()
        
        # Log input area
        if st.session_state.conversation_active and not st.session_state.logs_provided:
            st.subheader("Provide Incident Details")
            logs = st.text_area("Paste application, web, or database logs here:", height=200, 
                               placeholder="Paste logs here...")
            
            if st.button("Submit Logs", use_container_width=True):
                if logs:
                    st.session_state.incident_ai.logs = logs
                    st.session_state.logs_provided = True
                    st.session_state.incident_ai.logs_provided = True
                    st.session_state.conversation.append({"role": "User", "message": "I've provided the logs for analysis."})
                    st.session_state.incident_ai.process_user_input("I've provided the logs for analysis.")
                    st.rerun()
                else:
                    st.warning("Please provide logs before submitting.")
        
        # Voice input button
        if st.session_state.conversation_active and st.session_state.incident_ai.microphone:
            if st.button("🎤 Use Voice Input", use_container_width=True):
                user_input = st.session_state.incident_ai.listen()
                if user_input:
                    st.session_state.conversation.append({"role": "User", "message": user_input})
                    st.session_state.incident_ai.process_user_input(user_input)
                    st.rerun()
        
        # Text input
        if st.session_state.conversation_active:
            user_input = st.text_input("Type your message:", key="user_input")
            if st.button("Send Message", use_container_width=True) and user_input:
                st.session_state.conversation.append({"role": "User", "message": user_input})
                st.session_state.incident_ai.process_user_input(user_input)
                st.rerun()
    
    with col2:
        # Conversation display
        if st.session_state.conversation:
            st.subheader("Conversation")
            conversation_container = st.container()
            with conversation_container:
                for i, msg in enumerate(st.session_state.conversation):
                    if msg["role"] == "AI":
                        st.markdown(f"""
                        <div class="conversation-ai">
                            <b>IncidentBot:</b> {msg["message"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="conversation-user">
                            <b>You:</b> {msg["message"]}
                        </div>
                        """, unsafe_allow_html=True)
        
        # Analysis results
        if st.session_state.analysis_complete:
            st.subheader("Analysis Results")
            st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
            
            # Extract analysis from conversation
            analysis_text = ""
            for msg in reversed(st.session_state.conversation):
                if msg["role"] == "AI" and ("root cause" in msg["message"].lower() or 
                                          "fix" in msg["message"].lower() or 
                                          "solution" in msg["message"].lower()):
                    analysis_text = msg["message"]
                    break
            
            if analysis_text:
                # Display formatted analysis
                if "Root Cause Analysis" in analysis_text or "RCA" in analysis_text:
                    st.markdown("#### Root Cause Analysis (RCA)")
                    rca_text = extract_section(analysis_text, "Root Cause Analysis", "Proposed Fixes")
                    st.info(rca_text if rca_text != "Not available" else analysis_text)
                
                if "Proposed Fixes" in analysis_text or "Fixes" in analysis_text:
                    st.markdown("#### Proposed Fixes")
                    fixes_text = extract_section(analysis_text, "Proposed Fixes", "Preventative")
                    st.success(fixes_text if fixes_text != "Not available" else analysis_text)
                
                if "Preventative" in analysis_text or "SOPs" in analysis_text:
                    st.markdown("#### Preventative SOPs")
                    preventative_text = extract_section(analysis_text, "Preventative", None)
                    st.warning(preventative_text if preventative_text != "Not available" else analysis_text)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #B0BEC5;'>Powered by OpenAI GPT-4o mini and Streamlit</p>", 
                unsafe_allow_html=True)
    
    # Add TTS JavaScript
    st.components.v1.html(tts_js, height=0)

def extract_section(text, start_label, end_label):
    """Extract a section from analysis text between labels"""
    start_idx = text.find(start_label)
    if start_idx == -1:
        return "Not available"
    
    if end_label:
        end_idx = text.find(end_label, start_idx)
        if end_idx == -1:
            return text[start_idx + len(start_label):].strip()
        return text[start_idx + len(start_label):end_idx].strip()
    
    return text[start_idx + len(start_label):].strip()

if __name__ == "__main__":
    main()
