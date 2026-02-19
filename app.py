"""
Streamlit Frontend for Smart Textile Quality Inspector
Connects to FastAPI backend for defect detection and AI chatbot
"""

import streamlit as st
import requests
from PIL import Image
import io
import json
from datetime import datetime
import base64

# ==================== CONFIGURATION ====================

API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Smart Textile Quality Inspector",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
    .status-pass {
        color: #4caf50;
        font-weight: bold;
    }
    .status-reject {
        color: #f44336;
        font-weight: bold;
    }
    .status-review {
        color: #ff9800;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================

if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# ==================== HELPER FUNCTIONS ====================

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=3)
        return response.status_code == 200
    except:
        return False

def upload_image(file):
    """Upload image to API"""
    try:
        files = {'file': file}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error uploading image: {str(e)}")
        return None

def run_detection(session_id):
    """Run defect detection"""
    try:
        response = requests.post(f"{API_BASE_URL}/detect", params={'session_id': session_id})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Detection failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error running detection: {str(e)}")
        return None

def send_chat_message(session_id, message):
    """Send message to chatbot"""
    try:
        payload = {
            "session_id": session_id,
            "message": message
        }
        response = requests.post(f"{API_BASE_URL}/chat", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Chat failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error sending message: {str(e)}")
        return None

def get_annotated_image(session_id):
    """Get annotated image from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/image/{session_id}")
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        return None
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def get_status_class(decision):
    """Get CSS class for decision status"""
    if decision == "PASS":
        return "status-pass"
    elif decision == "REJECT":
        return "status-reject"
    else:
        return "status-review"

def format_confidence(confidence):
    """Format confidence percentage"""
    return f"{confidence * 100:.1f}%"

# ==================== SIDEBAR ====================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/textile.png", width=80)
    st.title("🧵 Textile Inspector")
    
    # API Status
    st.subheader("🔌 System Status")
    if check_api_health():
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")
        st.warning("Please start the FastAPI server:\n```python main.py```")
    
    st.divider()
    
    # Session Info
    st.subheader("📊 Session Info")
    if st.session_state.session_id:
        st.info(f"**Session ID:**\n`{st.session_state.session_id[:8]}...`")
        
        if st.button("🗑️ New Inspection", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.detection_results = None
            st.session_state.chat_history = []
            st.session_state.uploaded_file = None
            st.rerun()
    else:
        st.info("No active session")
    
    st.divider()
    
    # Instructions
    st.subheader("📋 Instructions")
    st.markdown("""
    1. **Upload** fabric image
    2. **Run** defect detection
    3. **Chat** with AI for analysis
    4. **Review** recommendations
    """)
    
    st.divider()
    
    # About
    st.subheader("ℹ️ About")
    st.markdown("""
    **Smart Textile Quality Inspector**
    
    AI-powered defect detection using:
    - YOLOv8 Classification
    - YOLOv8 Detection
    - Gemini 2.0 Flash AI
    
    Detects 5 defect types:
    - Horizontal
    - Vertical
    - Hole
    - Lines
    - Hole_variant
    """)

# ==================== MAIN CONTENT ====================

# Header
st.markdown('<div class="main-header">🧵 Smart Textile Quality Inspector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Fabric Defect Detection & Analysis</div>', unsafe_allow_html=True)

# Check API health
if not check_api_health():
    st.error("⚠️ Cannot connect to API. Please ensure the FastAPI server is running on http://localhost:8000")
    st.code("python main.py", language="bash")
    st.stop()

# ==================== STEP 1: IMAGE UPLOAD ====================

st.header("📤 Step 1: Upload Fabric Image")

uploaded_file = st.file_uploader(
    "Choose a fabric image...",
    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
    help="Upload a clear image of the fabric sample for inspection"
)

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    # Upload to API if new file
    if st.session_state.uploaded_file != uploaded_file.name:
        with st.spinner("Uploading image..."):
            uploaded_file.seek(0)  # Reset file pointer
            result = upload_image(uploaded_file)
            
            if result:
                st.session_state.session_id = result['session_id']
                st.session_state.uploaded_file = uploaded_file.name
                st.session_state.detection_results = None
                st.session_state.chat_history = []
                st.success(f"✅ Image uploaded successfully! Session: {result['session_id'][:8]}...")
    
    # ==================== STEP 2: RUN DETECTION ====================
    
    if st.session_state.session_id:
        st.header("🔍 Step 2: Run Defect Detection")
        
        if st.button("🚀 Run Detection", type="primary", use_container_width=True):
            with st.spinner("Running AI detection models..."):
                results = run_detection(st.session_state.session_id)
                
                if results:
                    st.session_state.detection_results = results
                    st.success("✅ Detection completed!")
                    st.rerun()
        
        # ==================== DISPLAY RESULTS ====================
        
        if st.session_state.detection_results:
            results = st.session_state.detection_results
            
            st.header("📊 Detection Results")
            
            # Get annotated image
            with col2:
                st.subheader("🎯 Detected Defects")
                annotated_img = get_annotated_image(st.session_state.session_id)
                if annotated_img:
                    st.image(annotated_img, use_column_width=True)
            
            st.divider()
            
            # Metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    "Defect Type",
                    results['classification']['defect_type']
                )
            
            with metric_col2:
                st.metric(
                    "Confidence",
                    format_confidence(results['classification']['confidence'])
                )
            
            with metric_col3:
                st.metric(
                    "Defects Found",
                    results['detection']['num_detections']
                )
            
            with metric_col4:
                decision = results['quality_assessment']['decision']
                st.markdown(f"**Decision**")
                st.markdown(f'<p class="{get_status_class(decision)}" style="font-size: 1.5rem; margin: 0;">{decision}</p>', unsafe_allow_html=True)
            
            st.divider()
            
            # Detailed Results
            tab1, tab2, tab3 = st.tabs(["📋 Classification", "🎯 Detection Details", "⚖️ Quality Assessment"])
            
            with tab1:
                st.subheader("Classification Results")
                
                cls_data = results['classification']
                st.markdown(f"**Primary Defect:** {cls_data['defect_type']}")
                st.markdown(f"**Confidence:** {format_confidence(cls_data['confidence'])}")
                
                st.markdown("**All Class Probabilities:**")
                probs = cls_data.get('all_probabilities', {})
                for defect_class, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    st.progress(prob, text=f"{defect_class}: {format_confidence(prob)}")
            
            with tab2:
                st.subheader("Detection Details")
                
                det_data = results['detection']
                st.markdown(f"**Total Detections:** {det_data['num_detections']}")
                
                if det_data['detections']:
                    for i, det in enumerate(det_data['detections'], 1):
                        with st.expander(f"Defect #{i}: {det['class']} ({format_confidence(det['confidence'])})"):
                            st.markdown(f"**Class:** {det['class']}")
                            st.markdown(f"**Confidence:** {format_confidence(det['confidence'])}")
                            st.markdown(f"**Bounding Box:** {[round(x, 1) for x in det['bbox']]}")
                else:
                    st.info("No defects detected with bounding boxes")
            
            with tab3:
                st.subheader("Quality Assessment")
                
                qa_data = results['quality_assessment']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Decision:** `{qa_data['decision']}`")
                    st.markdown(f"**Grade:** `{qa_data['grade']}`")
                    st.markdown(f"**Severity:** `{qa_data['severity']}`")
                
                with col2:
                    st.markdown(f"**Quality Score:** `{qa_data['quality_score']}/100`")
                    st.markdown(f"**Defect Count:** `{qa_data['num_defects']}`")
                    st.markdown(f"**Confidence:** `{format_confidence(qa_data['confidence'])}`")
                
                # Quality bar
                st.markdown("**Overall Quality Score:**")
                color = "green" if qa_data['quality_score'] >= 80 else "orange" if qa_data['quality_score'] >= 60 else "red"
                st.progress(qa_data['quality_score']/100)
            
            # ==================== STEP 3: AI CHATBOT ====================
            
            st.header("💬 Step 3: Chat with AI Expert")
            
            # Chat container
            st.markdown("### Conversation")
            
            # Display chat history
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f'<div class="chat-message user-message">👤 **You:** {msg["message"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message">🤖 **AI Expert:** {msg["message"]}</div>', unsafe_allow_html=True)
            
            # Quick action buttons
            st.markdown("**Quick Questions:**")
            quick_col1, quick_col2, quick_col3 = st.columns(3)
            
            with quick_col1:
                if st.button("❓ What caused this?", use_container_width=True):
                    msg = "What are the likely root causes of this defect?"
                    response = send_chat_message(st.session_state.session_id, msg)
                    if response:
                        st.session_state.chat_history.append({"role": "user", "message": msg})
                        st.session_state.chat_history.append({"role": "assistant", "message": response['response']})
                        st.rerun()
            
            with quick_col2:
                if st.button("🔧 How to fix?", use_container_width=True):
                    msg = "What immediate corrective actions should we take?"
                    response = send_chat_message(st.session_state.session_id, msg)
                    if response:
                        st.session_state.chat_history.append({"role": "user", "message": msg})
                        st.session_state.chat_history.append({"role": "assistant", "message": response['response']})
                        st.rerun()
            
            with quick_col3:
                if st.button("📊 Generate report", use_container_width=True):
                    msg = "Generate a comprehensive quality inspection report with all findings and recommendations"
                    response = send_chat_message(st.session_state.session_id, msg)
                    if response:
                        st.session_state.chat_history.append({"role": "user", "message": msg})
                        st.session_state.chat_history.append({"role": "assistant", "message": response['response']})
                        st.rerun()
            
            # Chat input
            with st.form(key='chat_form', clear_on_submit=True):
                user_input = st.text_input(
                    "Ask the AI expert anything about this defect...",
                    placeholder="E.g., What preventive measures should we implement?",
                    label_visibility="collapsed"
                )
                submit = st.form_submit_button("Send 💬", use_container_width=True)
                
                if submit and user_input:
                    with st.spinner("AI is analyzing..."):
                        response = send_chat_message(st.session_state.session_id, user_input)
                        
                        if response:
                            st.session_state.chat_history.append({"role": "user", "message": user_input})
                            st.session_state.chat_history.append({"role": "assistant", "message": response['response']})
                            st.rerun()

else:
    st.info("👆 Please upload a fabric image to begin inspection")
    
    # Sample instructions
    st.markdown("---")
    st.subheader("🎯 How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1️⃣ Upload")
        st.markdown("Upload a clear image of your fabric sample")
    
    with col2:
        st.markdown("### 2️⃣ Detect")
        st.markdown("AI analyzes and identifies defects automatically")
    
    with col3:
        st.markdown("### 3️⃣ Chat")
        st.markdown("Get expert insights and recommendations")

# ==================== FOOTER ====================

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666;">Smart Textile Quality Inspector v1.0 | Powered by YOLOv8 & Gemini 2.0 Flash</div>',
    unsafe_allow_html=True
)