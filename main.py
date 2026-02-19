"""
FastAPI Backend for Smart Textile Quality Inspector
Integrates YOLOv8 Detection + Classification with Gemini 2.0 Flash Chatbot
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import uuid
import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import json
from datetime import datetime

# YOLO imports
from ultralytics import YOLO

# Gemini imports
import google.generativeai as genai

# ==================== CONFIGURATION ====================

# Directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Model paths
CLASSIFICATION_MODEL_PATH = "runs/classify/fabric_defect_cls2/weights/best.pt"
DETECTION_MODEL_PATH = "runs/detect/fabric_defect_det/weights/best.pt"

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyCj5jhcoFTeYPcnAZa4-rwBj8VpstDzwNoaass"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize FastAPI
app = FastAPI(
    title="Smart Textile Quality Inspector API",
    description="AI-Powered Textile Defect Detection with Gen AI Chatbot",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== IN-MEMORY SESSION STORAGE ====================

# Session storage: {session_id: session_data}
sessions: Dict[str, Dict[str, Any]] = {}

# ==================== MODELS LOADING ====================

print("Loading YOLO models...")
try:
    classification_model = YOLO(CLASSIFICATION_MODEL_PATH)
    detection_model = YOLO(DETECTION_MODEL_PATH)
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    classification_model = None
    detection_model = None

# Gemini model initialization
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

# ==================== PYDANTIC MODELS ====================

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    timestamp: str

class DetectionResult(BaseModel):
    session_id: str
    classification: Dict[str, Any]
    detection: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    annotated_image_path: str

# ==================== HELPER FUNCTIONS ====================

def generate_session_id() -> str:
    """Generate unique session ID"""
    return str(uuid.uuid4())

def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 string"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def save_uploaded_file(upload_file: UploadFile, session_id: str) -> str:
    """Save uploaded file and return path"""
    file_extension = Path(upload_file.filename).suffix
    file_path = UPLOAD_DIR / f"{session_id}_original{file_extension}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return str(file_path)

def draw_bounding_boxes(image_path: str, detections: List[Dict], output_path: str) -> str:
    """Draw bounding boxes on image and save"""
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Define colors for different defect types
    colors = {
        "Horizontal": (0, 255, 0),      # Green
        "Vertical": (255, 0, 0),        # Blue
        "Hole": (0, 0, 255),            # Red
        "Lines": (255, 255, 0),         # Cyan
        "Hole_variant": (255, 0, 255),  # Magenta
    }
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        defect_class = det['class']
        confidence = det['confidence']
        
        color = colors.get(defect_class, (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Add label
        label = f"{defect_class}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (int(x1), int(y1) - label_size[1] - 10), 
                     (int(x1) + label_size[0], int(y1)), color, -1)
        cv2.putText(image, label, (int(x1), int(y1) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, image)
    return output_path

def process_classification_result(cls_result) -> Dict[str, Any]:
    """Extract classification model results"""
    if cls_result is None or len(cls_result) == 0:
        return {"error": "No classification result"}
    
    result = cls_result[0]
    probs = result.probs
    
    return {
        "defect_type": result.names[probs.top1],
        "confidence": float(probs.top1conf),
        "all_probabilities": {
            result.names[i]: float(probs.data[i]) 
            for i in range(len(probs.data))
        }
    }

def process_detection_result(det_result) -> Dict[str, Any]:
    """Extract detection model results"""
    if det_result is None or len(det_result) == 0:
        return {"error": "No detection result", "detections": []}
    
    result = det_result[0]
    detections = []
    
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            
            detections.append({
                "bbox": [float(x) for x in bbox],
                "confidence": conf,
                "class": result.names[cls],
                "class_id": cls
            })
    
    return {
        "num_detections": len(detections),
        "detections": detections
    }

def calculate_quality_assessment(classification: Dict, detection: Dict) -> Dict[str, Any]:
    """Calculate overall quality assessment"""
    num_defects = detection.get("num_detections", 0)
    cls_confidence = classification.get("confidence", 0)
    defect_type = classification.get("defect_type", "Unknown")
    
    # Severity mapping
    severity_map = {
        "Hole": "Critical",
        "Hole_variant": "Critical",
        "Horizontal": "Major",
        "Vertical": "Major",
        "Lines": "Minor"
    }
    
    severity = severity_map.get(defect_type, "Unknown")
    
    # Decision logic
    if num_defects == 0:
        decision = "PASS"
        grade = "A"
    elif severity == "Critical":
        decision = "REJECT"
        grade = "C"
    elif severity == "Major" and num_defects > 2:
        decision = "REJECT"
        grade = "C"
    elif severity == "Major":
        decision = "REVIEW"
        grade = "B"
    else:
        decision = "REVIEW"
        grade = "B"
    
    # Calculate quality score (0-100)
    base_score = 100
    deduction = num_defects * 10
    if severity == "Critical":
        deduction *= 2
    
    quality_score = max(0, base_score - deduction)
    
    return {
        "decision": decision,
        "grade": grade,
        "quality_score": quality_score,
        "severity": severity,
        "num_defects": num_defects,
        "confidence": cls_confidence
    }

def build_gemini_prompt(session_data: Dict, user_query: str) -> str:
    """Build comprehensive prompt for Gemini"""
    cls_result = session_data.get("classification", {})
    det_result = session_data.get("detection", {})
    quality = session_data.get("quality_assessment", {})
    
    prompt = f"""You are an expert textile manufacturing quality engineer with decades of experience in fabric defect analysis and production optimization.

CURRENT FABRIC ANALYSIS:

CLASSIFICATION RESULTS:
- Primary Defect Type: {cls_result.get('defect_type', 'Unknown')}
- Classification Confidence: {cls_result.get('confidence', 0)*100:.1f}%

DETECTION RESULTS:
- Total Defects Detected: {det_result.get('num_detections', 0)}
- Detailed Detections:
{json.dumps(det_result.get('detections', []), indent=2)}

QUALITY ASSESSMENT:
- Decision: {quality.get('decision', 'Unknown')}
- Quality Grade: {quality.get('grade', 'Unknown')}
- Quality Score: {quality.get('quality_score', 0)}/100
- Severity Level: {quality.get('severity', 'Unknown')}

INDUSTRY STANDARDS CONTEXT:
- IS 1963:2018 (Indian Standard for Textile Testing)
- ASTM D5034 (Standard Test Method for Breaking Strength)

USER QUERY: {user_query}

PROVIDE COMPREHENSIVE ANALYSIS:

1. ROOT CAUSE ANALYSIS:
   - What manufacturing processes likely caused this defect?
   - Machine-related issues (loom tension, needle wear, etc.)
   - Material quality factors
   - Environmental conditions impact

2. IMMEDIATE CORRECTIVE ACTIONS:
   - What should operators do RIGHT NOW?
   - Machine adjustments needed
   - Material inspection requirements
   - Line stoppage recommendations

3. PREVENTIVE MEASURES:
   - Long-term strategies to eliminate this defect
   - Maintenance schedule improvements
   - Quality control checkpoints
   - Operator training requirements

4. QUALITY IMPACT ASSESSMENT:
   - How does this affect fabric usability?
   - End-product suitability (garments, home textiles, industrial)
   - Customer acceptance likelihood
   - Rework vs. scrap decision

5. COST IMPLICATIONS:
   - Estimated production loss
   - Rework costs
   - Downstream impact on manufacturing
   - Prevention investment recommendations

6. COMPLIANCE & STANDARDS:
   - Does this meet IS 1963:2018 requirements?
   - ASTM D5034 compliance status
   - Export quality standards impact

Provide actionable, practical insights that production floor operators and quality managers can immediately implement. Be specific with machine settings, process parameters, and measurable metrics where applicable.
"""
    
    return prompt

def get_image_for_gemini(image_path: str):
    """Prepare image for Gemini API"""
    img = Image.open(image_path)
    return img

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "active",
        "service": "Smart Textile Quality Inspector API",
        "version": "1.0.0",
        "models_loaded": classification_model is not None and detection_model is not None
    }

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload fabric image and create new session
    Returns: session_id for subsequent operations
    """
    try:
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {allowed_extensions}")
        
        # Generate session ID
        session_id = generate_session_id()
        
        # Save uploaded file
        file_path = save_uploaded_file(file, session_id)
        
        # Initialize session
        sessions[session_id] = {
            "session_id": session_id,
            "original_image": file_path,
            "annotated_image": None,
            "classification": None,
            "detection": None,
            "quality_assessment": None,
            "chat_history": [],
            "created_at": datetime.now().isoformat(),
            "filename": file.filename
        }
        
        return {
            "session_id": session_id,
            "filename": file.filename,
            "message": "Image uploaded successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/detect")
async def detect_defects(session_id: str):
    """
    Run YOLOv8 classification and detection on uploaded image
    Returns: Detection results and quality assessment
    """
    try:
        # Validate session
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        image_path = session["original_image"]
        
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        # Check if models are loaded
        if classification_model is None or detection_model is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Run classification
        cls_result = classification_model.predict(image_path, verbose=False)
        classification = process_classification_result(cls_result)
        
        # Run detection
        det_result = detection_model.predict(image_path, verbose=False)
        detection = process_detection_result(det_result)
        
        # Calculate quality assessment
        quality_assessment = calculate_quality_assessment(classification, detection)
        
        # Draw bounding boxes and save annotated image
        annotated_path = str(UPLOAD_DIR / f"{session_id}_annotated.jpg")
        if detection["num_detections"] > 0:
            draw_bounding_boxes(image_path, detection["detections"], annotated_path)
        else:
            # No detections, copy original
            shutil.copy(image_path, annotated_path)
        
        # Update session
        session["classification"] = classification
        session["detection"] = detection
        session["quality_assessment"] = quality_assessment
        session["annotated_image"] = annotated_path
        
        return {
            "session_id": session_id,
            "classification": classification,
            "detection": detection,
            "quality_assessment": quality_assessment,
            "annotated_image_url": f"/image/{session_id}",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/chat")
async def chat_with_gemini(request: ChatRequest):
    """
    Chat with Gemini AI about detection results
    Returns: AI-generated response
    """
    try:
        # Validate session
        if request.session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[request.session_id]
        
        # Check if detection has been run
        if session["classification"] is None or session["detection"] is None:
            raise HTTPException(status_code=400, detail="Please run detection first")
        
        # Build comprehensive prompt
        prompt = build_gemini_prompt(session, request.message)
        
        # Prepare image for Gemini
        annotated_image = get_image_for_gemini(session["annotated_image"])
        
        # Call Gemini API with image
        response = gemini_model.generate_content([prompt, annotated_image])
        
        ai_response = response.text
        
        # Store in chat history
        session["chat_history"].append({
            "role": "user",
            "message": request.message,
            "timestamp": datetime.now().isoformat()
        })
        
        session["chat_history"].append({
            "role": "assistant",
            "message": ai_response,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "session_id": request.session_id,
            "response": ai_response,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/results/{session_id}")
async def get_results(session_id: str):
    """
    Get detection results for a session
    Returns: Complete detection and quality data
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    return {
        "session_id": session_id,
        "filename": session.get("filename"),
        "classification": session.get("classification"),
        "detection": session.get("detection"),
        "quality_assessment": session.get("quality_assessment"),
        "annotated_image_url": f"/image/{session_id}" if session.get("annotated_image") else None,
        "created_at": session.get("created_at")
    }

@app.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    """
    Get chat conversation history for a session
    Returns: Complete chat history
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    return {
        "session_id": session_id,
        "chat_history": session.get("chat_history", []),
        "total_messages": len(session.get("chat_history", []))
    }

@app.get("/image/{session_id}")
async def get_annotated_image(session_id: str):
    """
    Serve annotated image with bounding boxes
    Returns: Image file
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    annotated_image = session.get("annotated_image")
    
    if not annotated_image or not os.path.exists(annotated_image):
        # Fall back to original image
        original_image = session.get("original_image")
        if original_image and os.path.exists(original_image):
            return FileResponse(original_image)
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(annotated_image)

@app.get("/sessions")
async def list_sessions():
    """
    List all active sessions
    Returns: List of session IDs and metadata
    """
    return {
        "total_sessions": len(sessions),
        "sessions": [
            {
                "session_id": sid,
                "filename": data.get("filename"),
                "created_at": data.get("created_at"),
                "has_results": data.get("classification") is not None
            }
            for sid, data in sessions.items()
        ]
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and clean up files
    Returns: Confirmation message
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Delete files
    for key in ["original_image", "annotated_image"]:
        file_path = session.get(key)
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    
    # Remove from sessions
    del sessions[session_id]
    
    return {
        "message": "Session deleted successfully",
        "session_id": session_id
    }

# ==================== SERVER STARTUP ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Smart Textile Quality Inspector API")
    print("="*60)
    print(f"📁 Upload directory: {UPLOAD_DIR.absolute()}")
    print(f"🤖 Classification model: {CLASSIFICATION_MODEL_PATH}")
    print(f"🎯 Detection model: {DETECTION_MODEL_PATH}")
    print(f"🧠 AI Model: Gemini 2.0 Flash")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
