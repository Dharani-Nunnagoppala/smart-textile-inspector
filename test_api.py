"""
Test script for Smart Textile Quality Inspector API
Run after starting the FastAPI server
"""

import requests
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8002"
TEST_IMAGE_PATH = "/teamspace/studios/this_studio/data/Data Set/captured/Lines/20180531_140759.jpg"  # Corrected path to the dataset

def print_separator():
    print("\n" + "="*60 + "\n")

def test_health_check():
    """Test 1: Health Check"""
    print("TEST 1: Health Check")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_upload_image(image_path):
    """Test 2: Upload Image"""
    print("TEST 2: Upload Image")
    
    if not Path(image_path).exists():
        print(f"❌ Test image not found: {image_path}")
        print("Please provide a valid fabric image path")
        return None
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/upload", files=files)
    
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    return result.get('session_id') if response.status_code == 200 else None

def test_detect_defects(session_id):
    """Test 3: Run Detection"""
    print("TEST 3: Run Detection")
    
    if not session_id:
        print("❌ No session_id provided")
        return False
    
    response = requests.post(f"{BASE_URL}/detect", params={'session_id': session_id})
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Classification: {result['classification']['defect_type']} "
              f"({result['classification']['confidence']*100:.1f}%)")
        print(f"Detections: {result['detection']['num_detections']} defects found")
        print(f"Quality: {result['quality_assessment']['decision']} "
              f"(Grade: {result['quality_assessment']['grade']})")
        print(f"Annotated Image: {result['annotated_image_url']}")
        return True
    else:
        print(f"Error: {response.json()}")
        return False

def test_chat(session_id, message):
    """Test 4: Chat with Gemini"""
    print("TEST 4: Chat with Gemini")
    
    if not session_id:
        print("❌ No session_id provided")
        return False
    
    payload = {
        "session_id": session_id,
        "message": message
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"User: {message}")
        print(f"AI Response: {result['response'][:200]}...")
        return True
    else:
        print(f"Error: {response.json()}")
        return False

def test_get_results(session_id):
    """Test 5: Get Results"""
    print("TEST 5: Get Results")
    
    if not session_id:
        print("❌ No session_id provided")
        return False
    
    response = requests.get(f"{BASE_URL}/results/{session_id}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Session has results: {result['classification'] is not None}")
        return True
    else:
        print(f"Error: {response.json()}")
        return False

def test_get_history(session_id):
    """Test 6: Get Chat History"""
    print("TEST 6: Get Chat History")
    
    if not session_id:
        print("❌ No session_id provided")
        return False
    
    response = requests.get(f"{BASE_URL}/history/{session_id}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total Messages: {result['total_messages']}")
        return True
    else:
        print(f"Error: {response.json()}")
        return False

def test_list_sessions():
    """Test 7: List Sessions"""
    print("TEST 7: List Sessions")
    
    response = requests.get(f"{BASE_URL}/sessions")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total Sessions: {result['total_sessions']}")
        return True
    else:
        print(f"Error: {response.json()}")
        return False

def run_all_tests():
    """Run complete test suite"""
    print("\n" + "🧪 SMART TEXTILE QUALITY INSPECTOR - API TEST SUITE")
    print_separator()
    
    # Test 1: Health Check
    if not test_health_check():
        print("❌ Health check failed. Is the server running?")
        return
    print("✅ Health check passed")
    print_separator()
    
    # Test 2: Upload
    session_id = test_upload_image(TEST_IMAGE_PATH)
    if session_id:
        print(f"✅ Upload successful. Session ID: {session_id}")
    else:
        print("❌ Upload failed")
        return
    print_separator()
    
    # Test 3: Detection
    if test_detect_defects(session_id):
        print("✅ Detection successful")
    else:
        print("❌ Detection failed")
        return
    print_separator()
    
    # Test 4: Chat
    chat_messages = [
        "What defects did you find?",
        "What might have caused this?",
        "What should we do to fix this?"
    ]
    
    for msg in chat_messages:
        if test_chat(session_id, msg):
            print(f"✅ Chat successful: '{msg}'")
        else:
            print(f"❌ Chat failed: '{msg}'")
        print_separator()
    
    # Test 5: Get Results
    if test_get_results(session_id):
        print("✅ Get results successful")
    else:
        print("❌ Get results failed")
    print_separator()
    
    # Test 6: Get History
    if test_get_history(session_id):
        print("✅ Get history successful")
    else:
        print("❌ Get history failed")
    print_separator()
    
    # Test 7: List Sessions
    if test_list_sessions():
        print("✅ List sessions successful")
    else:
        print("❌ List sessions failed")
    print_separator()
    
    print("🎉 ALL TESTS COMPLETED!")
    print(f"\n📊 View annotated image at: {BASE_URL}/image/{session_id}")
    print(f"📖 API Documentation: {BASE_URL}/docs")

if __name__ == "__main__":
    print("\n⚠️  PREREQUISITES:")
    print("1. FastAPI server must be running (python main.py)")
    print("2. YOLO models must be loaded")
    print("3. Test image must exist at:", TEST_IMAGE_PATH)
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    run_all_tests()