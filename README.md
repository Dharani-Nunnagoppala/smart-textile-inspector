<<<<<<< HEAD
# Smart Textile Quality Inspector 🧵

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python Version](https://img.shields.io/badge/Python-3.10+-blue)
![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Streamlit-red)
![AI Models](https://img.shields.io/badge/AI-YOLOv8%20%7C%20Gemini%202.0-orange)

An AI-powered system for automated textile defect detection and quality analysis. This project combines computer vision (YOLOv8) for defect identification with Generative AI (Gemini 2.0 Flash) for intelligent root cause analysis and corrective recommendations.

## 🌟 Key Features

*   **Dual-Model Architecture:**
    *   **Classification:** YOLOv8n-cls model to identify the primary defect type.
    *   **Detection:** YOLOv8n model to localize specific defects with bounding boxes.
*   **Intelligent Analysis:** Integrated **Gemini 2.0 Flash** chatbot provides:
    *   Root cause analysis based on industry standards (IS 1963:2018, ASTM D5034).
    *   Immediate corrective actions for operators.
    *   Long-term preventive measures.
*   **Interactive Dashboard:** User-friendly Streamlit interface for image upload, visualization, and reporting.
*   **Automated Quality Scoring:** Calculates a quality grade (A/B/C) and score (0-100) based on defect severity and frequency.

## 📊 Model Performance

The system is trained on the [Fabric Defect Dataset](https://www.kaggle.com/datasets/rmshashi/fabric-defect-dataset) containing 666 images across 5 categories.

| Model Type | Architecture | Metric | Result |
| :--- | :--- | :--- | :--- |
| **Classification** | YOLOv8n-cls | Top-1 Accuracy | **90.4%** |
| **Detection** | YOLOv8n | mAP@50 | **78.9%** |

### Supported Defect Classes
1.  **Horizontal:** Weft-related defects.
2.  **Vertical:** Warp-related defects.
3.  **Hole:** Physical punctures or tears.
4.  **Lines:** Continuous linear irregularities.
5.  **Hole_variant:** Variations of hole defects.

## 🛠️ Technology Stack

*   **Frontend:** Streamlit
*   **Backend API:** FastAPI
*   **Computer Vision:** Ultralytics YOLOv8
*   **GenAI:** Google Gemini 2.0 Flash
*   **Image Processing:** OpenCV, PIL
*   **Language:** Python 3.12

## 🚀 Getting Started

### Prerequisites
*   Python 3.10 or higher
*   Google Gemini API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/smart-textile-inspector.git
    cd smart-textile-inspector
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Key:**
    *   Open `main.py` and replace `GEMINI_API_KEY` with your actual key (or set it as an environment variable).

### Usage

1.  **Start the Backend API:**
    ```bash
    uvicorn main:app --reload --port 8000
    ```
    The API documentation will be available at `http://localhost:8000/docs`.

2.  **Start the Frontend Dashboard:**
    ```bash
    streamlit run app.py
    ```
    The app will open in your browser at `http://localhost:8501`.

## 📱 Application Workflow

1.  **Upload:** User uploads an image of a fabric sample.
2.  **Analyze:** The system runs both classification and detection models.
3.  **Visualize:** Detected defects are highlighted with bounding boxes.
4.  **Assess:** A quality score and decision (PASS/REJECT/REVIEW) are generated.
5.  **Consult:** Users can chat with the AI expert to understand *why* the defect occurred and *how* to fix it.

## 📂 Project Structure

```
├── app.py                      # Streamlit frontend application
├── main.py                     # FastAPI backend server
├── test_api.py                 # API testing script
├── textile_defect_detection_yolov8.ipynb  # Model training notebook
├── runs/                       # Trained model weights
│   ├── classify/               # Classification model artifacts
│   └── detect/                 # Detection model artifacts
├── uploads/                    # Temporary storage for uploaded images
└── requirements.txt            # Python dependencies
```

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

*   **Dataset:** [Fabric Defect Dataset](https://www.kaggle.com/datasets/rmshashi/fabric-defect-dataset) by rmshashi.
*   **Ultralytics:** For the amazing YOLOv8 library.
*   **Google:** For the Gemini API.
=======
"# smart-textile-inspector" 
>>>>>>> 099cc6b5569b5da4a62ce78f85eaeca761eb4598
