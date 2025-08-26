# 🐾 PawHub

![PawHub logo](logo.jpg)

## Overview

PawHub helps communities track the drives conducted by NGOs for the welfare of animals, help animals in trouble, reunite lost pets and discover adoptables nearby. It brings together sightings, emergency reporting, adoption, and a social feed—on top of a modular, testable Flutter codebase.

## StatusCode2_pawhub-ML
This repository provides a **FastAPI-based ML service** for analyzing and identifying pets from images.  
It combines **Google Gemini AI**, **YOLOv8**, and **OpenAI CLIP** to:  

- 🐕 **Identify species & breed** (dog, cat, or others) with confidence scoring.  
- 📊 **Extract embeddings** for similarity search or matching.  

---

## 🚀 Features
- **Pet Identification**: Detects species and predicts breed using **Gemini**.  
- **Pet Embeddings**: Generates **CLIP embeddings**, cropped around detected pets with **YOLOv8**.  
- **REST API** with FastAPI (`/identify-pet/`, `/generate-embedding/`).  
- **Extensible**: Can be adapted for other animal recognition tasks.  

---

## 🛠️ Tech Stack
- [FastAPI](https://fastapi.tiangolo.com/) – REST API framework  
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) – Object detection for pet bounding boxes  
- [OpenAI CLIP](https://github.com/openai/CLIP) – Embedding generation  
- [Google Gemini](https://ai.google.dev/) – Species & breed classification  
- [PyTorch](https://pytorch.org/) – Deep learning backend  
- [Pillow / OpenCV](https://pillow.readthedocs.io/) – Image handling  

---
## 📂 Project Structure
- ├── main.py # FastAPI application with endpoints
- ├── requirements.txt # Dependencies
- ├── README.md # Documentation (this file)

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/pet-matching-ml-service.git
cd pet-matching-ml-service
```
### 2. Create a virtual environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt
```

### 3. Set up environment variables

Create a .env file and add your Gemini API key:
```bash
GEMINI_API_KEY=your_api_key_here
```

### 4. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```


## 🔌 API Endpoints

### 1. Identify Pet  
**POST** `/identify-pet/`

**Request Body:**
```json
{
  "url": "https://example.com/dog.jpg"
}
```
**Response Example:**
```json
{
  "species": "Dog",
  "breed": "Indian Pariah Dog",
  "breed_analysis": ["erect ears", "short coat", "slim build"],
  "confidence_score": 9
}
```

### 2. Generate Embedding  
**POST** `/generate-embedding/`

**Request Body:**
```json
{
  "url": "https://example.com/cat.jpg"
}
```
**Response Example:**
```json
{
  "embedding": [0.123, -0.456, 0.789, ...]
}
```

## 🌐 API Docs
Once the server is running, you can explore interactive documentation:
- Swagger UI → (http://localhost:8001/docs)
- ReDoc → (http://localhost:8001/redoc)

## 📦 Dependencies
Minimal `requirements.txt`:

```bash
fastapi
uvicorn
torch
numpy
Pillow
opencv-python
requests
git+https://github.com/openai/CLIP.git
ultralytics
google-generativeai
```

## 🧪 Testing the API
You can use the included test script snippet:
```bash
import requests

API_URL = "http://localhost:8001/identify-pet/"
data = {"url": "https://example.com/dog.jpg"}
response = requests.post(API_URL, json=data)
print(response.json())
```

## 📌 Notes
- Ensure you have CUDA installed if you want GPU acceleration.
- Default models used:
  - CLIP: `ViT-B/32`
  - YOLOv8: `yolov8x.pt`


## Related repositories for the project
- [Backend API](https://github.com/NarenKarthikBM/pawhubAPI/)
- [Flutter App](https://github.com/jayb71/pawhub/)
- [Web Dashboard](https://github.com/NirmanJaiswal36/PawHub-web-dashboard)
