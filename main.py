import logging
from io import BytesIO
from PIL import Image
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
import requests

# Import your ML model libraries
import torch
import clip
from ultralytics import YOLO
import cv2
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- The ML Service Class (Adapted for API use) ---
class PetMLService:
    def __init__(self):
        try:
            # IMPORTANT: Set your GEMINI_API_KEY as an environment variable on this server
            # or replace "YOUR_KEY_HERE" with the actual key.
            GEMINI_API_KEY = YOUR_KEY_HERE
            genai.configure(api_key="GEMINI_API_KEY")
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            logger.info("✅ Gemini model initialized.")
        except Exception as e:
            self.gemini_model = None
            logger.error(f"🔴 FAILED to initialize Gemini: {e}")

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.yolo_model = YOLO('yolov8x.pt')
            logger.info(f"✅ CLIP & YOLO models loaded on: {self.device}")
        except Exception as e:
            self.clip_model = None
            self.yolo_model = None
            logger.error(f"🔴 FAILED to initialize CLIP/YOLO: {e}")

    def identify_pet_details(self, image_pil: Image.Image) -> dict | None:
        if not self.gemini_model: return None
        prompt = """
        Analyze the provided image of an animal by following these steps:

        1. **Species Identification:** First, identify the primary species in the image (e.g., "Dog", "Cat", "Rabbit").

        2. **Conditional Analysis:**
        * **If the species is "Dog"**:
            a. **Feature Analysis:** Briefly list the dog's key physical features (e.g., ear type, snout shape, coat color and length, body build, tail).
            b. **Breed Comparison:** Compare these features against known breeds, paying close attention to characteristics of an Indian Pariah Dog (Indie) as well as common purebreds or their mixes found in India (e.g., Spitz, German Shepherd, Labrador). Note which features align with which breed.
            c. **Final Conclusion:** Conclude the most likely primary breed type. If it is a clear mix, identify the most visually dominant breed. If it is a quintessential street dog, identify it as an "Indian Pariah Dog".
        * **If the species is "Cat"**:
            a. **Feature Analysis:** Briefly list the cat's key physical features (e.g., face shape, ear size, coat pattern and length, body type).
            b. **Breed Comparison:** Compare these features against known cat breeds or types, such as "Indian Billboard Cat" (Domestic Shorthair), Persian, or Siamese.
            c. **Final Conclusion:** Conclude the most likely breed type. For most domestic cats, use "Indian Billboard Cat".
        * **If the species is neither "Dog" nor "Cat"**:
            a. Simply note the species and its primary visual characteristics.

        3. **JSON Output:** Finally, based only on your analysis, provide a strict JSON object with the following keys:
        * "species": The species identified in Step 1.
        * "breed": The single most likely primary breed identified in your conclusion. For species other than Dog or Cat, this can be the species name again.
        * "breed_analysis": An array of strings, each string being a short keyword or phrase (e.g., ["erect ears", "short coat", "slim build"]). If the species is neither Dog nor Cat, return an empty array [].
        * "confidence_score": A numerical score from 1 to 10, where 1 is a wild guess and 10 is virtual certainty.

        Do not include any text outside the final JSON object.
        """
        try:
            import json
            response = self.gemini_model.generate_content([prompt, image_pil])
            cleaned_json = response.text.replace('```json', '').replace('```', '').strip()
            return json.loads(cleaned_json)
        except Exception as e:
            logger.error(f"🔴 Error during Gemini ID: {e}")
            return None

    def generate_embedding(self, image_pil: Image.Image) -> np.ndarray | None:
        if not self.yolo_model or not self.clip_model: return None
        try:
            image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            results = self.yolo_model(image_cv)
            best_box = None
            highest_confidence = 0
            for r in results:
                for box in r.boxes:
                    cls_name = self.yolo_model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if cls_name in ['dog', 'cat'] and conf > highest_confidence:
                        highest_confidence = conf
                        best_box = box.xyxy[0].cpu().numpy().astype(int)
            
            target_image_pil = image_pil
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                cropped_cv = cv2.cvtColor(image_cv[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                target_image_pil = Image.fromarray(cropped_cv)

            processed = self.preprocess(target_image_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.clip_model.encode_image(processed)
                embedding /= embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"🔴 Error during embedding generation: {e}")
            return None

# --- FastAPI Setup ---
app = FastAPI(title="Pet Matching ML Service")
service = PetMLService() # Load models once on startup

class ImageURLRequest(BaseModel):
    url: str

def download_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        logger.error(f"🔴 Failed to download image: {e}")
        return None

@app.post("/identify-pet/")
async def api_identify_pet(request: ImageURLRequest):
    if not service.gemini_model:
        raise HTTPException(status_code=503, detail="Gemini Model unavailable.")
    image = download_image_from_url(request.url)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image URL or download failed.")
    details = service.identify_pet_details(image)
    if details is None:
        raise HTTPException(status_code=500, detail="Failed to identify pet.")
    return details

@app.post("/generate-embedding/")
async def api_generate_embedding(request: ImageURLRequest):
    if not service.clip_model:
        raise HTTPException(status_code=503, detail="Embedding Models unavailable.")
    image = download_image_from_url(request.url)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image URL or download failed.")
    embedding = service.generate_embedding(image)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Failed to generate embedding.")
    return {"embedding": embedding.tolist()}
