from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import requests
from .inference import OilSpillDetector
from .model import ImageURL
from fastapi.middleware.cors import CORSMiddleware


# TODO: 
# 1. added type checking code for the image
# 2. add pydantic for url


api = FastAPI(title="Oil Spill Detection API", version="1.0.0")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,   
    allow_methods=["*"],      
    allow_headers=["*"],     
)

# Initialize the model
# detector = OilSpillDetector(model_path="../model/oil_spill_best_md_obj.pt")
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "oil_spill_best_md_obj.pt")
detector = OilSpillDetector(model_path=model_path)

@api.get("/")
async def root():
    return {"message": "Oil Spill Detection API", "status": "running"}

@api.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": detector.is_loaded()}


@api.post("/predict")
async def predict_oil_spill(file: UploadFile = File(...)):
    """Predict oil spill in uploaded image"""

    try:

        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        image.verify()
        image = Image.open(io.BytesIO(image_data))

        results = detector.predict(image, return_image=True)

        return JSONResponse(content={
            "status": "Success",
            "detections": results["detections"],
            "total_detections": results["total_detections"],
            "processing_time": results["processing_time"],
            "annotated_image": results["annotated_image"]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed {str(e)}")

@api.post("/predict-url")
async def predict_oil_spill_from_url(image_url: ImageURL):

    try:
        response = requests.get(str(image_url.url), timeout=30)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="URL does not point to an image")

        image = Image.open(io.BytesIO(response.content))

        results = detector.predict(image,return_image=True)

        return JSONResponse(content={
            "status": "Success",
            "detections": results["detections"],
            "total_detections": results["total_detections"],
            "processing_time": results["processing_time"],
            "annotated_image": results["annotated_image"],
            "image_url": str(image_url.url)
        })

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

