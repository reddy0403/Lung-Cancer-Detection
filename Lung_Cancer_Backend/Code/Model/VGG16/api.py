from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

# Initialize FastAPI
app = FastAPI()

# CORS Middleware for frontend communication
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
MODEL_PATH = "C:/Lung-Cancer-Detection-main/Lung_Cancer_Backend/Code/Python/lung_cancer_vgg16_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

MODEL = tf.keras.models.load_model(MODEL_PATH)

# Class Labels
CLASS_NAMES = ["Adenocarcinoma", "Large cell carcinoma", "Normal", "Squamous Cell Carcinoma"]

# Health Check Endpoint
@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    with Image.open(BytesIO(data)) as img:
        img.thumbnail((256, 256))  # Resize while keeping aspect ratio
        image = np.array(img) / 255.0  # Normalize

    print(f"Image shape before model input: {image.shape}")  # Debugging
    return image

# Prediction Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())

    # Remove alpha channel if present
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    
    # Resize the image to 256x256
    image = tf.image.resize(image, (256, 256))
    
    # Expand dimensions for batch processing
    img_batch = tf.expand_dims(image, 0)
    
    # Make Prediction
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Save results to a file
    with open("model_results.txt", "w") as f:
        f.write(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}\n")

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

# Endpoint to fetch the latest prediction results
@app.get("/results")
def get_model_results():
    file_path = "model_results.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return {"Model Results": f.read()}
    return {"Model Results": "No results found. Train the model first."}

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
