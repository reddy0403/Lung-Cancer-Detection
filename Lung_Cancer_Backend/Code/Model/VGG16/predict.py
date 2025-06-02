import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model_path = "C:\Lung-Cancer-Detection-main\Lung_Cancer_Backend\Code\Python\lung_cancer_vgg16_model.h5"  # Update your model path
model = load_model(model_path)

# Define class labels
class_labels = ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Adjust if needed
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict a single image
def predict_image(image_path):
    processed_img = preprocess_image(image_path)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)
    
    print(f"✅ Prediction for {image_path}: {class_labels[predicted_class]}")
    return class_labels[predicted_class]

# Test with a sample image
test_image_path = "C:/Lung-Cancer-Detection-main/API/Data/test/adenocarcinoma/000108.png"  # Change to your test image path
predict_image(test_image_path)
