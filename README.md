# Lung Cancer Detection using Deep Learning

This project focuses on detecting **Non-Small Cell Lung Cancer (NSCLC)** from CT scan images using deep learning techniques. The model classifies the cancer into three categories: **Adenocarcinoma**, **Small Cell Carcinoma**, and **Large Cell Carcinoma**.

## 🧠 Models Used
- Convolutional Neural Network (CNN)
- VGG16
- ResNet50
- InceptionV3

## 📁 Dataset
The dataset used contains labeled CT scan images of three types of lung cancer:
- Adenocarcinoma
- Large Cell Carcinoma
- Small Cell Carcinoma

> [Include dataset source or note if it was from Kaggle, GitHub, or a private dataset]

## ⚙️ Tech Stack
- Python
- TensorFlow / Keras
- NumPy, Pandas
- OpenCV (for image processing)
- Matplotlib / Seaborn (for visualization)

## 📊 Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

## 🧪 Implementation Steps
1. **Data Preprocessing**
   - Resize CT scan images
   - Normalize pixel values
   - Label encoding

2. **Model Building**
   - Trained CNN from scratch
   - Fine-tuned VGG16, ResNet50, and InceptionV3

3. **Training**
   - Split into train, validation, and test sets
   - Used EarlyStopping and ModelCheckpoint

4. **Testing & Validation**
   - Compared all models
   - Evaluated using confusion matrix and performance metrics
