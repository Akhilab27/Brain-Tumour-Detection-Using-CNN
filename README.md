# Brain Tumor Detection using Deep Learning and Grad-CAM

## Project Overview

This project is an AI-powered **Brain Tumor Detection System** developed using **Deep Learning, Transfer Learning, and Explainable AI techniques**.

The system allows users to upload MRI scan images and automatically predicts whether a **brain tumor is present or not**.

In addition to prediction, the system provides **visual explainability using Grad-CAM heatmaps**, helping users understand the specific region of the MRI image where the model focused its attention.

The application is deployed using **Streamlit** to provide an interactive web-based user interface.

---

## Objective

The main objective of this project is to assist in the **early detection of brain tumors from MRI scan images** using Artificial Intelligence.

This system aims to:

- Detect tumor presence from MRI scans
- Improve diagnostic support
- Provide explainable AI-based visualization
- Build a deployable healthcare AI prototype

---

## Technologies Used

### Programming Language
- Python

### Deep Learning Framework
- TensorFlow
- Keras

### Model Architecture
- MobileNetV2 (Transfer Learning)

### Frontend / Deployment
- Streamlit

### Image Processing
- OpenCV
- PIL
- NumPy

### Explainable AI
- Grad-CAM

---

## Model Architecture

The project uses **MobileNetV2**, a pre-trained Convolutional Neural Network (CNN) architecture trained on the ImageNet dataset.

Transfer learning is used to leverage learned image features and improve model performance.

### Architecture Flow

1. Input MRI Image
2. MobileNetV2 Feature Extraction
3. Global Average Pooling
4. Dense Layer (ReLU)
5. Dropout Layer
6. Output Layer (Sigmoid)

---

## Project Structure

```bash
Brain Tumour Detection/
│
├── src/
│   ├── app.py
│   ├── train.py
│   ├── predict.py
│   ├── model.py
│   ├── dataset_loader.py
│   ├── grad_cam.py
│
├── brain_tumor_cnn_model.h5
├── requirements.txt
├── README.md