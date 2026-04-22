# Brain Tumor Detection using Deep Learning and Grad-CAM

## Project Overview

This project is an AI-powered Brain Tumor Detection System developed using Deep Learning, Transfer Learning, and Explainable AI techniques.

The system allows users to upload MRI scan images and automatically predicts whether a brain tumor is present or not.

In addition to prediction, the system provides visual explainability using Grad-CAM heatmaps, helping users understand the specific region of the MRI image where the model focused its attention.

The application is deployed using Streamlit to provide an interactive web-based user interface.

---

## Objective

The main objective of this project is to assist in the early detection of brain tumors from MRI scan images using Artificial Intelligence.

This system aims to:

* Detect tumor presence from MRI scans
* Improve diagnostic support
* Provide explainable AI-based visualization
* Build a deployable healthcare AI prototype

---

## Technologies Used

### Programming Language

* Python

### Deep Learning Framework

* TensorFlow
* Keras

### Model Architecture

* MobileNetV2 (Transfer Learning)

### Frontend / Deployment

* Streamlit
* Render

### Image Processing

* OpenCV
* PIL
* NumPy

### Explainable AI

* Grad-CAM

---

## Model Architecture

The project uses MobileNetV2, a pre-trained Convolutional Neural Network (CNN) architecture trained on the ImageNet dataset.

Transfer learning is used to leverage learned image features and improve model performance.

### Architecture Flow

* Input MRI Image
* MobileNetV2 Feature Extraction
* Global Average Pooling
* Dense Layer (ReLU)
* Dropout Layer
* Output Layer (Sigmoid)

---

## Features

* Upload MRI images through a web interface
* Detect whether a tumor is present or not
* Display prediction confidence score
* Generate Grad-CAM heatmap visualization
* Easy-to-use Streamlit interface
* Deployable and accessible online

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
```

---

## Deployment

The Brain Tumor Detection System has been deployed as a web application using Render and Streamlit.

Users can access the deployed application through the following link:

[https://brain-tumour-detection-using-cnn.onrender.com](https://brain-tumour-detection-using-cnn.onrender.com)

### Features of the Deployed Application

* Upload MRI scan images in JPG, JPEG, or PNG format
* Predict whether a brain tumor is present or not
* Display confidence score for the prediction
* Generate Grad-CAM heatmap visualization
* Provide an easy-to-use web interface for users

The deployed application allows users to interact with the trained deep learning model without installing any software or running the code locally.

---

## Future Enhancements

* Multi-class brain tumor classification
* Support for larger MRI datasets
* Improved Grad-CAM visualization
* Higher model accuracy using advanced architectures
* Integration with hospital management systems
* Mobile application deployment

---

## Conclusion

This project demonstrates how Deep Learning and Explainable AI can be used for medical image analysis. By combining MobileNetV2, Grad-CAM, and Streamlit, the system provides an effective and user-friendly solution for brain tumor detection from MRI scan images.
