import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from dataset_loader import val_generator

# Load model
model = load_model("brain_tumor_cnn_model.h5")

# Get predictions
predictions = model.predict(val_generator)
predicted_classes = (predictions > 0.5).astype(int)

true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Brain Tumor Detection")
plt.show()
