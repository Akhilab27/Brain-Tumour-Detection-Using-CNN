import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import load_model
from dataset_loader import val_generator

# Load model
model = load_model("brain_tumor_cnn_model.h5")

# Predictions
predictions = model.predict(val_generator)
predicted_classes = (predictions > 0.5).astype(int).reshape(-1)

# True labels
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# PRINT METRICS (VERY IMPORTANT)
print("\nClassification Report:\n")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)

print("\nConfusion Matrix:\n", cm)

# Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)

plt.title("Confusion Matrix - Brain Tumor Detection")
plt.show()