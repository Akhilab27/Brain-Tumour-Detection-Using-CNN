from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from tensorflow.keras.models import load_model
from dataset_loader import val_generator
import numpy as np

model = load_model("brain_tumor_cnn_model.h5", compile=False)

# Get actual labels
y_true = val_generator.classes

# Predict on validation set
predictions = model.predict(val_generator)
y_pred = (predictions > 0.5).astype(int).flatten()

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")