import numpy as np
from tensorflow.keras.models import load_model
from dataset_loader import val_generator

model = load_model("brain_tumor_cnn_model.h5")

loss, accuracy = model.evaluate(val_generator)

print(f"Validation Accuracy: {accuracy * 100:.2f}%")
