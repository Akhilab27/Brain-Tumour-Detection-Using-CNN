from tensorflow.keras.models import load_model

# Load existing .h5 model
model = load_model("brain_tumor_cnn_model.h5", compile=False)

# Save in new .keras format
model.save("brain_tumor_cnn_model.keras")

print("Model saved successfully as .keras")