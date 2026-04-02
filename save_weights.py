from tensorflow.keras.models import load_model

model = load_model("brain_tumor_cnn_model.h5", compile=False)
model.save_weights("brain_tumor_weights.weights.h5")

print("Weights saved successfully")