import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("brain_tumor_cnn_model.h5")

IMG_SIZE = 150

# Load image
img = image.load_img("dataset/no/1 no.jpeg", target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

print("Raw prediction value:", prediction[0][0])


# Output
confidence = prediction[0][0] * 100

prob = prediction[0][0]

print(f"Prediction confidence: {prob:.4f}")

if prob >= 0.5:
    print("Tumour Detected")
else:
    print("No Tumour Detected")
