import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

IMG_SIZE = 150

# Load model only once
model = load_model("brain_tumor_cnn_model.h5")

def predict_tumor(img_file):
    img = image.load_img(img_file, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    prob = prediction[0][0]

    return prob