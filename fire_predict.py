import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 150

model = load_model("fire_detection_model.h5")
print("Model loaded successfully!")

def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image. Please check the path and file format.")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    

    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        print("Fire detected!")
    else:
        print("No Fire detected.")


predict_image("fire3.jpg")  
