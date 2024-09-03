import tensorflow as tf
import numpy as np
import cv2

def load_model(model_path='plant_disease_model.h5'):
    model = tf.keras.models.load_model(model_path)
    return model

def prepare_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_disease(model, image):
    prediction = model.predict(image)
    class_indices = ['Healthy', 'Disease1', 'Disease2']  # Update with actual class names
    predicted_class = class_indices[np.argmax(prediction)]
    return predicted_class
