import numpy as np
import json
import io
import logging
import tensorflow as tf
from PIL import Image

def get_model(model_path, classes_path):
    model = tf.keras.models.load_model(model_path)
    with open(classes_path) as f:
        class_names = json.load(f)
    return model, class_names

def preprocess_leaf_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224)).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict_disease(model, class_names, image_bytes):
    try:
        arr = preprocess_leaf_image(image_bytes)
        output = model.predict(arr, verbose=0)[0]
        top3_idx = np.argsort(output)[::-1][:3]
        predictions = [
            {
                "disease": class_names[i],
                "confidence": round(float(output[i]) * 100, 2),
                "is_healthy": "healthy" in class_names[i].lower()
            }
            for i in top3_idx
        ]
        return {"predictions": predictions}
    except Exception as e:
        logging.error(f"Inference error: {e}")
        raise