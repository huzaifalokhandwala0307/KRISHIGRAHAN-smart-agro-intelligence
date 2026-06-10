import io
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def load_disease_model(model_path, class_names_path):
    """
    Loads the Keras model and class names JSON file, and validates they match.
    Raises RuntimeError if there's a mismatch.
    """
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load Keras model from {model_path}: {e}")

    try:
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load class names from {class_names_path}: {e}")

    expected_classes = model.output_shape[-1]
    actual_classes = len(class_names)
    
    if expected_classes != actual_classes:
        raise RuntimeError(
            f"Model expects {expected_classes} classes but class_names.json has {actual_classes} entries. "
            "Regenerate class_names.json."
        )
        
    return model, class_names

def preprocess_leaf_image(image_bytes):
    """
    Preprocesses the raw image bytes: opens, resizes to 224x224,
    converts to numpy array, adds batch dimension, and applies MobileNetV2 preprocessing.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        raise ValueError(f"Failed to preprocess image: {e}")

def predict_disease(model, class_names, image_bytes):
    """
    Preprocesses leaf image, runs model inference, and returns prediction details.
    """
    try:
        preprocessed_img = preprocess_leaf_image(image_bytes)
        predictions = model.predict(preprocessed_img)
        
        pred_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_idx]) * 100
        predicted_class = class_names[pred_idx]
        
        is_healthy = "healthy" in predicted_class.lower()
        
        return {
            "disease": predicted_class,
            "confidence": round(confidence, 2),
            "is_healthy": is_healthy
        }
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")
