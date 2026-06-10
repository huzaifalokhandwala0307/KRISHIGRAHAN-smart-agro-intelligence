import numpy as np
import json
import tensorflow as tf
from inference import predict_disease, load_disease_model

# Load model and classes
model, classes = load_disease_model("krishigrahan_plant_disease_v1.keras", "class_names.json")

# Create a mock image (224x224 RGB image as jpeg bytes)
from PIL import Image
import io
img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG')
img_bytes = img_byte_arr.getvalue()

# Predict disease
result = predict_disease(model, classes, img_bytes)
print("Top 3 Predictions Result structure:")
print(json.dumps(result, indent=2))