import numpy as np
import json
import io
import logging
from PIL import Image
import tflite_runtime.interpreter as tflite

def get_model(model_path, classes_path):
    """Load TFLite interpreter and class names. Called once on first request."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    with open(classes_path) as f:
        class_names = json.load(f)
    # Validate
    output_details = interpreter.get_output_details()
    num_classes = output_details[0]['shape'][-1]
    if num_classes != len(class_names):
        raise RuntimeError(
            f"Model expects {num_classes} classes but class_names.json has "
            f"{len(class_names)} entries. Regenerate class_names.json."
        )
    logging.info(f"TFLite model loaded. Classes: {num_classes}")
    return interpreter, class_names

def preprocess_leaf_image(image_bytes):
    """Resize, normalize to [-1, 1] for MobileNetV2."""
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224)).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)

def predict_disease(interpreter, class_names, image_bytes):
    """Run inference and return top-3 predictions."""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        arr = preprocess_leaf_image(image_bytes)
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

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