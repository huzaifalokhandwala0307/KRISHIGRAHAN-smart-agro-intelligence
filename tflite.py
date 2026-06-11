import tensorflow as tf

model = tf.keras.models.load_model("krishigrahan_plant_disease_v1.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("disease_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Done. File size:", len(tflite_model) / 1024 / 1024, "MB")