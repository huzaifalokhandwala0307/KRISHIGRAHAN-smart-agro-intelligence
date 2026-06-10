import tensorflow as tf

model = tf.keras.models.load_model(
    "krishigrahan_plant_disease_v1.keras"
)

print(model.output_shape)