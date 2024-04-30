import os
import shutil

import tensorflow as tf


def predict_car(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=0)
    return predictions[0] > 0.5


def test_model(directory_path):
    true_count = 0
    false_count = 0

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            image_path = os.path.join(root, file)
            result = predict_car(image_path)

            if result > 0.5:
                new_path = os.path.join(r"C:\Users\PC\PycharmProjects\lab1\AI\res\TestModel\cars", file)
                shutil.move(image_path, new_path)
                true_count += 1
            else:
                new_path = os.path.join(r"C:\Users\PC\PycharmProjects\lab1\AI\res\TestModel\no_cars", file)
                shutil.move(image_path, new_path)
                false_count += 1

    print("Cars:", false_count)
    print("No cars:", true_count)


saved_model_path = r"C:\Users\PC\PycharmProjects\lab1\AI\res\saved_model.h5"


model = tf.keras.models.load_model(saved_model_path)

test_model(r"C:\Users\PC\PycharmProjects\lab1\drive\cars_nocars")
