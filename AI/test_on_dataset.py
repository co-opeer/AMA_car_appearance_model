
import os
import tensorflow as tf

saved_model_path = r'C:\Users\PC\AMA_car_appearance_model\AI\res\saved_model.h5'
model = tf.keras.models.load_model(saved_model_path)

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


            if result:
                true_count += 1
            else:
                false_count += 1

    num_cars = len(os.listdir(directory_path))

    print("Photos: ", num_cars)

    print( true_count, " curs")
    print( false_count, " no cars")

    return num_cars, true_count, false_count


print("Photo only with cars")
a = test_model(r"C:\Users\PC\AMA_car_appearance_model\drive\cars_nocars\cars")
print("Photo only without  cars")
b = test_model(r"C:\Users\PC\AMA_car_appearance_model\drive\cars_nocars\no_cars")
print("cars", a[1]/a[0])
print("no_cars", b[2]/b[0])