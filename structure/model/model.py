
import tensorflow as tf
from keras_tuner.src.backend import keras
from tensorflow.keras import layers, models
import pandas as pd
from sklearn.model_selection import train_test_split

from structure.const import saved_model_path, path_to_csv

# Налаштування змінних для навчання моделі
batch_size = 20
epochs = 10
target_size = 400

# Перевірка наявності збереженої моделі


data = pd.read_csv(path_to_csv, encoding='latin1')

image_paths = data.iloc[:, 0].values
labels = data.iloc[:, 1].values

train_image_paths, test_image_paths, train_labels, test_labels = (
    train_test_split(image_paths, labels, test_size=0.2, random_state=42))

train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))

def load_and_process_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, expand_animations=False, channels=3)
    image = tf.image.resize(image, [target_size, target_size])
    image = image / 255.0
    return image, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_dataset = train_dataset.map(load_and_process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))
test_dataset = test_dataset.map(load_and_process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size)

model = models.Sequential([
    layers.Input(shape=(target_size, target_size, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset
)

print("Збереження навченої моделі...")
keras.saving.save_model(model, saved_model_path)
