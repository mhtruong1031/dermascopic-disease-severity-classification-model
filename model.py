import tensorflow as tf
from keras import layers

input_shape = (765,572,3)
kernal_size = (5, 5)

# TODO: Import Dataset

# VGG-16 Instantiation
model = tf.keras.Sequential([
    layers.Conv2D(input_shape=input_shape, kernal_size=kernal_size, padding='same', activation='relu'),
    layers.Conv2D(filters=64, kernal_size=kernal_size, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2), strides=(2,2)),
    layers.Conv2D(filters=128, kernal_size=kernal_size, padding='same', activation='relu'),
    layers.Conv2D(filters=128, kernal_size=kernal_size, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2), strides=(2,2)),
    layers.Conv2D(filters=256, kernal_size=kernal_size, padding='same', activation='relu'),
    layers.Conv2D(filters=256, kernal_size=kernal_size, padding='same', activation='relu'),
    layers.Conv2D(filters=256, kernal_size=kernal_size, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2), strides=(2,2)),
    layers.Conv2D(filters=512, kernal_size=kernal_size, padding='same', activation='relu'),
    layers.Conv2D(filters=512, kernal_size=kernal_size, padding='same', activation='relu'),
    layers.Conv2D(filters=512, kernal_size=kernal_size, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2), strides=(2,2)),
    layers.Conv2D(filters=512, kernal_size=kernal_size, padding='same', activation='relu'),
    layers.Conv2D(filters=512, kernal_size=kernal_size, padding='same', activation='relu'),
    layers.Conv2D(filters=512, kernal_size=kernal_size, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2), strides=(2,2)),
    layers.Flatten(),
    layers.Dense(units=4096, activation='relu'),
    layers.Dense(units=4096, activation='relu'),
    layers.Dense(units=3, activation='relu')
])


