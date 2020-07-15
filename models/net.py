import tensorflow as tf
import numpy as np
import pathlib
import datetime

output_units = 3


model = tf.keras.models.Sequential([
    # 1 conv
  tf.keras.layers.Conv2D(96, (11,11),strides=(4,4), activation='relu', input_shape=(227, 227, 3)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, strides=(2,2)),
    # 2 conv
  tf.keras.layers.Conv2D(256, (11,11),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
     # 3 conv
  tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
    # 4 conv
  tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
    # 5 Conv
  tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),
  # Flatten layer
  tf.keras.layers.Flatten(),
  # FC layer 1
  tf.keras.layers.Dense(4096, activation='relu'),
    # dropout 0.5
  #FC layer 2
  tf.keras.layers.Dense(4096, activation='relu'),
    # dropout 0.5
  tf.keras.layers.Dense(output_units, activation='softmax')
])

model.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()
