import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Mi prendo il dataset
f_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full),(x_test, y_test) = f_mnist.load_data()

# Creo un validation set e scalo tutto
x_valid, x_train = x_train_full[:5000] / 255.0, x_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_name = ["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt",
              "sneaker", "bag", "ankle boot"]

# Creo un modello
# Inizializzo un ogetto sequential
model = keras.models.Sequential()
# Primo layer di input
model.add(keras.layers.Flatten(input_shape=[28,28]))
# Creo tre layers
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# Guardo il modello
model.summary()

# Compilo il modello
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=["accuracy"])

# Faccio partire il training
history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))

import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

model.predict(x_train[:1])


