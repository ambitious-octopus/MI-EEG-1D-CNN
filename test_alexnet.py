import tensorflow as tf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow.keras import datasets, layers, models, losses
import numpy as np
from PIL import Image
import PIL

model = models.Sequential()
model.add(layers.Conv2D(96, (11,11), strides=(4,4), input_shape=(227, 227, 3),
                        activation='relu', data_format="channels_last"))
model.add(layers.MaxPooling2D((3,3), strides=(2,2)))
model.add(layers.Conv2D(256, (5,5), strides=(2,2), padding='valid', activation='relu'))
model.add(layers.MaxPooling2D((3,3), strides=(2,2)))
model.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(layers.Conv2D(384, 13, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(256, 6, strides=4, padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

image = Image.open("C:\\Users\\franc_pyl533c\\OneDrive\\Immagini\\Teenage wedding\\IMG_2780.JPG")
img = image.resize((227, 227), Image.ANTIALIAS)
data  = np.asarray(img)
input = data.reshape(1, data.shape[0], data.shape[1], data.shape[2])



blocks = [0]
outputs = [model.layers[i].output for i in blocks]
print(outputs)

new_model = tf.keras.Model(inputs=model.inputs , outputs=outputs)
features = new_model.predict(input)

fmap = features
print(fmap.shape)

# plt.imshow(fmap[0, :, :, 0], cmap='gray')
# plt.show()

# fig, ax = plt.subplots(1, 20)
# for a in range(20):
#     ax[a].imshow(fmap[0, :, :, a], cmap='Purples')
# plt.show()

# ncol = 20
# nrow = 3
#
# fig = plt.figure(figsize=(ncol+1, nrow+1))
# #
# gs = gridspec.GridSpec(nrow, ncol,
#          wspace=0.0, hspace=0.0,
#          top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1),
#          left=0.5/(ncol+1), right=1-0.5/(ncol+1))
#
# r = 0
# c = 0
# for i in range(60):
#     if c == 20:
#         c = 0
#         r += 1
#     ax = plt.subplot(gs[r, c])
#     ax.imshow(fmap[0, :, :, i], cmap="viridis")
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     c += 1
#
#
#
# plt.show()

plt.imshow(data)
plt.show()