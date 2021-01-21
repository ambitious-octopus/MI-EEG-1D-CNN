import numpy as np
import os
from data_processing.general_processor import Utils
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("TkAgg")
from sklearn.preprocessing import StandardScaler # Usare MIn MAx scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
#############################################################################
"""
Load only and split
"""
exclude = [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1,110) if n not in exclude]

base_path = "D:\\datasets\\eeg_dataset\\C3_C4_sub"
#Load data
xs = list()
ys = list()
for sub in subjects:
    xs.append(Utils.cut_width(np.load(os.path.join(base_path, "x_C3_C4_sub_" + str(sub) + ".npy"))))
    ys.append(np.load(os.path.join(base_path, "y_C3_C4_sub_" + str(sub) + ".npy")))

x_data = np.concatenate(tuple(xs))
y = np.concatenate(tuple(ys))

# Processing y
total_labels = np.unique(y)
mapping = {}
for x in range(len(total_labels)):
  mapping[total_labels[x]] = x
for x in range(len(y)):
  y[x] = mapping[y[x]]

y_categorical = np.array([int(label) for label in y])
y_one_hot = tf.keras.utils.to_categorical(y)

#Scale
#reshape (18511,1280)
x_data_resh = x_data.reshape(x_data.shape[0], x_data.shape[2]*x_data.shape[1])
# x_data_scale = MinMaxScaler().fit_transform(x_data_mono) #Fare MinMax Scare, portare tra 0 e 1
x_data_scale = minmax_scale(x_data_resh, axis=1)


# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# plt.subplot(1,2,1)
# plt.plot(x_data[0][0])
# plt.show()
# plt.subplot(1,2,2)
# plt.plot(x_data_scale[0][:640])
# plt.show()

x_resh = x_data_scale.reshape(x_data_scale.shape[0], x_data_scale.shape[1], 1)
x_train, x_test, y_train, y_test = train_test_split(x_resh, y_one_hot, stratify = y_one_hot, test_size=0.33, random_state=1)
#%%
#Convolution Neural Network
# [samples, time steps, features].
# real_x_train = x_train.reshape(14808, 640, 2)
# real_x_test = x_test.reshape(3703, 640, 2)
learning_rate = 1e-4 # default 1e-3
kernel_size = 3
drop_rate = 0.5
batch_size = 10
loss = tf.keras.losses.sparse_categorical_crossentropy  #tf.keras.losses.categorical_crossentropy

if loss == tf.keras.losses.sparse_categorical_crossentropy:
    x_resh = x_data_scale.reshape(x_data_scale.shape[0], x_data_scale.shape[1], 1)
    x_train, x_test, y_train, y_test = train_test_split(x_resh, y_categorical, stratify=y_categorical, test_size=0.33,
                                                        random_state=1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size, activation='relu', padding= "same", input_shape=(1280, 1)))
model.add(tf.keras.layers.AvgPool1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size, activation='relu', padding= "same"))
model.add(tf.keras.layers.AvgPool1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size, activation='relu', padding= "same"))
model.add(tf.keras.layers.AvgPool1D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(drop_rate))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(drop_rate))
model.add(tf.keras.layers.Dense(4, activation='softmax'))
model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=100, batch_size=batch_size, steps_per_epoch=x_train.shape[0]/batch_size,
                    shuffle=True, validation_data=(x_test,y_test))