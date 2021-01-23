import numpy as np
import os
from data_processing.general_processor import Utils
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # Usare MIn MAx scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#############################################################################


exclude =  [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1,109) if n not in exclude]
data_path = "D:\\datasets\\eeg_dataset\\C3_C4_sub_no_base" #save_path = "D:\\datasets\\eeg_dataset\\C3_C4_sub_no_base_min1_max3" "D:\\datasets\\eeg_dataset\\C3_C4_sub_no_base"
xs, ys = Utils.load_sub_by_sub(subjects, data_path)
xs, ys = Utils.scale_sub_by_sub(xs, ys)

# Questo fa una modifica alle label aggiungendo il nome del sogetto in modo da stratificare
new_y = list()
for x, y, index_sub in zip(xs, ys, range(len(xs))):
    subj_array = list()
    for index_label, label in enumerate(y):
        subj_array.append(label + str(index_sub))
    new_y.append(np.array(subj_array))
y = np.concatenate(new_y)
x = np.concatenate(xs)

#Qui faccio un reshape
x_resh = x.reshape(x.shape[0], x.shape[2]*x.shape[1])
#Statifico per soggetto ed esempi -> Proporzione perfetta tra soggetti e combinazioni di task in x_test
x_train, x_test, y_train, y_test = train_test_split(x_resh, y, stratify =y, test_size=0.30,  random_state=17)

#Processing y
y_train = Utils.to_numerical(y_train, by_sub=True)
y_test = Utils.to_numerical(y_test, by_sub=True)

#Reshape x -> (sample, width, height, channels)
x_train_resh = x_train.reshape(x_train.shape[0], int(x_train.shape[1]/2), 2, 1).astype(np.float64)
x_test_resh = x_test.reshape(x_test.shape[0], int(x_train.shape[1]/2), 2, 1).astype(np.float64)

# #Test rehsape
# plt.subplot(2,2,1, title="x_train_before")
# plt.plot(x_train[0][:640], color="red")
# plt.subplot(2,2,2, title="x_train_before")
# plt.plot(x_train[0][640:], color="red")
# plt.subplot(2,2,3, title="x_train_resh_after")
# plt.plot(x_train_resh[0][0].T, color="blue")
# plt.subplot(2,3,4, title="x_train_resh_after")
# plt.plot(x_train_resh[0][1].T, color="blue")
# plt.show()

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#%%
#Convolution Neural Network
# [samples, time steps, features].
# real_x_train = x_train.reshape(14808, 640, 2)
# real_x_test = x_test.reshape(3703, 640, 2)
learning_rate = 1e-3 # default 1e-3
drop_rate = 0.3

loss = tf.keras.losses.categorical_crossentropy  #tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam(lr=learning_rate) #tf.keras.optimizers.Adam(lr=learning_rate) tf.keras.optimizers.SGD(learning_rate=learning_rate)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=25, kernel_size=(2,2), activation='relu', padding= "same", input_shape=(640, 2, 1)))
model.add(tf.keras.layers.AvgPool2D(pool_size=1))
model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(2,2), activation='relu', padding= "same"))
model.add(tf.keras.layers.AvgPool2D(pool_size=1))
model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(2,2), activation='relu', padding= "same"))
model.add(tf.keras.layers.AvgPool2D(pool_size=1))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(drop_rate))
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
model.summary()
history = model.fit(x_train_resh, y_train, epochs=100, batch_size=5, validation_data=(x_test_resh, y_test))


print(model.predict(x_test_resh[:4]))
print(y_test[:4])