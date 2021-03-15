import tensorflow as tf
class HopefullModel(tf.keras.Model):
    def __init__(self):
        super(HopefullModel, self).__init__()
         #Define layers
        self.kernel_size_0 = 20
        self.kernel_size_1 = 6
        self.drop_rate = 0.5

        self.conv1 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding= "same",
                                            input_shape=(640, 2))

        self.conv2 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding= "valid")

        self.batch_n_1 = tf.keras.layers.BatchNormalization()

        self.spatial_drop_1 = tf.keras.layers.SpatialDropout1D(self.drop_rate)

        self.conv3 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding= "valid")

        self.spatial_drop_2 = tf.keras.layers.SpatialDropout1D(self.drop_rate)

        self.conv4 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding= "valid")

        self.spatial_drop_3 = tf.keras.layers.SpatialDropout1D(self.drop_rate)

        self.flat = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(256, activation='relu')

        self.dropout1 = tf.keras.layers.Dropout(self.drop_rate)

        self.dense2 = tf.keras.layers.Dense(64, activation='relu')

        self.dropout2 = tf.keras.layers.Dropout(self.drop_rate)

        self.dense3 = tf.keras.layers.Dense(5, activation='softmax')

    def call(self, input_tensor):
        conv1 = self.conv1(input_tensor)
        conv2 = self.conv2(conv1)
        batch_n_1 = self.batch_n_1(conv2)
        spatial_drop_1 = self.spatial_drop_1(batch_n_1)
        conv3 = self.conv3(spatial_drop_1)
        spatial_drop_2 = self.spatial_drop_2(conv3)
        conv4 = self.conv4(spatial_drop_2)
        spatial_drop_3 = self.spatial_drop_3(conv4)
        flat = self.flat(spatial_drop_3)
        dense1 = self.dense1(flat)
        dropout1 = self.dropout1(dense1)
        dense2 = self.dense2(dropout1)
        dropout2 = self.dropout2(dense2)
        return self.dense3(dropout2)