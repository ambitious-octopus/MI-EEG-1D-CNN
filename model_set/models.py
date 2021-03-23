import tensorflow as tf

class HopefullNet(tf.keras.Model):
    def __init__(self):
        super(HopefullNet, self).__init__()
         #Define layers
        self.kernel_size_0 = 20
        self.kernel_size_1 = 6
        self.drop_rate = 0.5

        self.conv1 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding= "same",
                                            input_shape=(640, 2))

        self.batch_n_1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_0,
                                            activation='relu',
                                            padding= "valid")

        self.batch_n_2 = tf.keras.layers.BatchNormalization()

        self.spatial_drop_1 = tf.keras.layers.SpatialDropout1D(self.drop_rate)

        self.conv3 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding= "valid")
        self.avg_pool1 = tf.keras.layers.AvgPool1D(pool_size=2)
        self.conv4 = tf.keras.layers.Conv1D(filters=32,
                                            kernel_size=self.kernel_size_1,
                                            activation='relu',
                                            padding= "valid")

        self.spatial_drop_2 = tf.keras.layers.SpatialDropout1D(self.drop_rate)

        self.flat = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(296, activation='relu')

        self.dropout1 = tf.keras.layers.Dropout(self.drop_rate)

        self.dense2 = tf.keras.layers.Dense(148, activation='relu')

        self.dropout2 = tf.keras.layers.Dropout(self.drop_rate)

        self.dense3 = tf.keras.layers.Dense(74, activation='relu')

        self.dropout3 = tf.keras.layers.Dropout(self.drop_rate)

        self.out = tf.keras.layers.Dense(5, activation='softmax')

    def call(self, input_tensor):
        conv1 = self.conv1(input_tensor)
        batch_n_1 = self.batch_n_1(conv1)
        conv2 = self.conv2(batch_n_1)
        batch_n_2 = self.batch_n_2(conv2)
        spatial_drop_1 = self.spatial_drop_1(batch_n_2)
        conv3 = self.conv3(spatial_drop_1)
        avg_pool1 = self.avg_pool1(conv3)
        conv4 = self.conv4(avg_pool1)
        spatial_drop_2 = self.spatial_drop_2(conv4)
        flat = self.flat(spatial_drop_2)
        dense1 = self.dense1(flat)
        dropout1 = self.dropout1(dense1)
        dense2 = self.dense2(dropout1)
        dropout2 = self.dropout2(dense2)
        return self.out(dropout2)