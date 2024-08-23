import tensorflow as tf
import numpy as np
import random
import sys
from data import *
from learn.dense import create_dense, train_dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from learn.plot import show_history
import gen_tflite as gen
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

NUM_FEATURES = 14

@gen.tflite_model_class
class TfLiteModel(gen.BaseTFLiteModel):
    X_SHAPE = [NUM_FEATURES]
    Y_SHAPE = [2]

    def __init__(self):
        self.model = create_dense(NUM_FEATURES)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])



if __name__ == "__main__":
    files = import_files()
    params = ('activity', 'battery_status', 'location', 'locked', 'volume_mode')
    notification_list = filter_device(files, params)
    notification_list = [item for sublist in notification_list for item in sublist]
    random.shuffle(notification_list)
    notification_labels = np.array([(notification.has_user_interacted(), not notification.has_user_interacted()) for notification in notification_list], dtype=np.float32)
    ohe = OneHotEncoder(sparse_output=False)
    print(len(notification_list))
    print(np.average([not notification.has_user_interacted() for notification in notification_list]))
    notification_data = ohe.fit_transform([item[params] for item in notification_list])
    train_data, test_data, train_labels, test_labels = train_test_split(notification_data, notification_labels, test_size=0.3, stratify=notification_labels)

    layer = tf.keras.layers.Normalization(axis=-1)
    layer.adapt(train_data)

    if len(sys.argv) > 1 and sys.argv[1] == "save":
        model = TfLiteModel()
        train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(8)
        for epoch in range(10):
            num_batches = 0
            epoch_loss = 0
            for x, y in train_ds:
                epoch_loss = model.train(x, y)
                num_batches += 1
            print(f"Epoch {epoch} Loss: {epoch_loss}")
        
        gen.save_model(model, "saved_model")
        tflite_model = gen.convert_saved_model("saved_model")
        gen.save_tflite_model(tflite_model, "model.tflite")

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(train_data.shape[1],)),
        layer,
        tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu),
        tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    model, history = train_dense(model, train_data, train_labels)
    #show_history(history)

    print("---EVALUATION---")
    model.evaluate(test_data, test_labels, verbose=2)
    for i in range(10):
        print(model.predict(np.array([test_data[i]])))
        print(test_labels[i])
    print("---MANUAL EVALUATION---")
    print(np.average([label[0] for label in test_labels]))
    print(np.average([label[1] for label in test_labels]))

