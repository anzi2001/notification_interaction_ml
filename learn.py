import tensorflow as tf
import numpy as np
import random
import sys
from data import *
from learn.dense import create_dense, train_dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import gen_tflite as gen

#os.environ["TF_USE_LEGACY_KERAS"] = "1"

NUM_FEATURES = 14

@gen.tflite_model_class
class TfLiteModel(gen.BaseTFLiteModel):
    X_SHAPE = [NUM_FEATURES]
    Y_SHAPE = [2]

    def __init__(self):
        self.model = create_dense(NUM_FEATURES)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

encoder = OneHotEncoder(sparse_output=False)
ordinal = OrdinalEncoder(categories=[["Other","Work","Home"]])
scaler = MinMaxScaler()

def modify_attribs(data):
    location = [row.location for row in data]
    volume_mode = [row.volume_mode for row in data]
    activity = [row.activity for row in data]
    locked = [int(not row.locked) * 2 for row in data]

    interaction_time = [ datetime.fromtimestamp(row.interaction_time / 1000).hour for row in data]
    interaction_time = scaler.transform(np.array(interaction_time).reshape(-1, 1)).flatten()
    
    ordinal_data = ordinal.transform(np.array(location).reshape(-1, 1))
    categorical_data = np.array([volume_mode, activity]).T
    categorical_encoded = encoder.transform(categorical_data)
    print(location)
    print(ordinal_data)

    result = []
    for i in range(len(data)):
        result.append(np.concatenate((
            [interaction_time[i], locked[i]],
            categorical_encoded[i],
            ordinal_data[i], 
        )).tolist())

    return result


if __name__ == "__main__":
    files = import_files()
    params = ('activity', 'location', 'volume_mode', "locked", 'interaction_time')
    notification_list = filter_device(files, params)
    notification_list = [item for sublist in notification_list for item in sublist]
    encoder.fit(np.array([[notification.volume_mode, notification.activity] for notification in notification_list]))
    if "location" in params:
        ordinal.fit(np.array([notification.location for notification in notification_list]).reshape(-1, 1))
    if "interaction_time" in params:
        scaler.fit(np.array([datetime.fromtimestamp(notification.interaction_time / 1000).hour for notification in notification_list]).reshape(-1, 1))

    notification_labels = np.array([[notification.has_user_interacted(), not notification.has_user_interacted()] for notification in notification_list])

    train_data, test_data, train_labels, test_labels = train_test_split(
        notification_list, notification_labels, test_size=0.25, random_state=42
    )
    
    train_data = np.array(modify_attribs(train_data))
    test_data = np.array(modify_attribs(test_data))

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

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tensorflow", histogram_freq=1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(train_data.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])
    print(train_data)

    model, history = train_dense(model, train_data, train_labels, tensorboard_callback)
    #show_history(history)

    print("---EVALUATION---")
    model.evaluate(test_data, test_labels, verbose=2)
    for i in range(10):
        print(model.predict(np.array([test_data[i]])))
        print(test_labels[i])
    print("---MANUAL EVALUATION---")
    print(np.average([label[0] for label in test_labels]))
    print(np.average([label[1] for label in test_labels]))

