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
from datetime import datetime

#os.environ["TF_USE_LEGACY_KERAS"] = "1"

categorical_features = [ "prox", "screen",'volume', "battery", "location"]
numerical_features = []

NUM_FEATURES = 19

@gen.tflite_model_class
class TfLiteModel(gen.BaseTFLiteModel):
    X_SHAPE = [NUM_FEATURES]
    Y_SHAPE = [2]

    def __init__(self):
        inputs = tf.keras.layers.Input(shape=(NUM_FEATURES,))
        model = tf.keras.layers.Dense(NUM_FEATURES, activation='relu')(inputs)
        model = tf.keras.layers.Dense(128, activation='relu')(model)
        model = tf.keras.layers.Dropout(0.6)(model, training=True)
        model = tf.keras.layers.Dense(256, activation='relu')(model)
        model = tf.keras.layers.Dropout(0.6)(model, training=True)
        model = tf.keras.layers.Dense(64, activation='relu')(model)
        model = tf.keras.layers.Dropout(0.6)(model, training=True)
        model = tf.keras.layers.Dense(2, activation='sigmoid')(model)
        self.model = tf.keras.models.Model(inputs=inputs, outputs=model)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

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
    data = load_data()
    data = merge_data(*data)
    
    labels = process_labels(data).map(lambda x: [x, 1-x]).to_list()
    labels = np.array(labels)
    scaler, encoder = fit_data(data, categorical_features, numerical_features)
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.25, random_state=42
    )

    train_data = preprocess_data(train_data, scaler, encoder, categorical_features, numerical_features)
    test_data = preprocess_data(test_data, scaler, encoder, categorical_features, numerical_features)
    print(train_data.shape)



    if len(sys.argv) > 1 and sys.argv[1] == "save":
        model = TfLiteModel()
        gen.save_model(model, "saved_model")
        tflite_model = gen.convert_saved_model("saved_model")
        gen.save_tflite_model(tflite_model, "model.tflite")
        exit()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tensorflow", histogram_freq=1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(train_data.shape[1],)),
        tf.keras.layers.Dense(train_data.shape[1], activation='relu'),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.6),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.6),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.6),

        tf.keras.layers.Dense(1, activation='sigmoid')
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

