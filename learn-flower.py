import tensorflow as tf
import numpy as np
from data import *


categorical_features = [ "prox", "screen",'volume', "battery", "location"]
numerical_features = []

NUM_FEATURES = 19

device_list = ["f388a39a209144198f1312861dab3105", "ef24e3578f264168ae5adfb7e75e4cea", "db9a6313e6e84616b9be27054a2094b2", "d143f2f107854051bfc9e2cbe8c913b5", "adf653615f6644e5a2feeabf5f7ac675", "adf5c3a65c8e4d0abc900fccf2b665fe", "94f50e4b32c14cc287c6165e4681cd6b"]
train_list = ["3ac8550c4587483290c57e5e98d6ee05", "43fd6ac3ba544d0aa99bb50a266d1f76", "6dbcd91158e248bea04dd5dd7c81e76e", "6dbd5e5e6c6040938ed458705ece452e", "6fcc5b3f284642e7b91d8c16df97167a"]

if __name__ == "__main__":
    data = load_data()
    data = merge_data(*data)
    test_data = data[data["device_id"].isin(device_list)]
    train_data = data[data["device_id"].isin(train_list)]
    
    train_labels = process_labels(train_data).map(lambda x: [float(x), 1-x]).to_list()
    test_labels = process_labels(test_data).map(lambda x: [float(x), 1-x]).to_list()
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    scaler, encoder = fit_data(data, categorical_features, numerical_features)

    train_data = preprocess_data(train_data, scaler, encoder, categorical_features, numerical_features)
    test_data = preprocess_data(test_data, scaler, encoder, categorical_features, numerical_features)
    print(train_data.shape)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(NUM_FEATURES,)),
        tf.keras.layers.Dense(NUM_FEATURES, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy',tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.F1Score()])

    stop_early = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, min_delta=0.001,)
    model.fit(train_data, train_labels, epochs=50, batch_size=16, validation_split=0.2, callbacks=[stop_early])
    #show_history(history)

    print("---EVALUATION---")
    model.evaluate(test_data, test_labels, verbose=2)
    print("---MANUAL EVALUATION---")
    print(test_labels)
    print(np.average([label[0] for label in test_labels]))
    print(np.average([label[1] for label in test_labels]))

