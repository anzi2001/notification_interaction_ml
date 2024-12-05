import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC
from data import *


categorical_features = ['volume', "prox", "screen", "battery", "location", "activity","ble_cluster", "wifi_cluster" ]
numerical_features = ["mean_crossing_rate", "variance", "peak", "mean","spectral_entropy", ]

if __name__ == "__main__":
    data = load_data()
    data = merge_data(*data)

    baselineAcc = []
    mlEval = []
    rfEval = []
    svmEval = []

    for device,deviceData in data.groupby("device_id"):
        print(f"Device: {device}, Notification count: {len(deviceData)}")
        if len(deviceData) < 4:
            continue

        labels = process_labels(deviceData)

        ble_cluster, wifi_cluster = cluster(deviceData)

        print(len(deviceData), len(ble_cluster), len(wifi_cluster))

        deviceData["ble_cluster"] = ble_cluster
        deviceData["wifi_cluster"] = wifi_cluster

        scaler, encoder = fit_data(deviceData, categorical_features, numerical_features)

        train_data, test_data, train_labels, test_labels = train_test_split(
            deviceData, labels, test_size=0.25, random_state=42
        )

        train_data = preprocess_data(train_data, scaler, encoder, categorical_features, numerical_features)
        test_data = preprocess_data(test_data, scaler, encoder, categorical_features, numerical_features)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(train_data.shape[1],)),
            tf.keras.layers.Dense(train_data.shape[1], activation='relu'),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.7),

            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.7),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.7),

            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        rf = RandomForestClassifier(n_estimators=300, max_depth=3, max_features=3)
        rf.fit(train_data, train_labels)
        svm = SVC(C=3)
        svm.fit(train_data, train_labels)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.F1Score()])
        history = model.fit(train_data, train_labels, epochs=70, batch_size=16, validation_split=0.2)
        print("---EVALUATION---")
        #for i in range(len(test_data)):
        #    print(f"Predicted: {model.predict(np.array([test_data[i]], dtype=float))} Actual: {test_labels[i]}")
        loss, acc, prec, recall, f1 =  model.evaluate(test_data, test_labels, verbose=2)
        mlEval.append([acc, prec, recall, f1])
        rfEval.append([rf.score(test_data, test_labels), precision_score(test_labels, rf.predict(test_data), average='binary'), recall_score(test_labels, rf.predict(test_data), average='binary'), f1_score(test_labels, rf.predict(test_data), average='binary')])
        svmEval.append([svm.score(test_data, test_labels), precision_score(test_labels, svm.predict(test_data), average='binary'), recall_score(test_labels, svm.predict(test_data), average='binary'), f1_score(test_labels, svm.predict(test_data), average='binary')])
        baselineAcc.append(np.mean(test_labels))
        print()

    print("Baseline accuracy:", np.average(baselineAcc))
    print()
    print("ML accuracy:", np.average([eval[0] for eval in mlEval]))
    print("ML precision:", np.average([eval[1] for eval in mlEval]))
    print("ML recall:", np.average([eval[2] for eval in mlEval]))
    print("ML f1:", np.average([eval[3] for eval in mlEval]))
    print()
    print("RF accuracy:", np.average([eval[0] for eval in rfEval]))
    print("RF precision:", np.average([eval[1] for eval in rfEval]))
    print("RF recall:", np.average([eval[2] for eval in rfEval]))
    print("RF f1:", np.average([eval[3] for eval in rfEval]))
    print()
    print("SVM accuracy:", np.average([eval[0] for eval in svmEval]))
    print("SVM precision:", np.average([eval[1] for eval in svmEval]))
    print("SVM recall:", np.average([eval[2] for eval in svmEval]))
    print("SVM f1:", np.average([eval[3] for eval in svmEval]))

    
