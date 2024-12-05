import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score
from data import *

def show_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


TIME_DIFF = 33000

categorical_features = ['volume', "prox", "screen", "battery", "location", "activity"]
numerical_features = ["mean_crossing_rate", "variance", "peak", "mean", "spectral_entropy", "ble_cluster", "wifi_cluster"]

if __name__ == "__main__":
    #create a random forest classifier for each user
    data = load_data()
    data = merge_data(*data)
    labels = process_labels(data)

    baseline_acc = []
    rf_acc = []
    svm_acc = []

    encoder = categorical_fit(data, categorical_features)

    for device, deviceData in data.groupby("device_id"):

        labels = process_labels(deviceData)

        ble_cluster, wifi_cluster = cluster(deviceData)
        print(len(deviceData), len(ble_cluster), len(wifi_cluster))
        deviceData["ble_cluster"] = ble_cluster
        deviceData["wifi_cluster"] = wifi_cluster

        scaler = numerical_fit(deviceData, numerical_features)
        
        train_data, test_data, train_labels, test_labels = train_test_split(
            deviceData, labels, test_size=0.30, random_state=42
        )


        train_data = preprocess_data(train_data, scaler, encoder, categorical_features, numerical_features)
        test_data = preprocess_data(test_data, scaler, encoder, categorical_features, numerical_features)


        print("Feature count:", train_data.shape)

        rf = RandomForestClassifier(max_depth=3, n_estimators=300, max_features=3)
        svm = SVC(C=3)
        svm.fit(train_data, train_labels)
        rf.fit(train_data, train_labels)
        baseline_acc.append(np.mean(labels))
        print(f"Device: {device}, RF Accuracy: {accuracy_score(test_labels, rf.predict(test_data))}")
        #print(f"Device: {device}, RF Precision: {precision_score(test_labels, rf.predict(test_data))}")
        print(f"Device: {device}, Baseline Accuracy: {max(np.mean(test_labels), 1 - np.mean(test_labels))}")

        print(f"Device: {device}, SVM Accuracy: {accuracy_score(test_labels, svm.predict(test_data))}")
        #print(f"Device: {device}, SVM Precision: {precision_score(test_labels, svm.predict(test_data))}")
        rf_acc.append(accuracy_score(test_labels, rf.predict(test_data)))
        svm_acc.append(accuracy_score(test_labels, svm.predict(test_data)))
                       
        print()

    print("baseline accuracy:", np.average(baseline_acc))
    print(f"Average RF Accuracy: {np.average(rf_acc)}")
    print(f"Average SVM Accuracy: {np.average(svm_acc)}")

