import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from data import *

if __name__ == "__main__":
    data = load_data()
    data = merge_data(*data)

    #exclude random device_id from the data
    device = random.choice(data["device_id"].unique())
    loo_data = data[data["device_id"] == device]
    data = data[data["device_id"] != device]
    labels = process_labels(data)


    loo_labels = process_labels(loo_data)

    scaler, encoder = fit_data(data)
    train_data = preprocess_data(data, scaler, encoder)
    test_data = preprocess_data(loo_data, scaler, encoder)

    rf = RandomForestClassifier(max_depth=10, n_estimators=50, max_features=3)
    rf.fit(train_data, labels)
    print("RF Accuracy:", accuracy_score(loo_labels, rf.predict(test_data)))

    svm = SVC()
    svm.fit(train_data, labels)
    print("SVM Accuracy:", accuracy_score(loo_labels, svm.predict(test_data)))

    print("Baseline Accuracy:", np.average(loo_labels))