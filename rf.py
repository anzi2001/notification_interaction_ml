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
    labels = process_labels(data)

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.25, random_state=42, stratify=labels)

    scaler, encoder = fit_data(train_data)
    train_data = preprocess_data(train_data, scaler, encoder)
    test_data = preprocess_data(test_data, scaler, encoder)

    rf = RandomForestClassifier(max_depth=10, n_estimators=50, max_features=3)
    rf.fit(train_data, train_labels)
    print("RF Accuracy:", accuracy_score(test_labels, rf.predict(test_data)))

    svm = SVC()
    svm.fit(train_data, train_labels)
    print("SVM Accuracy:", accuracy_score(test_labels, svm.predict(test_data)))

    print("Baseline Accuracy:", np.average(test_labels))