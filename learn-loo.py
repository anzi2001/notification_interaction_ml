import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.svm import SVC
from data import *
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

categorical_features = ['prox', 'screen', 'volume', 'battery', 'location']
numerical_features = ["hour"]

if __name__ == "__main__":
    data = load_data()
    data = merge_data(*data)
    baseline_acc = []
    ml_eval = []
    rf_eval = []
    svm_eval = []
    for device, deviceData in data.groupby("device_id"):
        notification_data = data[data["device_id"] != device]
        print(f"Device: {device}, LOO Notification count: {len(deviceData)}")
        
        train_ml_labels = notification_data["action"].map(lambda notification: int(notification == "Clicked"))
        loo_ml_labels = deviceData["action"].map(lambda notification: int(notification == "Clicked"))

        scaler, encoder = fit_data(notification_data, categorical_features, numerical_features)
        train_data = preprocess_data(notification_data, scaler, encoder, categorical_features, numerical_features)
        test_data = preprocess_data(deviceData, scaler, encoder, categorical_features, numerical_features)


        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(train_data.shape[1],)),
            tf.keras.layers.Dense(train_data.shape[1], activation='relu'),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.6),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.6),

            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        stop_early = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=["accuracy", tf.keras.metrics.Precision(),tf.keras.metrics.Recall(), tf.keras.metrics.F1Score()])
        model.fit(train_data, train_ml_labels, epochs=100, batch_size=32, validation_split=0.2, callbacks=[stop_early])
        rf = RandomForestClassifier( n_estimators=300, max_features=3, max_depth=3)
        rf.fit(train_data, train_ml_labels)
        svm = SVC(C=3)
        svm.fit(train_data, train_ml_labels)
        train_evaluation = [train_ml_labels.mean(), 1 - train_ml_labels.mean()]
        loo_evaluation = [loo_ml_labels.mean(), 1 - loo_ml_labels.mean()]
        acc = loo_evaluation[train_evaluation.index(max(train_evaluation))]
        baseline_acc.append(acc)
        ml_eval.append(model.evaluate(test_data, loo_ml_labels))
        rf_eval.append([accuracy_score(loo_ml_labels, rf.predict(test_data)), precision_score(loo_ml_labels, rf.predict(test_data)), recall_score(loo_ml_labels, rf.predict(test_data)), f1_score(loo_ml_labels, rf.predict(test_data))])
        svm_eval.append([accuracy_score(loo_ml_labels, svm.predict(test_data)), precision_score(loo_ml_labels, svm.predict(test_data)), recall_score(loo_ml_labels, svm.predict(test_data)), f1_score(loo_ml_labels, svm.predict(test_data))])

    print("---AVERAGES---")
    print(f"Baseline accuracy: {np.average(baseline_acc)}")
    print()
    print(f"ML accuracy: {np.average([eval[1] for eval in ml_eval])}")
    print(f"ML precision: {np.average([eval[2] for eval in ml_eval])}")
    print(f"ML recall: {np.average([eval[3] for eval in ml_eval])}")
    print(f"ML f1: {np.average([eval[4] for eval in ml_eval])}")
    print()
    print(f"RF accuracy: {np.average([eval[0] for eval in rf_eval])}")
    print(f"RF precision: {np.average([eval[1] for eval in rf_eval])}")
    print(f"RF recall: {np.average([eval[2] for eval in rf_eval])}")
    print(f"RF f1: {np.average([eval[3] for eval in rf_eval])}")
    print()
    print(f"SVM accuracy: {np.average([eval[0] for eval in svm_eval])}")
    print(f"SVM precision: {np.average([eval[1] for eval in svm_eval])}")
    print(f"SVM recall: {np.average([eval[2] for eval in svm_eval])}")
    print(f"SVM f1: {np.average([eval[3] for eval in svm_eval])}")
