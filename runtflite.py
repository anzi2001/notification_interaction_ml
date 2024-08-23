import tensorflow as tf
import numpy as np
import random
from data import import_files, filter_device, preprocess_data
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def run_tflite():
    interpreter = tf.lite.Interpreter("model.tflite")
    interpreter.allocate_tensors()
    my_signature = interpreter.get_signature_runner("predict")
    files = import_files()
    notification_list = filter_device(files)
    notification_list = [item for sublist in notification_list for item in sublist]
    random.shuffle(notification_list)
    notification_labels = np.array([notification.has_user_interacted() for notification in notification_list], dtype=np.float64)
    test_data = [item.to_rf_array() for item in notification_list]
    test_data, test_labels, train_data, train_labels = preprocess_data(notification_list, notification_list)

    output = my_signature(x = test_data)
    probabilities = output["output"]
    print(probabilities)
    correct = 0
    for i in range(len(probabilities)):
        if (probabilities[i] > 0.5 and notification_labels[i] == 1) or (probabilities[i] <= 0.5 and notification_labels[i] == 0):
            correct += 1
    print(f"Accuracy: {correct/len(probabilities)}")
    print(f"Manual evaluation: {np.average(notification_labels)}")
    print(f"Manual evaluation: {1 - np.average(notification_labels)}")

if __name__ == "__main__":
    run_tflite()

    