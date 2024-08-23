import numpy as np
import random
from data import *
from learn.dense import train_dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    files = import_files()
    baseline_acc = []
    inv_baseline_acc = []
    rf_acc = []
    ml_acc = []
    filtered_users = filter_dict(files)
    for device in filtered_users.keys():
        notification_list, loo_device = remove_user(filtered_users, device)
        print(f"Device: {device} Num devices: {len(notification_list)}, device ids: {notification_list.keys()} LOO Notification count: {len(loo_device)}")
        
        train_device_list = filter_device(notification_list)
        train_list = [item for sublist in train_device_list for item in sublist]    
        loo_list = filter_device_list(loo_device)
        if len(loo_list) == 0 or len(train_list) == 0:
            continue
        random.shuffle(train_list)
        random.shuffle(loo_list)
        train_ml_labels = np.array([(notification.has_user_interacted(), not notification.has_user_interacted()) for notification in train_list], dtype=np.float32)
        loo_ml_labels = np.array([(notification.has_user_interacted(), not notification.has_user_interacted()) for notification in loo_list], dtype=np.float32)

        train_data, train_labels, loo_data, loo_labels = preprocess_data(train_list, loo_list)

        #model, history = train_dense(train_data, train_ml_labels)
        #rf = RandomForestClassifier(max_depth=3, max_features=3, n_estimators=300)
        #rf.fit(train_data, [notification.has_user_interacted() for notification in train_list])
        train_evaluation = [np.average([notification.has_user_interacted() for notification in train_list]), 1 - np.average([notification.has_user_interacted() for notification in train_list])]
        loo_evaluation = [np.average([notification.has_user_interacted() for notification in loo_list]), 1 - np.average([notification.has_user_interacted() for notification in loo_list])]
        acc = loo_evaluation[train_evaluation.index(max(train_evaluation))]
        baseline_acc.append(acc)
        inv_baseline_acc.append(1 - acc)
        #rf_acc.append(rf.score(loo_data, [notification.has_user_interacted() for notification in loo_list]))
        #ml_acc.append(model.evaluate(loo_data, loo_ml_labels, verbose=2)[1])

        print("---MANUAL EVALUATION---")
        print(np.average([notification.has_user_interacted() for notification in loo_list]))
        print(1 - np.average([notification.has_user_interacted() for notification in loo_list]))
    
    print(f"Baseline accuracy: {np.average(baseline_acc)}")
    print(f"Baseline inverse accuracy: {np.average(inv_baseline_acc)}")
    print(f"RF accuracy: {np.average(rf_acc)}")
    print(f"ML accuracy: {np.average(ml_acc)}")

