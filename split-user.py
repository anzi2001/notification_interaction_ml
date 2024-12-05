from data import *
from sklearn.model_selection import train_test_split
import sys
import os
import json

categorical_features = ["prox", "screen",'volume', "battery", "location"]
numerical_features = []

def split_user(split_num):
    data = load_data()
    data = merge_data(*data)

    encoder = categorical_fit(data, categorical_features)

    i = 0
    os.mkdir("users/train")
    os.mkdir("users/test")
    for device, deviceData in data.groupby("device_id"):
        labels = process_labels(deviceData)

        if("ble_cluster" in numerical_features or "wifi_cluster" in numerical_features):
            ble_cluster, wifi_cluster = cluster(deviceData)
            print(len(deviceData), len(ble_cluster), len(wifi_cluster))
            deviceData["ble_cluster"] = ble_cluster
            deviceData["wifi_cluster"] = wifi_cluster

        scaler = numerical_fit(deviceData, numerical_features)

        
        train_data, test_data, train_labels, test_labels = train_test_split(
            deviceData, labels, test_size=0.30, random_state=42
        )

        preprocessed_train_data = preprocess_data(train_data, scaler, encoder, categorical_features, numerical_features)
        preprocessed_test_data = preprocess_data(test_data, scaler, encoder, categorical_features, numerical_features)


        train_data["preprocessed_data"] = list(preprocessed_train_data)
        test_data["preprocessed_data"] = list(preprocessed_test_data)

        print(f"Device: {device}, Notification count: {len(deviceData)}")
            

        with open(f"users/partition_{i}_test.txt", "w") as f:
            f.write(f"test/{i}_test.json")
        with open(f"users/partition_{i}_train.txt", "w") as f:
            f.write(f"train/{i}_train.json")
        with open(f"users/train/{i}_train.json", "w") as f:
            train_data.to_json(path_or_buf=f,orient='records')
        with open(f"users/test/{i}_test.json", "w") as f:
            test_data.to_json(path_or_buf=f,orient='records')
        i += 1          
        print()


if __name__ == "__main__":
    #get number of users to split
    if sys.argv[1] == "print":
        files = os.listdir("users/train")
        files = sorted(files)
        for file in files:
            with open(f"users/train/{file}", "r") as f:
                data = pd.read_json(f)
                print(f"User: {file}, Notification count: {len(data)}")
        exit()

    if len(sys.argv) > 1:
        num_users = int(sys.argv[1])
    else:
        num_users = 19
    

    split_user(num_users)
    print("Done!")