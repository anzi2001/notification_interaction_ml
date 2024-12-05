import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.cluster import DBSCAN
import numba as nb

def load_data():
    activity = pd.read_csv('data/Activity.tab', sep='\t')
    notifications = pd.read_csv('data/ProcessedNotif.tab', sep='\t', usecols=lambda x: x != "index")
    location = pd.read_csv('data/Location.tab', sep='\t')
    proximity = pd.read_csv('data/Proximity.tab', sep='\t')
    screen = pd.read_csv('data/Screen.tab', sep='\t')
    volume = pd.read_csv('data/Volume.tab', sep='\t')
    light = pd.read_csv('data/Light.tab', sep='\t')
    battery = pd.read_csv('data/Battery.tab', sep='\t')
    wifi = pd.read_csv('data/Wifi.tab', sep='\t')
    bluetooth = pd.read_csv('data/Bluetooth.tab', sep='\t')
    acceleration = pd.read_csv('data/Accelerometer.tab', sep='\t')
    return activity, notifications, location, proximity, screen, volume, light, battery, wifi, bluetooth, acceleration

# Merge data based on common keys
def merge_data(activity, notifications, location, proximity, screen, volume, light, battery, wifi, bluetooth, acceleration):
    data = notifications.merge(activity, on=['device_id', 'notification_id'], how='left')
    data = data.merge(location, on=['device_id', 'notification_id'], how='left')
    data = data.merge(proximity, on=['device_id', 'notification_id'], how='left')
    data = data.merge(screen, on=['device_id', 'notification_id'], how='left')
    data = data.merge(volume, on=['device_id', 'notification_id'], how='left')
    wifi = wifi.groupby(["device_id", 'notification_id'],as_index=False)['ssid'].agg(lambda x: ' '.join(x))
    data = data.merge(wifi, on=['device_id', 'notification_id'], how='left')
    bluetooth = bluetooth.groupby(["device_id", 'notification_id'],as_index=False)['name'].agg(lambda x: ' '.join(x))
    data = data.merge(bluetooth, on=['device_id', 'notification_id'], how='left')
    data = data.merge(acceleration, on=['device_id', 'notification_id'], how='left')
    data = data.merge(light, on=['device_id', 'notification_id'], how='left')
    data = data.merge(battery, on=['device_id', 'notification_id'], how='left')

    data.drop_duplicates(subset=['notification_id'],inplace=True)
    data = data[(data["action"] == "Clicked") | (data["action"] == "Removed")]
    data = data.fillna({"mean_crossing_rate": 0, "variance": 0, "peak": 0, "mean": 0, "energy1": 0, "energy2": 0, "energy3": 0, "energy4": 0, "energy_ratio1": 0, "energy_ratio2": 0, "energy_ratio3": 0, "spectral_entropy": 0})
    data = data.fillna({"ssid": "", "name": ""})
    
    data.to_csv("data/merged_data.csv")
    return data

def categorical_fit(data, features):
    encoder = OneHotEncoder()
    encoder.fit(data[features])
    return encoder

def numerical_fit(data, features):
    scaler = MinMaxScaler()
    if len(features) > 0:
        scaler.fit(data[features])
    return scaler

def fit_data(data, features, num_features):
    scaler = numerical_fit(data, num_features)
    encoder = categorical_fit(data, features)
    return scaler, encoder

def process_labels(data):
    return data['action'].apply(lambda x: 1 if x == 'Clicked' else 0)


@nb.jit(nopython=True, cache=True)
def jaccard_similarity(str1, str2):
    if str1 == "" or str2 == "":
        return 1
    set1 = set(str1.split())
    set2 = set(str2.split())
    jaccard_similarity = len(set1.intersection(set2)) / len(set1.union(set2))
    return 1 - jaccard_similarity

# Create a similarity matrix
@nb.jit(nopython=True, cache=True)
def create_similarity_matrix(featureList):
    n = len(featureList)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            similarity = jaccard_similarity(featureList[i], featureList[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    return similarity_matrix


def cluster(data):
    matrix = create_similarity_matrix(data["name"].to_list())
    dbscan = DBSCAN(metric='precomputed', eps=0.6, min_samples=2)
    ble_cluster = dbscan.fit_predict(matrix)
    wifi_matrix = create_similarity_matrix(data["ssid"].to_list())
    wifi_cluster = dbscan.fit_predict(wifi_matrix)

    top_n_clusters = 6
    unique, counts = np.unique(ble_cluster, return_counts=True)
    counts = dict(zip(unique, counts))
    counts.pop(-1, None)
    top_ble_clusters = sorted(counts, key=counts.get, reverse=True)[:top_n_clusters]
    ble_cluster = np.where(np.isin(ble_cluster, top_ble_clusters), ble_cluster, -1)
    unique, counts = np.unique(wifi_cluster, return_counts=True)
    counts = dict(zip(unique, counts))
    counts.pop(-1, None)

    top_wifi_clusters = sorted(counts, key=counts.get, reverse=True)[:top_n_clusters]
    wifi_cluster = np.where(np.isin(wifi_cluster, top_wifi_clusters), wifi_cluster, -1)
    #label the top n clusters with the highest label, top should have 6, 5, 4, 3, 2, 1
    for i in range(len(top_ble_clusters)):
        ble_cluster = np.where(ble_cluster == top_ble_clusters[i], top_n_clusters - i, ble_cluster)

    for i in range(len(top_wifi_clusters)):
        wifi_cluster = np.where(wifi_cluster == top_wifi_clusters[i], top_n_clusters - i, wifi_cluster)

    return ble_cluster, wifi_cluster




# Preprocess data
def preprocess_data(data, scaler, encoder, categorical_features, numerical_features):
    print(encoder.categories_)
    # Encode categorical features
    encoded_features = encoder.transform(data[categorical_features]).toarray()

    
    # Scale numerical features
    if len(numerical_features) > 0:
        scaled_features = scaler.transform(data[numerical_features])
    else:
        #make empty 2d array if no numerical features
        scaled_features = np.empty((data.shape[0], 0))
    #print(matrix)
    
    # Combine features
    features = np.hstack((encoded_features, scaled_features))
    
    return features