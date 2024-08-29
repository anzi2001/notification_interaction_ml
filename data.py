import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    return activity, notifications, location, proximity, screen, volume, wifi, light, battery

# Merge data based on common keys
def merge_data(activity, notifications, location, proximity, screen, volume, wifi, light, battery):
    data = notifications.merge(activity, on=['device_id', 'notification_id'], how='left')
    data = data.merge(location, on=['device_id', 'notification_id'], how='left')
    data = data.merge(proximity, on=['device_id', 'notification_id'], how='left')
    data = data.merge(screen, on=['device_id', 'notification_id'], how='left')
    data = data.merge(volume, on=['device_id', 'notification_id'], how='left')
    data = data.merge(wifi, on=['device_id', 'notification_id'], how='left')
    data = data.merge(light, on=['device_id', 'notification_id'], how='left')
    data = data.merge(battery, on=['device_id', 'notification_id'], how='left')


    data.dropna(inplace=True)
    data = data[data["response_time"] > 0]
    data = data[(data["action"] == "Clicked") | (data["action"] == "Removed")]

    return data

categorical_features = ['location', 'volume', 'activity', "battery"]
numerical_features = ['light']

def fit_data(data):
    scaler = StandardScaler()
    encoder = OneHotEncoder()
    encoder.fit(data[categorical_features])
    scaler.fit(data[numerical_features])
    return scaler, encoder

def process_labels(data):
    return data['action'].apply(lambda x: 1 if x == 'Clicked' else 0)

# Preprocess data
def preprocess_data(data, scaler, encoder):
    # Fill missing values
    #drop rows with missing values
    #data.fillna(method='ffill', inplace=True)
    #data.fillna(method='bfill', inplace=True)
    
    # Encode categorical features
    encoded_features = encoder.transform(data[categorical_features]).toarray()
    
    # Scale numerical features
    scaled_features = scaler.transform(data[numerical_features])

    locked = data['screen'].apply(lambda x: 1 if x == 'Locked' else 0)
    #proximity is already 0 or 1
    proximity = data['prox']
    
    # Combine features
    features = np.hstack((encoded_features, scaled_features, locked.values.reshape(-1, 1), proximity.values.reshape(-1, 1)))
    # Extract labels where clicked is true and removed is false
    
    return features