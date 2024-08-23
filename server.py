"""Start a Flower server.

Derived from Flower Android example.
"""

from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvgAndroid
from flwr.common import NDArrays, Scalar
import tensorflow as tf
from learn.dense import create_dense
from typing import Dict, Optional, Tuple
from data import *
import json
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

PORT = 8080


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 8,
        "local_epochs": 5,
    }
    return config

def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    with(open("users/2_test.json", "r")) as f:
        data = json.load(f)
        notification_list = [DataRow(item) for item in data]
        notification_labels = np.array([(notification.has_user_interacted(), not notification.has_user_interacted()) for notification in notification_list], dtype=np.float32)
        ohe = OneHotEncoder(sparse_output=False)
        minmax = MinMaxScaler()
        notification_data = ohe.fit_transform([item.to_rf_array() for item in notification_list])
        notification_light = minmax.fit_transform([[item.light_lux] for item in notification_list])
        notification_data = np.hstack((notification_data, notification_light), dtype=np.float32)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        parameters.count
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(notification_data, notification_labels)
        return loss, {"accuracy": accuracy}

    return evaluate

def main():
    model = create_dense(14)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    strategy = FedAvgAndroid(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        evaluate_fn= get_evaluate_fn(model),
        min_available_clients=2,
        on_fit_config_fn=fit_config,
    )

    try:
        # Start Flower server for 10 rounds of federated learning
        start_server(
            server_address=f"0.0.0.0:{PORT}",
            config=ServerConfig(num_rounds=10),
            strategy=strategy,
        )
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()