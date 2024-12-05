"""Start a Flower server.

Derived from Flower Android example.
"""

from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvgAndroid
from flwr.common import NDArrays, Scalar
import tensorflow as tf
from learn.dense import create_dense
import time
from typing import Dict, Optional, Tuple
from data import *
import sys

PORT = 8080

categorical_features = [ "prox", "screen",'volume', "battery", "location"]
numerical_features = []

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 8,
        "local_epochs": 6,
    }
    return config



def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load validation data
    data = load_data()
    data = merge_data(*data)
    device_list = ["f388a39a209144198f1312861dab3105", "ef24e3578f264168ae5adfb7e75e4cea", "db9a6313e6e84616b9be27054a2094b2", "d143f2f107854051bfc9e2cbe8c913b5", "adf653615f6644e5a2feeabf5f7ac675", "adf5c3a65c8e4d0abc900fccf2b665fe"]
    scaler, encoder = fit_data(data, categorical_features, numerical_features)

    
    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        weights = model.get_weights()
        #reshape parameter layers to match "weights" shape
        print("round", server_round)
        for i in range(0, len(parameters)):
            parameters[i] = parameters[i].reshape(weights[i].shape)
        model.set_weights(parameters)  # Update model with the latest parameters
        totalAccuracy = 0
        totalLoss = 0
        for device in device_list:
            device_data = data[data["device_id"] == device]
            labels = np.array(process_labels(device_data).map(lambda x: [x, 1-x]).to_list())
            processed = preprocess_data(device_data, scaler, encoder, categorical_features, numerical_features)
            loss, accuracy = model.evaluate(processed, labels, verbose=2)
            totalAccuracy += accuracy
            totalLoss += loss
            print(f"Device: {device}, Loss: {loss}, Accuracy: {accuracy}")

        return totalLoss/len(device_list), {"accuracy": totalAccuracy / len(device_list)}

    return evaluate

def main():
    device_num = int(sys.argv[1])
    rounds_num = int(sys.argv[2])
    print("Device num:", device_num)
    print("Rounds num:", rounds_num)
    model = create_dense(19)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    strategy = FedAvgAndroid(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=device_num,
        min_evaluate_clients=device_num,
        evaluate_fn= get_evaluate_fn(model),
        min_available_clients=device_num,
        on_fit_config_fn=fit_config,
    )

    try:
        # Start Flower server for 10 rounds of federated learning
        start_milis = int(round(time.time() * 1000))
        start_server(
            server_address=f"0.0.0.0:{PORT}",
            config=ServerConfig(num_rounds=rounds_num),
            strategy=strategy,
        )
        end_milis = int(round(time.time() * 1000))
        print("Time taken:", end_milis - start_milis)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()