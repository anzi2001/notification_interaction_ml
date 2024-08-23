import tensorflow as tf
from data import *
import numpy as np

if __name__ == "__main__":
    data = np.array([[0,1,2],[1,2,3],[2,3,4]])
    input = np.array([0,1,2])
    layer = tf.keras.layers.Normalization(axis=-1)
    layer.adapt(data)
    print(layer.mean)
    print(layer.variance)
    print(layer(input))

