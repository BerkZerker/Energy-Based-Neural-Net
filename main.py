from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf  # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler


# 1. Check if TF can see the GPU
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs detected: {len(gpus)}")

# 2. Print the details
if gpus:
    print(f"Device Name: {gpus[0]}")
    # Run a quick test computation on GPU
    with tf.device("/GPU:0"):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("\nTest calculation successful:")
        print(c)
else:
    print("No GPU found. TensorFlow is running on CPU.")
