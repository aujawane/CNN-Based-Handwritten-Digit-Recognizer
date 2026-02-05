from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = fetch_openml('mnist_784', as_frame=False, cache=False)

# Dataset shape: 70,000 images × 784 features
print(mnist.data.shape)

# preprocessing data
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')
X /= 255.0 # scale each pixel value between 0 and 1
X.min(), X.max()