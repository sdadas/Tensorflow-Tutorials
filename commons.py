from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_regression(X, Y, w):
    plt.scatter(X, Y)
    plt.plot(X, X * w, 'r')
    plt.show()

def mnist() -> Datasets:
    return input_data.read_data_sets("downloads/MNIST_data/", one_hot=True)