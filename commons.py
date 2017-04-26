import time
from keras.engine import Layer
from typing import List, Tuple
from keras.models import Sequential
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from keras import backend as K
from PIL import Image
import numpy as np

def plot_regression(X, Y, a, b):
    plt.scatter(X, Y)
    plt.plot(X, X * a + b, 'r')
    plt.show()

def mnist() -> Datasets:
    return input_data.read_data_sets("downloads/MNIST_data/", one_hot=True)

class Visualizer(object):

    def __init__(self, model: Sequential, image_shape: Tuple[int, int, int]):
        self.model = model
        self.image_shape = image_shape
        self.input_img = model.input
        self.channels_first = (K.image_data_format() == 'channels_first')
        self.channels = image_shape[2]
        self.layer_dict = dict([(layer.name, layer) for layer in model.layers])
        model.summary()


    def visualize_conv(self, layer_name: str, num_filters: int):
        layer = self.layer_dict[layer_name]
        kept_filters = []
        for filter_idx in range(0, num_filters):
            res = self.__visualize_conv_filter(layer, filter_idx)
            if res is not None: kept_filters.append(res)

        # the filters that have the highest loss are assumed to be better-looking.
        kept_filters.sort(key=lambda x: x[1], reverse=True)
        self.__save_conv_image(kept_filters, layer_name)


    def __save_conv_image(self, kept_filters, layer_name: str, images_num: int = 16):
        margin = 3
        img_width = self.image_shape[0]
        img_height = self.image_shape[1]
        width = images_num * img_width + (images_num - 1) * margin
        result = np.zeros((img_height, width, self.channels))
        for idx in range(images_num):
            img, _ = kept_filters[idx]
            from_pos = (img_width + margin) * idx
            to_pos = (img_width + margin) * idx + img_width
            result[0:img_height, from_pos:to_pos] = img

        if self.channels == 1:
            result = np.reshape(result, [result.shape[0], result.shape[1]])

        image = Image.fromarray(result)
        image = image.convert('RGB')
        image.save('layer_%s.png' % (layer_name,))


    def __visualize_conv_filter(self, layer: Layer, filter_idx: int, iter: int = 20):
        print('Processing filter %d' % filter_idx)
        start_time = time.time()
        if self.channels_first:  loss = K.mean(layer.output[:, filter_idx, :, :])
        else: loss = K.mean(layer.output[:, :, :, filter_idx])

        grads = K.gradients(loss, self.input_img)[0]
        grads = self.__normalize(grads)
        iterate = K.function([self.input_img, K.learning_phase()], [loss, grads])
        input_img_data = self.__random_image()

        step = 1.
        loss_value = 0.
        for i in range(iter):
            loss_value, grads_value = iterate([input_img_data, 0])
            input_img_data += grads_value * step
            #print('Current loss value:', loss_value)
            if loss_value <= 0.: break

        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_idx, end_time - start_time))

        if loss_value > 0: return (self.__deprocess_image(input_img_data[0]), loss_value)
        else: return None


    # generate gray image with random noise
    def __random_image(self):
        width = self.image_shape[0]
        height = self.image_shape[1]
        if self.channels_first: input_img_data = np.random.random((1, self.channels, width, height))
        else: input_img_data = np.random.random((1, width, height, self.channels))
        input_img_data = (input_img_data - 0.5) * 20 + 128
        return input_img_data


    # util function to convert a tensor into a valid image
    def __deprocess_image(self, x):
        # normalize tensor: center on 0., ensure std is 0.1
        res = x - x.mean()
        res /= (res.std() + 1e-5)
        res *= 0.1

        # clip to [0, 1]
        res += 0.5
        res = np.clip(res, 0, 1)

        # convert to RGB array
        res *= 255
        if self.channels_first: res = res.transpose((1, 2, 0))
        res = np.clip(res, 0, 255).astype('uint8')
        return res


    # utility function to normalize a tensor by its L2 norm
    def __normalize(self, x):
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)