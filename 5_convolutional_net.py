from typing import List
import tensorflow as tf
from commons import mnist

data = mnist()

x_input = tf.placeholder(tf.float32, [None, 784]) # 28 * 28 pixels
y_input = tf.placeholder(tf.float32, [None, 10])  # number of classes
x_input2d = tf.reshape(x_input, [-1, 28, 28, 1]) # [examples, width, height, channels]

def layer(input, shape: List[int], activation, name: str):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
        output = activation(tf.matmul(input, W) + b)
    return output

def conv2d(input, shape: List[int], stride:int, name:str):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
        output = tf.nn.conv2d(input, W, strides=[1, stride, stride, 1], padding='SAME')
        output = tf.nn.bias_add(output, b)
    return tf.nn.relu(output)

def maxpool2d(input, k):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

conv_dropout = tf.placeholder(tf.float32)
fc_dropout = tf.placeholder(tf.float32)

conv1 = conv2d(x_input2d, [3, 3, 1, 32], stride=1, name='conv1')
conv1 = maxpool2d(conv1, 2)
conv1 = tf.nn.dropout(conv1, conv_dropout)

conv2 = conv2d(conv1, [3, 3, 32, 64], stride=1, name='conv2')
conv2 = maxpool2d(conv2, 2)
conv2 = tf.nn.dropout(conv2, conv_dropout)

conv3 = conv2d(conv2, [3, 3, 64, 128], stride=1, name='conv3')
conv3 = maxpool2d(conv3, 2)
conv3 = tf.nn.dropout(conv3, conv_dropout)

fc_input = tf.reshape(conv3, [-1, 4*4*128]) # 28*28 / 2 / 2 / 2 = 4*4
fc1 = layer(fc_input, [4*4*128, 625], tf.nn.relu, name='fc1')
fc1 = tf.nn.dropout(fc1, fc_dropout)
y = layer(fc1, [625, 10], lambda _: _, name='output')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y))
train = tf.train.AdamOptimizer().minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
while data.train.epochs_completed < 100:
    x_batch, y_batch = data.train.next_batch(50)
    sess.run(train, {x_input: x_batch, y_input: y_batch, conv_dropout: 0.8, fc_dropout: 0.5})

correct_prediction = tf.equal(tf.argmax(y_input, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, {x_input: data.test.images, y_input: data.test.labels, conv_dropout: 1., fc_dropout: 1.}))

sess.close() # ~99.4%
