from typing import List
import tensorflow as tf
from commons import mnist

data = mnist()

x_input = tf.placeholder(tf.float32, [None, 784]) # 28 * 28 pixels
y_input = tf.placeholder(tf.float32, [None, 10])  # number of classes

def layer(input, shape: List[int], activation, name: str):
    with tf.name_scope(name):
        W = tf.Variable(tf.random_uniform(shape) * 0.01)
        b = tf.Variable(tf.random_uniform([shape[-1]]) * 0.01)
        output = activation(tf.matmul(input, W) + b)
    return output

input_dropout = tf.placeholder(tf.float32)
layer_dropout = tf.placeholder(tf.float32)

dropout_input = tf.nn.dropout(x_input, input_dropout)
hidden1 = layer(dropout_input, [784, 625], tf.nn.relu, 'hidden1')
hidden1 = tf.nn.dropout(hidden1, layer_dropout)
hidden2 = layer(hidden1, [625, 625], tf.nn.relu, 'hidden2')
hidden2 = tf.nn.dropout(hidden2, layer_dropout)
y = layer(hidden2, [625, 10], lambda _: _, 'output')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y))
train = tf.train.AdamOptimizer().minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
while data.train.epochs_completed < 100:
    x_batch, y_batch = data.train.next_batch(100)
    sess.run(train, {x_input: x_batch, y_input: y_batch, input_dropout: 0.8, layer_dropout: 0.5})

correct_prediction = tf.equal(tf.argmax(y_input, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, {x_input: data.test.images, y_input: data.test.labels, input_dropout: 1., layer_dropout: 1.}))

sess.close() # ~98.6%
