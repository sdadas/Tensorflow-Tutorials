from commons import plot_regression
import tensorflow as tf
import numpy as np

x_train = np.linspace(-1, 1, 101)
y_train = 2 * x_train + np.random.randn(len(x_train)) * 0.33

x_input = tf.placeholder(tf.float32)
y_input = tf.placeholder(tf.float32)

a = tf.Variable(0.0)
b = tf.Variable(0.0)
y = x_input * a + b

cost = tf.reduce_sum(tf.square(y_input - y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10):
    for x_one, y_one in zip(x_train, y_train):
        sess.run(train, feed_dict={x_input: x_one, y_input: y_one})

plot_regression(x_train, y_train, sess.run(a), sess.run(b))
sess.close()