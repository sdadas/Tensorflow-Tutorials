import tensorflow as tf
from commons import mnist

data = mnist()

x_input = tf.placeholder(tf.float32, [None, 784]) # 28 * 28 pixels
y_input = tf.placeholder(tf.float32, [None, 10])  # number of classes

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x_input, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y))
train = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
while data.train.epochs_completed < 100:
    x_batch, y_batch = data.train.next_batch(100)
    sess.run(train, feed_dict={x_input: x_batch, y_input: y_batch})

correct_prediction = tf.equal(tf.argmax(y_input, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x_input: data.test.images, y_input: data.test.labels}))

sess.close() # ~92.5%