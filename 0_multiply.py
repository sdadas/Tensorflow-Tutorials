import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

y = a * b

with tf.Session() as sess:
    print(sess.run(y, {a: 2, b: 3}))