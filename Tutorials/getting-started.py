import tensorflow as tf

# building computational graph

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

# running the computational graph

sess = tf.Session()
print(sess.run([node1, node2]))

