# -*- coding: utf-8 -*-
"""
@created on: 9/28/18,
@author: Himaprasoon,
@version: v0.0.1

Description:

Sphinx Documentation Status:

"""

import tensorflow as tf

tf.set_random_seed(0)
from rnn_tranfer_learning.transfer_utils import get_transfered_weights_or_bias
from rnn_tranfer_learning.BasicRNN import save_path

tf.logging.set_verbosity(tf.logging.ERROR)
# hyperparameters
n_neurons = 128
learning_rate = 0.001
batch_size = 128
n_epochs = 1
# parameters
n_steps = 28  # 28 rows
n_inputs = 28  # 28 cols
n_outputs = 10  # 10 classes
# build a rnn model
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)

outputs, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
cell_kernel_assign = cell._kernel.assign(get_transfered_weights_or_bias(model_path=save_path,
                                                                        variable_name="rnn/myrnn/kernel"))
cell_bias_assign = cell._bias.assign(get_transfered_weights_or_bias(model_path=save_path,
                                                                    variable_name="rnn/myrnn/bias"))

output_transposed = tf.transpose(outputs, [1, 0, 2])
logits = tf.matmul(output_transposed[-1], tf.Variable(name="output", initial_value=get_transfered_weights_or_bias(model_path=save_path,
                               variable_name="output")))
# logits = tf.layers.dense(state, n_outputs)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
prediction = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

# input data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/mnsit")
X_test = mnist.test.images  # X_test shape: [num_test, 28*28]
X_test = X_test.reshape([-1, n_steps, n_inputs])
y_test = mnist.test.labels

init = tf.global_variables_initializer()
saver = tf.train.Saver()
# train the model
with tf.Session() as sess:
    sess.run(init)
    sess.run([cell_kernel_assign, cell_bias_assign])
    n_batches = mnist.train.num_examples // batch_size
    for epoch in range(n_epochs):
        for batch in range(n_batches):
            X_train, y_train = mnist.train.next_batch(batch_size)
            X_train = X_train.reshape([-1, n_steps, n_inputs])
            # sess.run(optimizer, feed_dict={X: X_train, y: y_train})
        loss_train, acc_train = sess.run(
            [loss, accuracy], feed_dict={X: X_train, y: y_train})
        print('Epoch: {}, Train Loss: {:.3f}, Train Acc: {:.3f}'.format(
            epoch + 1, loss_train, acc_train))
    loss_test, acc_test = sess.run(
        [loss, accuracy], feed_dict={X: X_test, y: y_test})
    print('Test Loss: {:.3f}, Test Acc: {:.3f}'.format(loss_test, acc_test))
    print(sess.run(cell._kernel))
