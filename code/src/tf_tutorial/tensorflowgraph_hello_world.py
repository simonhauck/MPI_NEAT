# from time import time
#
# import tensorflow as tf
# import numpy as np
#
# # Tutorial code based on https://www.easy-tensorflow.com/tf-tutorials/basics/graph-and-session
#
#
# print(tf.__version__)
#
# a = 2
# b = 3
#
# c = tf.add(a, b, 'Add')
# print(c)
#
# x = 2
# y = 3
# add_op = tf.add(x, y, name='Add')
# mul_op = tf.multiply(x, y, name='Multiply')
# pow_op = tf.pow(add_op, mul_op, name='Power')
# useless_op = tf.multiply(x, add_op, name='Useless')
#
# print(useless_op)
#
# # Tutorial Tensor Types
# print("Tutorial tensor types")
#
# a = tf.constant(2)
# b = tf.constant(3)
# c = a + b
# print(c)
#
# # Variables
# a = tf.Variable(name="var_1", initial_value=1)
# b = tf.Variable(name="var_2", initial_value=2)
# c = tf.add(a, b, name="Add1")
# print(a)
# print(b)
# print(c)
#
#
# # Placeholder
#
# def tf_add(param1, param2):
#     return tf.add(param1, param2)
#
# print("New placeholder function")
# print(tf_add([1, 2], [2, 3]))
# print(tf_add(2, 3))
# print(tf_add(5, 6))
#
# # Build test network
# print("Testnetwork")
#@tf.function optionally, can be a speed boost, but is not currently
# def simple_network(i1, i2, bias):
    # node1 = tf.([i1, i2], [1.0, 2.0])
   # node1 = tf.reduce_sum(node1, 1)
    # node1 = tf.nn.sigmoid(node1)

    # node2 = tf.multiply([i2, bias], tf.constant([4.0, 5.0]))
    # node2 = tf.reduce_sum(node2, 1)
    # node2 = tf.nn.sigmoid(node2)

    # output = tf.multiply([node1, i2, node2], tf.constant([-6.0, -3.0, 7.0]))
    # output = tf.reduce_sum(output, 1)
    # output = tf.nn.sigmoid(output)

#     return node1
#
# startTime = time()
# network_result = simple_network([0.0], [1.0], [1.0])
# endTime = time()
# print(network_result)
# print("RequiredTime", endTime - startTime)

# startTime2 = time()
# network_result2 = simple_network(tf.identity([0.0, 1.0]), tf.identity([1.0, 0.0]), tf.identity([1.0, 1.0]))
# endTime2 = time()
# print(network_result2)
# print("RequiredTime", endTime2 - startTime2)

