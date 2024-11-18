import tensorflow as tf
import numpy as np

#Simple Tensors
a = tf.constant([[1,2],[3,4]])
print(a)
a = tf.random.normal(shape=(10,3))
print(a)

print(a-a[0])
print(tf.exp(a)[0].numpy())

#Variables
s = tf.Variable(tf.zeros_like(a[0]))
for i in a:
    s.assign_add(i)
print(s)

tf.reduce_sum(a,axis=0)