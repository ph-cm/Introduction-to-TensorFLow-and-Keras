import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import random

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

#Computing Gradients
a = tf.random.normal(shape=(2, 2))
b = tf.random.normal(shape=(2, 2))

with tf.GradientTape() as tape:
    tape.watch(a)
    c = tf.sqrt(tf.square(a) + tf.square(b))
    dc_da = tape.gradient(c, a)
    print(dc_da)

#Example 1: Linear Regression
np.random.seed(13)

train_x = np.linspace(0, 3, 120)
train_labels = 2 * train_x + 0.9 + np.random.randn(*train_x.shape) * 0.5

plt.scatter(train_x, train_labels)
plt.show()

input_dim = 1
output_dim = 1
learning_rate = 0.1

#weight matrix
w = tf.Variable([[100.0]])
#bias vector
b = tf.Variable(tf.zeros(shape=(output_dim,)))

def f(x):
    return tf.matmul(x,w) + b
def compute_loss(labels, predictions):
    return tf.reduce_mean(tf.square(labels - predictions))

def train_on_batch(x, y):
    with tf.GradientTape() as tape:
        predictions = f(x)
        loss = compute_loss(y, predictions)
        dloss_dw, dloss_db = tape.gradient(loss, [w, b])
    w.assign_sub(learning_rate * dloss_dw)
    b.assign_sub(learning_rate * dloss_db)
    return loss

#shuffe the data
indices = np.random.permutation(len(train_x))
features = tf.constant(train_x[indices], dtype = tf.float32)
labels = tf.constant(train_labels[indices],dtype = tf.float32)

batch_size = 4
for epoch in range(10):
    for i in range(0, len(features), batch_size):
        loss = train_on_batch(tf.reshape(features[i:i+batch_size], (-1,1)), tf.reshape(labels[i:i+batch_size], (-1,1)))
    print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))

w, b

plt.scatter(train_x, train_labels)
x = np.array([min(train_x), max(train_x)])
y = w.numpy()[0,0]*x+b.numpy()[0]
plt.plot(x,y,color='red')
plt.show()

#Computional Graph and GPU Computations

@tf.function
def train_on_batch(x,y):
    with tf.GradientTape() as tape:
        predictions = f(x)
        loss = compute_loss(y, predictions)
        dloss_dw, dloss_db = tape.gradient(loss, [w,b])
    w.assign_sub(learning_rate * dloss_dw)
    b.assign_sub(learning_rate * dloss_db)
    return loss
#code doesnt change at all but if it runs in a GPU and on a larger dataset-it would have a difference in speed
       
#Dataste API
w.assign([[10.0]])
b.assign([0.0])

dataset =  tf.data.Dataset.from_tensor_slices((train_x.astype(np.float32), train_labels.astype(np.float32)))
dataset = dataset.shuffle(buffer_size=1024).batch(256)

for epoch in range(10):
    for step, (x, y) in enumerate(dataset):
        loss = train_on_batch(tf.reshape(x,(-1,1)), tf.reshape(y,(-1,1)))
    print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))

#Example 2: Classification
np.random.seed(0)

n = 100
X, Y = make_classification(n_samples=n, n_features=2, n_redundant=0, n_informative=2, flip_y=0.05, class_sep=1.5)

X = X.astype(np.float32)
Y = Y.astype(np.float32)

split = [70*n//100, (15+70)*n//100]
train_x, valid_x, test_x = np.split(X, split)
train_labels, valid_labels, test_labels = np.split(Y, split)

def plot_dataset(features, labels, W=None, b=None):
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('$x_i[0]$ -- (feature 1)')
    ax.set_ylabel('$x_i[1]$ -- (feature 2)')
    colors = ['r' if l else 'b' for l in labels]
    ax.scatter(features[:,0], features[:,1], marker='o', c=colors, s=100,alpha=0.5)
    if W is not None:
        min_x = min(features[:,0])
        max_x = max(features[:,1])
        min_y = min(features[:,1])*(1-.1)
        max_y = max(features[:,1])*(1+.1)
        cx = np.array([min_x, max_x], dtype=np.float32)
        cy = (0.5-W[0]*cx-b)/W[1]
        ax.plot(cx,cy,'g')
        ax.set_ylim(min_y,max_y)
    fig.show()
    plt.show()
plot_dataset(train_x,train_labels)

#NOrmalizing Data
train_x_norm = (train_x-np.min(train_x)) / (np.max(train_x)-np.min(train_x))
valid_x_norm = (valid_x-np.min(train_x)) / (np.max(train_x)-np.min(train_x))
test_x_norm = (test_x-np.min(train_x)) / (np.max(train_x)-np.min(train_x))