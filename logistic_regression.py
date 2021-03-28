import tensorflow as tf 
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt 

""" This example is using MNIST handwritten digits. 
The dataset contains 60,000 examples for training and 
10,000 examples for testing. The digits have been 
size-normalized and centered in a fixed-size image (28x28 pixels) 
with values from 0 to 255. 

In this example, each image will be converted to float32, normalized 
to [0, 1] and flattened to a 1-D array of 784 features (28*28).

![MNIST Dataset]
(http://neuralnetworksanddeeplearning.com/images/mnist_100_digits.png)

More info: http://yann.lecun.com/exdb/mnist/ """

#MNIST dataset parameters
num_classes = 10 #0 to 9
num_features = 784 #28x28 flattened 

#training params
learning_rate = 0.01
training_steps = 1000
batch_size = 256
display_step = 50

#prepare MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#cast training and test data as float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
#reshape data to 1-dimensional 
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
#normalize from 255 grayscale to 0-1
x_train, x_test = x_train/255., x_test/255.

#use tf.data API to shuffle and batch dataset
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

#weight of shape [784, 10], the 28x28 features, and number of classes
W = tf.Variable(tf.ones([num_features, num_classes]), name="weight")
#bias for 10 classes
b = tf.Variable(tf.zeros([num_classes]), name="bias")

def logistic_regression(x):
    #applying softmax normalizes to a probability distribution
    return tf.nn.softmax(tf.matmul(x, W) + b)

#cross-entropy loss function
def cross_entropy(y_pred, y_true):
    #encode variable to a one-hot vector
    y_true = tf.one_hot(y_true, depth=num_classes)
    #clip prediction values to avoid log(0) error
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    #compute and return cross-entropy
    return tf.reduce_mean(-tf.reduce_sum(y_true*tf.math.log(y_pred),1))

#define accuracy metric
def accuracy(y_pred, y_true):
    #the predicted class is the index with the highest probability
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.optimizers.SGD(learning_rate)

def run_optimization(x, y):
    with tf.GradientTape() as g:
        pred = logistic_regression(x)
        loss = cross_entropy(pred, y)
    
    gradients = g.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))

for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimization(batch_x, batch_y)

    if step % display_step == 0:
        pred = logistic_regression(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

#test model on validation set
pred = logistic_regression(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

#predict 5 images from validation set and display predictions
n_images = 5
test_images = x_test[:n_images]
predictions = logistic_regression(test_images)

for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model Prediction: %i" % np.argmax(predictions.numpy()[i]))
