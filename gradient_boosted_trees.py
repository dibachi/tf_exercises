"""Implement a Gradient Boosted Decision Tree (GBDT) with TensorFlow. 
This example is using the Boston Housing Value dataset as training samples. 
The example supports both Classification (2 classes: value > $23000 or not) 
and Regression (raw home value as target). 

Boston Housing Dataset:

The dataset contains information collected by the U.S Census Service 
concerning housing in the area of Boston Mass. It was obtained from the 
StatLib archive (http://lib.stat.cmu.edu/datasets/boston), and has been 
used extensively throughout the literature to benchmark algorithms. 
However, these comparisons were primarily done outside of Delve and are 
thus somewhat suspect. The dataset is small in size with only 506 cases.

The data was originally published by Harrison, D. and Rubinfeld, D.L. 
`Hedonic prices and the demand for clean air', J. Environ. Economics 
& Management, vol.5, 81-102, 1978.`

"""

#ignore gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

import tensorflow as tf
import numpy as np
import copy

num_classes = 2 #classify as >=$23000 or not
num_features = 13 #features from imported data

max_steps = 2000
batch_size = 256
learning_rate = 1.0
l1_regul = 0.0
l2_regul = 0.1

num_batches_per_layer = 1000
num_trees = 10
max_depth = 4
print("Made it here 0 \n")
from tensorflow.keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

#for binary classification, we split classes along $23000
def to_binary_class(y):
    for i, label in enumerate(y):
        if label >= 23.0:
            y[i] = 1
        else:
            y[i] = 0
    return y

## GBDT CLASSIFIER
#convert y data from continuous regression to discrete classification data
y_train_binary = to_binary_class(copy.deepcopy(y_train))
y_test_binary = to_binary_class(copy.deepcopy(y_test))
print("Made it here 1 \n")
#build the input function
train_input_fn = tf.data.numpy_input_fn(
    x={'x':x_train}, y=y_train_binary, batch_size=batch_size, 
    num_epochs=None, shuffle=True
)
test_input_fn = tf.data.numpy_input_fn(
    x={'x': x_test}, y=y_test_binary,batch_size=batch_size, 
    num_epochs=1, shuffle=False
)
test_train_input_fn = tf.data.numpy_input_fn(
    x={'x': x_train}, y=y_train_binary, batch_size=batch_size, 
    num_epochs=1, shuffle=False
)
print("Made it here 2 \n")
#GBDT models from tf estimator requires 'feature_column' data format
feature_columns = [tf.feature_column.numeric_column(key='x', shape=(num_features,))]
print("Made it here 3 \n")
gbdt_classifier = tf.estimator.BoostedTreesClassifier(
    n_batches_per_layer=num_batches_per_layer,
    feature_columns=feature_columns, 
    n_classes=num_classes,
    learning_rate=learning_rate, 
    n_trees=num_trees,
    max_depth=max_depth,
    l1_regularization=l1_regul, 
    l2_regularization=l2_regul
)
print("Made it here 4 \n")
gbdt_classifier.train(train_input_fn, max_steps=max_steps)
gbdt_classifier.evaluate(test_train_input_fn)
gbdt_classifier.evaluate(test_input_fn)
print("Made it here 5 \n")
## GBDT REGRESSOR
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': x_train}, y=y_train,
    batch_size=batch_size, num_epochs=None, shuffle=True)
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': x_test}, y=y_test,
    batch_size=batch_size, num_epochs=1, shuffle=False)
# GBDT Models from TF Estimator requires 'feature_column' data format.
feature_columns = [tf.feature_column.numeric_column(key='x', shape=(num_features,))]
print("Made it here 6 \n")
gbdt_regressor = tf.estimator.BoostedTreesRegressor(
    n_batches_per_layer=num_batches_per_layer,
    feature_columns=feature_columns, 
    learning_rate=learning_rate, 
    n_trees=num_trees,
    max_depth=max_depth,
    l1_regularization=l1_regul, 
    l2_regularization=l2_regul
)
print("Made it here 7 \n")
gbdt_regressor.train(train_input_fn, max_steps=max_steps)
gbdt_regressor.evaluate(test_input_fn)
print("Made it here final \n")
