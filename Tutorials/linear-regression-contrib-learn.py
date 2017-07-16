import tensorflow as tf
import numpy as np

'''
contrib.learn simplifies the mechanics of ML by
    1. running training loop
    2. running evaluation loop
    3. managing data sets
    4. managing feeds

it also defines many common models
'''

# declare a list of features
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

'''
Estimators are the thing that invoke training and evaluation(inference).
There are many predefined types.
'''
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

'''
Tensorflow have many helper to read and setup data sets.
We have to tell the function how many batches of data (number_epochs)
we want and how big each batch should be
'''
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train,
                                              batch_size=4,
                                              num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_eval}, y_eval,
                                                   batch_size=4,
                                                   num_epochs=1000)

# invoke 1000 training steps by invoke the method and passing the training dataset
estimator.fit(input_fn=input_fn, steps=1000)

# Now evaluate how well the model did
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r" % train_loss)
print("eval loss: %r" % eval_loss)
