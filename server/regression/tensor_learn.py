# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:18:38 2018

@author: GUR44996
"""

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 200
pd.options.display.float_format = '{:.1f}'.format


data_frame = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv",sep = ",")
data_frame = data_frame.reindex(np.random.permutation(data_frame.index))
data_frame["median_house_value"]/=1000.0

#total_room colum as series
total_room = data_frame[["total_rooms"]]

#created a feature colum of it
feature_column = [tf.feature_column.numeric_column("total_rooms")]

#median house value colum as series
targets = data_frame["median_house_value"]

tf_optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0000001)
tf_optimizer = tf.contrib.estimator.clip_gradients_by_norm(tf_optimizer,5.0)
linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_column,optimizer=tf_optimizer)


def input_func(features,targets,batch_size=1,shuffle=True,num_of_epochs=None):
    
    features = {key:np.array(value) for key,value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features,targets))
    ds = ds.batch(batch_size).repeat(num_of_epochs)
    
    if shuffle:
        ds = ds.shuffle(buffer_size=100000)
        
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


linear_regressor.train(input_fn= lambda:input_func(total_room,targets),steps=100)

prediction_input_fn =lambda: input_func(total_room, targets, num_of_epochs=1, shuffle=False)

predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# Format predictions as a NumPy array, so we can calculate error metrics.
predictions = np.array([item['predictions'][0] for item in predictions])

mean_squared_error = metrics.mean_squared_error(predictions, targets)

root_mean_squared_error = math.sqrt(mean_squared_error)

min_val = data_frame["median_house_value"].min()
max_val = data_frame["median_house_value"].max()

difference = max_val - min_val

print("Mean Squared Error: %0.5f" % mean_squared_error)
print("Root Mean Squared Error: %0.5f" % root_mean_squared_error)
print("Difference between min and max: %0.5f" % difference)

calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
print(calibration_data)
