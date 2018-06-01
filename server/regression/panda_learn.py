# -*- coding: utf-8 -*-
"""
Created on Tue May 22 11:52:34 2018

@author: Prathmesh
"""
from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot,sys

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
#print(california_housing_dataframe.describe())
#print (california_housing_dataframe.head())
#california_housing_dataframe.hist()
latitude_series = california_housing_dataframe["latitude"]
#print(latitude_series)
#print(latitude_series[:4])
#latitude_log = np.log(latitude_series)
#assertion_series = latitude_series.apply(lambda val:val<3.51)
#print(assertion_series)

