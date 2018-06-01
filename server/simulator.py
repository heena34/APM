# Load libraries
import pandas as pd
import numpy as np 
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import time
import operator
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from datetime import datetime
from sklearn import preprocessing
import numpy, json, os
from pandas import read_csv
from sklearn.decomposition import PCA
import socket
import json
import os
import sys
global data

i = 0
total_cycles = 192
while i < total_cycles:
    print("Cycle Value : %d" %i)

    print("Printing  - LIVE SENSOR DATA")
    #data.head()
    data = pd.read_csv('Dataset/PM_test.csv', sep=" ", header=None) 

    #print(data.head(i))
    print(data.ix[i])
    data = data.ix[i]
    data.to_csv('Output/live.csv', encoding='utf-8',index = None)
    print("Engine Working ..................................")
    time.sleep(5)
    print("Engine just Woke up and Provided data ...........")
    print("\n")
    print("\n")
    dataFromLive = pd.read_csv('Output/live.csv', sep=" ", header=None)
    print("showing data from live sheet")
    print(dataFromLive)

    if i is (total_cycles-1):
        i=0
    else:
        i = i+1
