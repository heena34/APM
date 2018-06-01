# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 23:23:42 2018

@author: GUR44996
"""

import json

import threading

import pandas as pd

from multiprocessing.pool import ThreadPool

from server.ml.lk_regression1 import MLPredictions



class Live_Streaming:
    
    def __init__(self,frame_length):
        self.frame_length = frame_length
        self.pool = ThreadPool(processes = 1)
        
        
    def trainOnAlgoName(self,algo_name,approach,w1,w0,selected_frame):
        mp_exe = MLPredictions(algo_name,approach,w1,w0)
        result = mp_exe.start_testing_on_stream(selected_frame)
        return selected_frame


        
    def start_streaming(self,w1,w0,number_of_records):

        if number_of_records > self.frame_length:
            number_of_records = self.frame_length


        selected_data_frame = pd.read_csv('Output/live.csv',nrows=number_of_records)
        result_one = self.pool.apply_async(self.trainOnAlgoName,("LinearRegression","PCA",w1,w0,selected_data_frame))
        print(result_one.get())

        result_two = self.pool.apply_async(self.trainOnAlgoName,("LinearDiscriminantAnalysis","PCA",w1,w0,selected_data_frame))
        print(result_two.get())

        result_three = self.pool.apply_async(self.trainOnAlgoName,("DecisionTreeRegressor","PCA",w1,w0,selected_data_frame))
        print(result_three.get())


        result_four = self.pool.apply_async(self.trainOnAlgoName,("LogisticRegression","PCA",w1,w0,selected_data_frame))
        print(result_four.get())

        result_five = self.pool.apply_async(self.trainOnAlgoName,("KNeighborsClassifier","PCA",w1,w0,selected_data_frame))
        print(result_five.get())