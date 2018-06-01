# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 23:23:42 2018

@author: GUR44996
"""

import json

import threading

from multiprocessing.pool import ThreadPool

from server.ml.lk_regression1 import MLPredictions



class Live_Streaming:
    
    def __init__(self,frame_length):
        self.frame_length = frame_length
        self.pool = ThreadPool(processes = 1)
        
        
    def trainOnAlgoName(self,algo_name,approach,w1,w0):
        mp_exe = MLPredictions(algo_name,approach,w1,w0)
        return mp_exe.start_model_testing()


        
    def start_streaming(self,w1,w0):

        result_one = self.pool.apply_async(self.trainOnAlgoName,("LinearRegression","PCA",w1,w0))
        print(result_one.get())

        result_three = self.pool.apply_async(self.trainOnAlgoName,("DecisionTreeRegressor","PCA",w1,w0))
        print(result_three.get())


        result_four = self.pool.apply_async(self.trainOnAlgoName,("LogisticRegression","PCA",w1,w0))
        print(result_four.get())

        result_two = self.pool.apply_async(self.trainOnAlgoName,("LinearDiscriminantAnalysis","PCA",w1,w0))
        print(result_two.get())

        result_five = self.pool.apply_async(self.trainOnAlgoName,("KNeighborsClassifier","PCA",w1,w0))
        print(result_five.get())



lv = Live_Streaming(50)
lv.start_streaming(30,15)
        