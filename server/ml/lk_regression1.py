# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:21:06 2018

@author: GUR30486
"""


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
import numpy, json,traceback
from pandas import read_csv
from sklearn.decomposition import PCA



class MLPredictions:
    
    
    PCA = 'PCA'
    
    def __init__(self,algo_name,approach,w1,w0):
        self.algoName = algo_name
        self.approach = approach    
        self.w1 = w1
        self.w0 = w0
    

    def general_data_processing(self,w1,w0):
        ##################################
        # Data Ingestion
        ##################################        
        # read training data - It is the aircraft engine run-to-failure data.
        self.train_df = pd.read_csv('./server/Dataset/PM_train.txt', sep=" ", header=None)
        self.train_df.drop(self.train_df.columns[[26, 27]], axis=1, inplace=True)
        self.train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                             's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                             's15', 's16', 's17', 's18', 's19', 's20', 's21']
        
        self.train_df = self.train_df.sort_values(['id','cycle'])
        
        # read test data - It is the aircraft engine operating data without failure events recorded.
        self.test_df = pd.read_csv('./server/Dataset/PM_test.txt', sep=" ", header=None)
        self.test_df.drop(self.test_df.columns[[26, 27]], axis=1, inplace=True)
        self.test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                             's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                             's15', 's16', 's17', 's18', 's19', 's20', 's21']
        
        # read ground truth data - It contains the information of true remaining cycles for each engine in the testing data.
        self.truth_df = pd.read_csv('./server/Dataset/PM_truth.txt', sep=" ", header=None)
        self.truth_df.drop(self.truth_df.columns[[1]], axis=1, inplace=True)
        #org_truth_df = truth_df
        #print(org_truth_df)
        ##################################
        # Data Preprocessing
        ##################################
        
        #######
        # TRAIN
        #######
        # Data Labeling - generate column RUL(Remaining Usefull Life or Time to Failure)
        rul = pd.DataFrame(self.train_df.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id', 'max']
        self.train_df = self.train_df.merge(rul, on=['id'], how='left')
        self.train_df['RUL'] = self.train_df['max'] - self.train_df['cycle']
        self.train_df.drop('max', axis=1, inplace=True)
        
        # generate label columns for training data
        # "label1" only to be used for binary classification, 
        # while trying to answer the question: is a specific engine going to fail within w1 cycles?

        self.train_df['label1'] = np.where(self.train_df['RUL'] <= w1, 1, 0 )
        self.train_df['label2'] = self.train_df['label1']
        self.train_df.loc[self.train_df['RUL'] <= w0, 'label2'] = 2
      
        # MinMax normalization (from 0 to 1)
        self.train_df['cycle_norm'] = self.train_df['cycle']
        cols_normalize = self.train_df.columns.difference(['id','cycle','RUL','label1','label2'])
        min_max_scaler = preprocessing.MinMaxScaler()
        norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(self.train_df[cols_normalize]), 
                                     columns=cols_normalize, 
                                     index=self.train_df.index)
        join_df = self.train_df[self.train_df.columns.difference(cols_normalize)].join(norm_train_df)
        self.train_df = join_df.reindex(columns = self.train_df.columns)
        self.train_df=self.train_df.drop(['setting3', 's1', 's5', 's10', 's16', 's18', 's19'], axis=1)
        self.train_df.to_csv('./server/Dataset/lk_training1.csv', encoding='utf-8',index = None)
    
        
        ######
        # TEST
        ######
        # MinMax normalization (from 0 to 1)
        self.test_df['cycle_norm'] = self.test_df['cycle']
        norm_test_df = pd.DataFrame(min_max_scaler.transform(self.test_df[cols_normalize]), 
                                    columns=cols_normalize, 
                                    index=self.test_df.index)
        test_join_df = self.test_df[self.test_df.columns.difference(cols_normalize)].join(norm_test_df)
        self.test_df = test_join_df.reindex(columns = self.test_df.columns)
        self.test_df = self.test_df.reset_index(drop=True)
        #print(test_df.head())
        
        # Using ground truth dataset to generate labels for the test data.
        # Using ground truth dataset to generate labels for the test data.
        # generate column max for test data
        rul = pd.DataFrame(self.test_df.groupby('id')['cycle'].max()).reset_index()
        print(rul)
        rul.columns = ['id', 'max']
        self.truth_df.columns = ['more']
        self.truth_df['id'] = self.truth_df.index + 1
        self.truth_df['max'] = rul['max'] + self.truth_df['more']
        self.truth_df.drop('more', axis=1, inplace=True)
        
        # generate RUL for test data
        self.test_df = self.test_df.merge(self.truth_df, on=['id'], how='left')        
    
        self.test_df['RUL'] = self.test_df['max'] - self.test_df['cycle']
    
        self.test_df.drop('max', axis=1, inplace=True)
        
        # generate label columns w0 and w1 for test data
        self.test_df['label1'] = np.where(self.test_df['RUL'] <= w1, 1, 0 )
        self.test_df['label2'] = self.test_df['label1']
        self.test_df.loc[self.test_df['RUL'] <= w0, 'label2'] = 2
        rul1 = pd.DataFrame(self.test_df.groupby('id').last()).reset_index()
        #print(rul1)
        self.test_df=rul1
        
        self.test_df=self.test_df.drop(['setting3', 's1', 's5', 's10', 's16', 's18', 's19'], axis=1)
        #print(test_df)
        self.test_df.to_csv('./server/Dataset/lk_test1.csv', encoding='utf-8',index = None)
    
    
    def data_processing_regression(self):    
        self.reg_train_df=self.train_df.drop(['label1', 'label2'], axis=1)
        self.reg_test_df=self.test_df.drop(['RUL','label1', 'label2'], axis=1)
    
            
    def calculate_feature_ranking_PCA(self):
        # Feature Extraction with PCA
    
        reg_train_df_features = self.reg_train_df.drop(['id','cycle','RUL'], axis=1)
        X = reg_train_df_features.values
        y = self.reg_train_df['RUL'].values
        
        # load data
     #   url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
      #  names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
      #  dataframe = read_csv(url, names=names)
      #  array = dataframe.values
      #  X = array[:,0:8]
      #  Y = array[:,8]
        # feature extraction
        pca = PCA(n_components=18)
        fit = pca.fit(X, y)
        # summarize components
        #print("Explained Variance: %s") % fit.explained_variance_ratio_
        print(fit.components_)
        print(pca.explained_variance_ratio_)
        
        
        #importances = fit.explained_variance_ratio_
        importances = pca.explained_variance_ratio_
        feature_names = reg_train_df_features.columns.tolist()
        feature_imp_dict = dict(zip(feature_names, importances))
        sorted_features = sorted(feature_imp_dict.items(), key=operator.itemgetter(1), reverse=True)
        indices = np.argsort(importances)[::-1]
        
        # Print the feature ranking
        print("Feature ranking:")
    
        for feat in range(X.shape[1]):
            print("feature %d : %s (%f)" % (indices[feat], sorted_features[feat][0], sorted_features[feat][1]))
    
        # Plot the feature importances of the forest
        plt.figure(0)
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
            color="r", align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        #plt.show()
        return reg_train_df_features
        
        
    def calculate_feature_ranking_extraTreeClasifier(self):
        """This Function calculate feature ranking and importances in a particular senerio"""
        reg_train_df_features = self.reg_train_df.drop(['id'], axis=1)
        X = reg_train_df_features.values
    
        # Store target feature in y array i.e storing vehicle values in y asix
        y = self.reg_train_df['RUL'].values
    
        tree_clf = ExtraTreesClassifier()
        
    
    
        # fit the model
        tree_clf.fit(X, y)
        importances = tree_clf.feature_importances_
        feature_names = reg_train_df_features.columns.tolist()
        feature_imp_dict = dict(zip(feature_names, importances))
        sorted_features = sorted(feature_imp_dict.items(), key=operator.itemgetter(1), reverse=True)
        indices = np.argsort(importances)[::-1]
        
        # Print the feature ranking
        print("Feature ranking:")
    
        for feat in range(X.shape[1]):
            print("feature %d : %s (%f)" % (indices[feat], sorted_features[feat][0], sorted_features[feat][1]))
    
        # Plot the feature importances of the forest
        plt.figure(0)
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
            color="r", align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        #plt.show()
        return reg_train_df_features
    
    def plot_gaussian_histogram(self):
        """" Function will plot a histogram that will tell the frequluncy of a vehicle at a point of time"""
        data = self.reg_train_df.RUL
        binwidth = 1
        plt.hist(data, bins=range(min(data), max(data) + binwidth, binwidth), log=False)
        plt.title("Gaussian Histogram")
        plt.xlabel("RUL")
        plt.ylabel("Number of times")
        #plt.show()
    
    def train_regression_model(self,algoName, cleanApproach):
        """Creating x-axis and y axis corresponding data to train our model so that we can make future predictions"""
        #Previously calculated feature ranking, Currently treating all feature important so not removing any feature
        
        #df_train_features = reg_train_df.drop(['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's3','s4', 's5', 's6', 's7', 's10', 's11', 's12', 's13', 's14','s15', 's16', 's17', 's18', 's19', 's20', 's21','RUL','label1','label2'], axis=1)
        
        #Removing S6 from training set since its not ranked by extraTreeClasifier.
    
        if cleanApproach == "PCA":
            print("Cleaning Approach is PCA - Train data")
            df_train_features = self.reg_train_df.drop(['cycle','RUL','id','s7','s8','s9','s11', 's12','s13','s14','s15','s17','s20','s21'], axis=1)
            
        elif cleanApproach == "treeClasifier":
            print("Cleaning Approach is treeClasifier - Train Data")
            df_train_features = self.reg_train_df.drop(['RUL'], axis=1)
        else:
            print("Invalid Clean approach")
        #df_train_features = reg_train_df
        # store features in X array
        X = df_train_features.values
        print(df_train_features.values)
        # store target in y array
        y = self.reg_train_df['RUL'].values
        print(y)
        # Create decision tree object
        # clf = DecisionTreeRegressor()
    
        self.train_model(algoName, X, y )
        
        
    
    def train_model(self,algo_choosen, X, y): 
        if algo_choosen == "DecisionTreeRegressor":
            self.algo_instant = DecisionTreeRegressor()
        elif algo_choosen == "LinearRegression":
            self.algo_instant = LinearRegression()
        elif algo_choosen == "LogisticRegression":
            self.algo_instant = LogisticRegression()
        elif algo_choosen == "LinearDiscriminantAnalysis":
            self.algo_instant = LinearDiscriminantAnalysis()
        elif algo_choosen == "KNeighborsClassifier":
            self.algo_instant = KNeighborsClassifier()
        elif algo_choosen == "DecisionTreeClassifier":
            self.algo_instant = DecisionTreeClassifier()
        elif algo_choosen == "RandomForestClassifier":
            self.algo_instant = RandomForestClassifier()
        elif algo_choosen == "ExtraTreesClassifier":
            self.algo_instant = ExtraTreesClassifier()
        elif algo_choosen == "GaussianNB":
            self.algo_instant = GaussianNB()
        elif algo_choosen == "SVC":
            self.algo_instant = SVC()
        # algo_instant = DecisionTreeRegressor()
    
        # fit the model    
        self.algo_instant.fit(X, y)
        
        # tree_clf = LogisticRegression()
    
    def test_model(self,aName, cleanApproach):
        #reg_test_df1 =""
        if cleanApproach == "PCA":
            print("Cleaning Approach is PCA - Test Data")
            reg_test_df1 = self.reg_test_df.drop(['id','cycle','id','s7','s8','s9','s11', 's12','s13','s14','s15','s17','s20','s21'], axis=1)
            X_test = reg_test_df1
            print(reg_test_df1)
        elif cleanApproach == "treeClasifier":
            print("Cleaning Approach is treeClasifier - Test Data")
            X_test = self.reg_test_df
            #df_train_features = reg_train_df.drop(['RUL'], axis=1)
        else:
            print("Invalid Clean approach")
            
        df_solution = pd.DataFrame()
        #df_solution['id'] = dataset_test.id
        #df_solution['RUL'] = dataset_test.RUL
        
        print("called test_model")
        # Starting time for time calculations
        start_time = time.time()
        #ExtraTreeClasifier -- revmonig s6
        #reg_test_ranked_df = reg_test_df.drop(['s6'], axis=1)
    
        #X_test=reg_test_ranked_df
        # print(X_Day_TEST)
        #print(X_test)
        
        predictions = self.algo_instant.predict(X_test)
        # predictions_day = algo_instant.predict(X_Day_TEST)
        print("The time taken to execute is %s seconds" % (time.time() - start_time))
    
        # Prepare Solution dataframe
        
        df_solution['Engine_ID'] = self.reg_test_df.id
        df_solution['Predicted_RUL'] = predictions
    
        rul_list = []
        engine_id_list = []
    
        #list_days = pd.DataFrame()        
        plt.plot(list(df_solution['Engine_ID']), predictions, color='lightblue')    
        csvName = str("./server/Output/") + aName + str("_") + cleanApproach + str("_Predicted.csv")
        print(csvName)
        #df_solution.to_csv('../Dataset/predicted.csv', encoding='utf-8',index = None)
        truth_data = pd.read_csv('./server/Dataset/PM_truth.txt', sep=" ", header=None)
        truth_data = truth_data.drop(truth_data.columns[[1]], axis=1)
        truth_data.columns = ['Actual_RUL']
        truth_data['Engine_ID'] = truth_data.index + 1
        #truth_data['max'] = rul['max'] + truth_df['more']
        print("LK Mishra")
        #print(truth_data)
        df_solution = df_solution.merge(truth_data, on=['Engine_ID'], how='left')
        df_solution.to_csv(csvName, encoding='utf-8',index = None)
            
        plt.plot(rul_list, engine_id_list, color='lightblue')
        plt.xlabel("EngineID")
        plt.ylabel("Predicted_RUL")
                
        plt.title(aName)
        #plt.show()
        return df_solution      




    def test_stream_data(self,aName, cleanApproach,test_dataframe):
        if cleanApproach == "PCA":
            print("Cleaning Approach is PCA - Test Data")
            test_dataframe = test_dataframe.drop(['id','cycle','id','s7','s8','s9','s11', 's12','s13','s14','s15','s17','s20','s21'], axis=1)
            X_test = test_dataframe
        elif cleanApproach == "treeClasifier":
            print("Cleaning Approach is treeClasifier - Test Data")
            X_test = test_dataframe
        else:
            print("Invalid Clean approach")
            
        df_solution = pd.DataFrame()
        
        print("called test_model")
        # Starting time for time calculations
        start_time = time.time()
        
        predictions = self.algo_instant.predict(X_test)
        # predictions_day = algo_instant.predict(X_Day_TEST)
        print("The time taken to execute is %s seconds" % (time.time() - start_time))
    
        # Prepare Solution dataframe    
        df_solution['Engine_ID'] = test_dataframe.id
        df_solution['Predicted_RUL'] = predictions
    
        return df_solution      
    
    
    def start_model_training(self):
        self.general_data_processing(self.w1,self.w0)    
        self.data_processing_regression()
        try:
            if self.approach is 'PCA':    
                self.calculate_feature_ranking_PCA()
            else:
                self.calculate_feature_ranking_extraTreeClasifier()
        
            self.plot_gaussian_histogram()
        
            self.train_regression_model(self.algoName, self.approach)
            
            return True
            
        except:
            traceback.print_exc()
        return False
        
        
        
    def start_model_testing(self):

        self.general_data_processing(self.w1,self.w0)    
        self.data_processing_regression()

        try:
            if self.approach is PCA:    
                self.calculate_feature_ranking_PCA()
            else:
                self.calculate_feature_ranking_extraTreeClasifier()
            
            self.plot_gaussian_histogram()
            
            self.train_regression_model(self.algoName, self.approach)
            
            df_solution = self.test_model(self.algoName, self.approach)        
            jsonObject = {}            
            jsonObject["engine_id"] = df_solution['Engine_ID'].values.tolist()
            jsonObject["predicted_rul"] = df_solution['Predicted_RUL'].values.tolist()
            jsonObject["actual_rul"] = df_solution['Actual_RUL'].values.tolist()
        
            jsonName = str("./server/Output/") + self.algoName + str("_") + self.approach + str("_Predicted.json")
            with open(jsonName, 'w') as outfile:
                json.dump(jsonObject, outfile)
            
            return jsonObject
            
        except:
            traceback.print_exc()    
        
        return None


    def preprocess_data_frame_stream(self,tf_dataframe):
        tf_dataframe.drop(self.tf_dataframe.columns[[26, 27]], axis=1, inplace=True)
        tf_dataframe.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                             's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                             's15', 's16', 's17', 's18', 's19', 's20', 's21']


        tf_dataframe['cycle_norm'] = tf_dataframe['cycle']
        norm_test_df = pd.DataFrame(min_max_scaler.transform(self.tf_dataframe[cols_normalize]), 
                                    columns=cols_normalize, 
                                    index=tf_dataframe.index)
        test_join_df = tf_dataframe[self.tf_dataframe.columns.difference(cols_normalize)].join(norm_test_df)
        tf_dataframe = test_join_df.reindex(columns = self.test_df.columns)
        tf_dataframe = tf_dataframe.reset_index(drop=True)
        tf_dataframe = tf_dataframe.drop(['setting3', 's1', 's5', 's10', 's16', 's18', 's19'], axis=1)
        tf_dataframe = tf_dataframe.drop(['RUL','label1', 'label2'], axis=1)


    def start_testing_on_stream(self,tf_dataframe):
        process_tf_dataframe = self.preprocess_data_frame_stream(self,tf_dataframe)

        try:  
            self.train_regression_model(self.algoName, self.approach)        
            df_solution = self.test_stream_data(self.algoName, self.approach,)        
            jsonObject = {}            
            jsonObject["engine_id"] = df_solution['Engine_ID'].values.tolist()
            jsonObject["predicted_rul"] = df_solution['Predicted_RUL'].values.tolist()    
            return jsonObject
            
        except:
            traceback.print_exc()    
        
        return None