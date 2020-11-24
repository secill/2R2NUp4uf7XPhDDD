# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:28:28 2020

@author: secil
"""
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from preprocess import preprocessing


def modeling(dt, c, kf, cl_model, numeric_columns, binary_columns, categorical_columns, sm=None): 
    
    dt = preprocessing(dt, numeric_columns, binary_columns, categorical_columns)
    x = dt.iloc[:,:c]
    y = dt.iloc[:,-1:]
    
    acc_scores = []
    rc_scores = []
    for train_index, test_index in kf.split(x,y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if sm is not None:
            x_train, y_train = sm.fit_resample(x_train, y_train)
        cl_model.fit(x_train,y_train)
        y_pred = cl_model.predict(x_test)
        #cm = confusion_matrix(y_test, y_pred) 
        #tn,fp,fn,tp = cm.ravel()
        #print(cm)
        acc_score = accuracy_score(y_test, y_pred)
        acc_scores.append(acc_score)
        rc_score = recall_score(y_test,y_pred)
        rc_scores.append(rc_score)      
    return acc_scores, rc_scores

