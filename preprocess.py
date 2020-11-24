# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 11:35:04 2020

@author: secil
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from imblearn.over_sampling._smote import SMOTE
import scipy as sp
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

def read_csv(path):
    data = pd.read_csv(path)
    return data 

def one_hot_encoding(dt,column):
    columnTransformer = ColumnTransformer([('encoder', 
                                        OneHotEncoder(), 
                                        [0])], 
                                      remainder='passthrough')
    ohe_data = columnTransformer.fit_transform(dt[[column]])
    
    if isinstance(ohe_data,sp.sparse.csr.csr_matrix):
        ohe_data = ohe_data.toarray()
        
    for j in np.arange(np.size(ohe_data,1)):
            dt[column+'_'+str(j)] = ohe_data[:,j]
        
    dt.drop([column], axis=1, inplace=True)

def label_encoding(dt,column):
    label_encoder = LabelEncoder()   
    dt[[column]] = label_encoder.fit_transform(dt[[column]])
        
def normalisation(dt,column):
    min_max_scaler = MinMaxScaler()  
    dt[[column]] = min_max_scaler.fit_transform(dt[[column]])
    
def preprocessing(dt, numeric_columns, binary_columns, categorical_columns):   
    #encoding
    for i in binary_columns:
        label_encoding(dt, i)
       
    for i in categorical_columns:
        one_hot_encoding(dt, i)
        
    #normalisation    
    for i in numeric_columns:
        normalisation(dt, i)       
    return dt
