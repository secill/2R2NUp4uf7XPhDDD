# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:51:52 2020

@author: secil
"""
import pandas as pd
from preprocess import read_csv
from model import modeling
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from imblearn.over_sampling._smote import SMOTE
import statistics

def main():
    
    data = read_csv("term-deposit-marketing-2020.csv")

    numeric_columns = ["age","balance","day","duration","campaign"]
    binary_columns = ["default","housing","loan","y"]
    categorical_columns = ["job","marital","education","contact","month"]
    
    '''no etiketli veri sayısı, yes etiketli veri sayısından çok fazla 
    olduğu için model yes etiketli verileri doğru tahmin edemiyor.
    Bu problemi aşmak için imbalanced learning yöntemlerinden smote kullanıldı.'''
    sm = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=0)
    
    kf = KFold(n_splits=5,shuffle=True)
    svc = SVC()
    c = 13
    acc_scores, rc_scores = modeling(data,c,kf, svc, numeric_columns, binary_columns, categorical_columns, sm)
    print(statistics.mean(acc_scores))
    print(statistics.mean(rc_scores))
   
if __name__=='__main__':
    main()
#data = pd.read_csv("term-deposit-marketing-2020.csv")