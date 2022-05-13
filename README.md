# code implementation of the report

student name:XiuyuanZhu \
student ID: 1216251


## Usage
In this report we apply three classifiers to train the data and predict the sentiment. After each prediction you can print the classification report and confusion matrix to see the performance of each model.

## import package
          import itertools
          import numpy as np
          import pandas as pd
          import datetime as dt
          import seaborn as sns
          from sklearn.metrics import confusion_matrix
          from sklearn.model_selection import *
          import matplotlib.pyplot as plt

## model implementation
#### KNN classifier
        from sklearn.neighbors import KNeighborsClassifier 
#### Random Forest
        from sklearn.ensemble import RandomForestClassifier
#### Logistic Regression
        from sklearn.linear_model import LogisticRegression
#### Grid search with cross validation is used in all these classifiers to obtain the best params

## functions
        feature_extract(df,col,length)
        // this function is to extract fearue vectors into seperate columns
        
## soem basic datasets
1. train_st (training dataset from sentence-transform)
2. train_st_AAE, train_st_SAE (extract two demographic groups of datasets)
3. dev_st (development dataset from sentence-transform)
4. dev_st_AAE, dev_st_SAE (extract two demographic groups of datasets)
5. train_tfidf (training dataset from TFIDF)
6. train_tfidf_AAE, train_tfidf_SAE (extract two demographic groups of datasets)
7. dev_tfidf (development dataset from TFIDF)
8. dev_tfidf_AAE, dev_tfidf_SAE (extract two demographic groups of datasets)
