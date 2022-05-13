# code implementation of the report

student name:XiuyuanZhu \
student ID: 1216251


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


