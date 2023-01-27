from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,BayesianRidge,LinearRegression,SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.utils import shuffle
import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    pwd = os.path.realpath(os.path.dirname(__file__))
    train_csv = pwd + '/data/Past_Students.csv'
    X = []
    Y = []
    with open(train_csv, "r") as f:
        for entry in f.read().split('\n')[1:]:
            X.append(list(map(float, entry.split(",")))[1:-1])
            Y.append(list(map(float, entry.split(",")))[-1])
    X = StandardScaler().fit_transform(X)

    print("Initializing Models...")
    models = [
            #['LinearRegression: ', LinearRegression()],
            ['SGDRegressor: ', SGDRegressor()],
            #['Ridge: ', Ridge()],
            #['BayesianRidge: ', BayesianRidge()]
            ]

    scoring = "neg_mean_squared_error"
    k_folds = 4 #number of folds to cut data into for training

    parameter_spaces = {
        'SGDRegressor: ': {
            'loss':['squared_error', 'huber'],
            'penalty': ['l2', 'l1', 'elasticnet', None],
            'alpha': [0.0001, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02],
            #'max_iter': [300, 700, 1100, 1500, 1800],
            'tol': [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008],
            'learning_rate': ['constant', 'optimal'],
            'eta0': [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15],
        }}
    
    while True:
        for name, model in models:
            search = GridSearchCV(model, parameter_spaces[name], scoring=scoring, cv=k_folds, verbose=1)
            result = search.fit(X, Y)
            if result.best_score_ > -0.004:
                print(result.best_score_, result.best_params_)
        


#-0.003916198943713893 {'alpha': 0.0075, 'eta0': 0.09, 'learning_rate': 'constant', 'loss': 'squared_error', 'penalty': 'l2', 'tol': 0.007}
#-0.003797457304596389 {'alpha': 0.0001, 'eta0': 0.08, 'learning_rate': 'constant', 'loss': 'huber', 'penalty': 'l2', 'tol': 0.005}
#-0.003961090403868041 {'alpha': 0.01, 'eta0': 0.09, 'learning_rate': 'constant', 'loss': 'squared_error', 'penalty': 'l2', 'tol': 0.007}
#-0.0036809531033871463 {'alpha': 0.0075, 'eta0': 0.07, 'learning_rate': 'constant', 'loss': 'huber', 'penalty': None, 'tol': 0.005}
#-0.003927602372818901 {'alpha': 0.0025, 'eta0': 0.05, 'learning_rate': 'constant', 'loss': 'squared_error', 'penalty': 'l2', 'tol': 0.007}
#-0.003909683261451422 {'alpha': 0.0025, 'eta0': 0.08, 'learning_rate': 'constant', 'loss': 'huber', 'penalty': 'l2', 'tol': 0.003}
#-0.003826305910135558 {'alpha': 0.0001, 'eta0': 0.1, 'learning_rate': 'constant', 'loss': 'huber', 'penalty': 'l1', 'tol': 0.008}
#-0.003935293040466775 {'alpha': 0.0001, 'eta0': 0.07, 'learning_rate': 'constant', 'loss': 'squared_error', 'penalty': 'l1', 'tol': 0.003}
#-0.003949430956689225 {'alpha': 0.0075, 'eta0': 0.08, 'learning_rate': 'constant', 'loss': 'huber', 'penalty': None, 'tol': 0.003}
#-0.003891719965183685 {'alpha': 0.005, 'eta0': 0.06, 'learning_rate': 'optimal', 'loss': 'huber', 'penalty': None, 'tol': 0.006}
#-0.0039931520732431796 {'alpha': 0.0001, 'eta0': 0.1, 'learning_rate': 'constant', 'loss': 'squared_error', 'penalty': 'elasticnet', 'tol': 0.007}
#-0.0038459190064670753 {'alpha': 0.0075, 'eta0': 0.15, 'learning_rate': 'constant', 'loss': 'huber', 'penalty': 'l1', 'tol': 0.007}
#-0.0039042579125390282 {'alpha': 0.0001, 'eta0': 0.08, 'learning_rate': 'constant', 'loss': 'squared_error', 'penalty': 'elasticnet', 'tol': 0.005} 
#-0.003995384233881876 {'alpha': 0.005, 'eta0': 0.06, 'learning_rate': 'optimal', 'loss': 'huber', 'penalty': 'elasticnet', 'tol': 0.003}
#
#
#
#alpha: 0.0075
#eta0: 0.075
#learning_rate: 'constant'
#loss: 'squared_error'
#penalty: l2
#tol: 0.007