from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,BayesianRidge,LinearRegression,SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.utils import shuffle
import os
import numpy as np

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

    best_scores = {}
    try:
        while True:
            parameter_spaces = {
            #'LinearRegression: ': {},
            'SGDRegressor: ': {
                'penalty': ['l2', 'l1', 'elasticnet', None],
                'alpha': np.random.uniform(0.001, 0.02, 1),
                'max_iter': np.random.uniform(300, 1800, 1),
                'l1_ratio': np.random.uniform(0.2, 1, 1),
                'tol': np.random.uniform(0.002, 0.0075, 1),
                'epsilon': np.random.uniform(0.2, 0.5, 1),
                'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                'eta0': np.random.uniform(0.05, 0.1, 1),
                'power_t': np.random.uniform(0.1, 0.8, 1),
                'early_stopping': [True, False],
                'validation_fraction': np.random.uniform(0, 1, 1),
                'n_iter_no_change': np.random.uniform(5, 20, 1),
                'warm_start': [True, False],
                'average': [False]
                }}

            for name, model in models:
                search = GridSearchCV(model, parameter_spaces[name], scoring=scoring, cv=k_folds, verbose=1, n_jobs=-1)
                print(" Searching parameter space for " + name)
                result = search.fit(X, Y)
                if best_scores.get(name, (-1.0, None))[0] < result.best_score_:
                    best_scores[name] = (result.best_score_, result.best_params_)
    except:
        print("========FINAL========")
        for name, results in best_scores.items():
            score, params = results
            print('Best Score for ' + name + str(-1*score))
            print('Best Hyperparameters: ' + str(params))