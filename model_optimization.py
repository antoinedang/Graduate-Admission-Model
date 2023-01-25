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
            ['Ridge: ', Ridge()],
            ['BayesianRidge: ', BayesianRidge()]]

    scoring = "neg_mean_squared_error"
    k_folds = 5 #number of folds to cut data into for training

    best_scores = {}
    try:
        while True:
            parameter_spaces = {
            #'LinearRegression: ': {},
            'SGDRegressor: ': {
                'penalty': ['l2', 'l1', 'elasticnet', None],
                'alpha': np.random.uniform(0.00001, 0.01, 1),
                'max_iter': np.random.uniform(100, 2000, 1),
                'l1_ratio': np.random.uniform(0, 1, 1),
                'tol': np.random.uniform(0.0001, 0.01, 1),
                'epsilon': np.random.uniform(0.001, 0.5, 1),
                'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                'eta0': np.random.uniform(0.001, 0.1, 1),
                'power_t': np.random.uniform(0.1, 0.9, 1),
                'early_stopping': [True, False],
                'validation_fraction': np.random.uniform(0, 1, 1),
                'n_iter_no_change': np.random.uniform(2, 20, 1),
                'warm_start': [True, False],
                'average': [False, True, 5, 10, 50]
                },
            'Ridge: ': {
                'alpha': np.random.uniform(0, 100, 5),
                'max_iter': np.random.uniform(100, 10000, 5),
                'tol': np.random.uniform(0.0001, 0.01, 5),
                'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']},
            'BayesianRidge: ': {
                'n_iter': list(map(int, np.random.uniform(100, 1000, 3))),
                'tol': np.random.uniform(0.0001, 0.01, 3),
                'alpha_1': np.random.uniform(0.000001, 0.001, 3),
                'alpha_2': np.random.uniform(0.000001, 0.001, 3),
                'lambda_1': np.random.uniform(0.000001, 0.001, 3),
                'lambda_2': np.random.uniform(0.000001, 0.001, 3) }}

            for name, model in models:
                search = GridSearchCV(model, parameter_spaces[name], scoring=scoring, cv=k_folds, verbose=1, n_jobs=-1)
                print(" -------------- Searching parameter space for " + name)
                result = search.fit(X, Y)
                if best_scores.get(name, (-1.0, None))[0] < result.best_score_:
                    best_scores[name] = (result.best_score_, result.best_params_)
    except:
        print("========FINAL========")
        for name, results in best_scores.items():
            score, params = results
            print('Best Score for ' + name + str(-1*score))
            print('Best Hyperparameters: ' + str(params))


#Best Score for LinearRegression: 0.004858920158946343
#Best Hyperparameters: {}
#========FINAL========
#Best Score for SGDRegressor: 0.004506350511833713
#Best Hyperparameters: {'alpha': 0.004503730715707953, 'average': False, 'early_stopping': False, 'epsilon': 0.0955642641032819, 'eta0': 0.010880712160005118, 'l1_ratio': 0.06437136839773694, 'learning_rate': 'constant', 'max_iter': 1721.8404396679703, 'n_iter_no_change': 6.598111597175541, 'penalty': 'elasticnet', 'power_t': 0.8159247741879346, 'tol': 0.0028661064983024746, 'validation_fraction': 0.4738082970614147, 'warm_start': False}
#Best Score for Ridge: 0.004784011194734735
#Best Hyperparameters: {'alpha': 0.03960034330464657, 'max_iter': 757.3018206054315, 'solver': 'sparse_cg', 'tol': 0.009654811047378482}
#Best Score for BayesianRidge: 0.004851090313835032
#Best Hyperparameters: {'alpha_1': 2.7306255735020833e-05, 'alpha_2': 0.0007844687955435624, 'lambda_1': 0.0007063395523616791, 'lambda_2': 1.7328872926128296e-06, 'n_iter': 806, 'tol': 0.004509224206212482}