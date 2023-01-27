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
    print_results_every = 100
    searches = 0
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
                search = GridSearchCV(model, parameter_spaces[name], scoring=scoring, cv=k_folds, verbose=0)
                result = search.fit(X, Y)
                if best_scores.get(name, (-1.0, None))[0] < result.best_score_:
                    best_scores[name] = (result.best_score_, result.best_params_)
            searches += 1
            if searches % print_results_every == 0:
                print("========FINAL========")
                for name, results in best_scores.items():
                    score, params = results
                    print('Best Score for ' + name + str(-1*score))
                    print('Best Hyperparameters: ' + str(params))
            else:
                periods = "..."
                for i in range(searches % print_results_every):
                    periods += "."
                print(periods, end='\r')
    except:
        print("========FINAL========")
        for name, results in best_scores.items():
            score, params = results
            print('Best Score for ' + name + str(-1*score))
            print('Best Hyperparameters: ' + str(params))


#Best Score for SGDRegressor: 0.004288750125935283
#Best Hyperparameters: {'alpha': 0.013106009453288915, 'average': False, 'early_stopping': False, 'epsilon': 0.4827674714162379, 'eta0': 0.05411408054047862, 'l1_ratio': 0.7796991892279388, 'learning_rate': 'constant', 'max_iter': 728.1787354909575, 'n_iter_no_change': 17.822663907664175, 'penalty': 'l2', 'power_t': 0.3549481127237305, 'tol': 0.0029966473228709486, 'validation_fraction': 0.17067006140956376, 'warm_start': False}

#Best Score for SGDRegressor: 0.004164163031474828
#Best Hyperparameters: {'alpha': 0.007652311917012859, 'average': False, 'early_stopping': True, 'epsilon': 0.3375889972201883, 'eta0': 0.06760802480023628, 'l1_ratio': 0.9788996590938879, 'learning_rate': 'constant', 'max_iter': 1777.2450825643216, 'n_iter_no_change': 16.23280217205433, 'penalty': 'l1', 'power_t': 0.5988042987892187, 'tol': 0.005818378199693471, 'validation_fraction': 0.8617253305366581, 'warm_start': True}

#Best Score for SGDRegressor: 0.00410901774332884
#Best Hyperparameters: {'alpha': 0.011645653773536845, 'average': False, 'early_stopping': False, 'epsilon': 0.25990852598564523, 'eta0': 0.05658332084622414, 'l1_ratio': 0.20933289424195253, 'learning_rate': 'constant', 'max_iter': 934.620808601195, 'n_iter_no_change': 10.931230044565119, 'penalty': 'elasticnet', 'power_t': 0.4122587632108734, 'tol': 0.0057044175583122496, 'validation_fraction': 0.7360968427149389, 'warm_start': False}

#Best Score for SGDRegressor: 0.004087553268180311
#Best Hyperparameters: {'alpha': 0.009166825877263324, 'average': False, 'early_stopping': True, 'epsilon': 0.48603692560719686, 'eta0': 0.09494253843658729, 'l1_ratio': 0.6709250863141156, 'learning_rate': 'adaptive', 'max_iter': 638.0431561784704, 'n_iter_no_change': 9.853412557152556, 'penalty': 'l2', 'power_t': 0.4343839765964862, 'tol': 0.0023648882307393125, 'validation_fraction': 0.8948412725060997, 'warm_start': False}

