from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lars,Lasso,LassoLars,Ridge,BayesianRidge,ElasticNet,HuberRegressor,LinearRegression,SGDRegressor,PassiveAggressiveRegressor,ARDRegression,OrthogonalMatchingPursuit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils import shuffle
import os
import numpy as np

if __name__ == '__main__':
    pwd = os.path.realpath(os.path.dirname(__file__))
    x_csv = pwd + '/data/clean/data_x.csv'
    X = []
    with open(x_csv, "r") as f:
        for entry in f.read().split('\n'):
            X.append(list(map(float, entry.split(","))))
    X = StandardScaler().fit_transform(X)
    y_csv = pwd + '/data/clean/data_y.csv'
    Y = []
    with open(y_csv, "r") as f:
        for entry in f.read().split('\n'):
            Y.append(float(entry))

    scoring = "neg_mean_squared_error"
    k_fold_max = 7 #number of folds to cut data into for training
    num_shuffles = 25 #number of times to score models (data shuffled every time)
    print("Initializing Models...")
    models = [
            ['DecisionTreeRegressor :',DecisionTreeRegressor()],
            ['LinearRegression :', LinearRegression()],
            ['PassiveAggressiveRegressor :', PassiveAggressiveRegressor()],
            ['OrthogonalMatchingPursuit :', OrthogonalMatchingPursuit(normalize=False)],
            ['MLPRegressor :', MLPRegressor()],
            ['SGDRegressor :', SGDRegressor()],
            ['RandomForestRegressor :', RandomForestRegressor()],
            ['KNeighborsRegressor :', KNeighborsRegressor()],
            ['SVR :', SVR()],
            ['NuSVR :', NuSVR()],
            ['AdaBoostRegressor :', AdaBoostRegressor()],
            ['GradientBoostingRegressor: ', GradientBoostingRegressor()],
            ['PLSRegression: ', PLSRegression()],
            ['Lasso: ', Lasso()],
            ['Lars: ', Lars(normalize=False)],
            ['LassoLars: ', LassoLars(normalize=False)],
            ['ARDRegression: ', ARDRegression()],
            ['Ridge: ', Ridge()],
            ['KernelRidge: ', KernelRidge()],
            ['BayesianRidge: ', BayesianRidge()],
            ['ElasticNet: ', ElasticNet()],
            ['HuberRegressor: ', HuberRegressor()]]

    for k_fold in range(k_fold_max-1): #get worst score for each model for each k-fold value
        worst_scores = {} #save worst score for each model over all attempts with this k-fold
        for i in range(num_shuffles):
            X, Y = shuffle(X, Y)
            for j in range(len(models)):
                name, model = models[j]
                scores = cross_val_score(model, X, Y, cv=k_fold+2, scoring=scoring)
                worst_scores[name] = min(min(scores), worst_scores.get(name, 0))
                current_k_progress = ((j+1)+(i*len(models)))/(len(models)*num_shuffles)
                overall_progress = (k_fold + current_k_progress)/(k_fold_max)
                print("Scoring models... " + str(100*overall_progress) + "%                ", end="\r")
    print("Final results w max. " + str(k_fold_max) + " k-folds:           \n" + str(worst_scores))


    #Using this method best (most generalizable) models are:
    #Note this is without any hyperparameter tuning
    #Some models could perform better than those listed below after hyperparameter tuning
    #However due to timing contraints, per-model hyperparameter tuning for all models is not feasible
    #Therefore, hyperparameter tuning will only be done on the below models (subject to change)
        # LinearRegression
        # SGDRegressor
        # Lars
        # Ridge
        # BayesianRidge
    #Showed potential
        # OrthogonalMatchinPursuit
        # RandomForestRegressor
        # KNeighboursRegressor
        # NuSVR
        # AdaBoostRegressor
        # GradientBoostingRegressor
        # PLSRegression
        # ARDRegression
        # HuberRegressor