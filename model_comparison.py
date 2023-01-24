from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,HuberRegressor,LinearRegression,LogisticRegression,SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':

    models = [['DecisionTree :',DecisionTreeRegressor()],
            ['Linear Regression :', LinearRegression()],
            ['RandomForest :',RandomForestRegressor()],
            ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)],
            ['SVM :', SVR()],
            ['AdaBoostClassifier :', AdaBoostRegressor()],
            ['GradientBoostingClassifier: ', GradientBoostingRegressor()],
            ['Xgboost: ', XGBRegressor()],
            ['CatBoost: ', CatBoostRegressor(logging_level='Silent')],
            ['Lasso: ', Lasso()],
            ['Ridge: ', Ridge()],
            ['BayesianRidge: ', BayesianRidge()],
            ['ElasticNet: ', ElasticNet()],
            ['HuberRegressor: ', HuberRegressor()]]

    X = 
    
    k_folds = 5
    scoring = "neg_mean_squared_error"
    #neg_mean_absolute_error
    #neg_mean_squared_error
    #neg_root_mean_squared_error
    #neg_mean_squared_log_error

    clf = svm.SVC(kernel='linear', C=1, random_state=42)
    scores = cross_val_score(clf, X, y, cv=k_folds)

    print("Results...")
    for name,model in models:
        model = model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(name, (np.sqrt(mean_squared_error(y_test, predictions))))