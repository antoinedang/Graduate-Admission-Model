import os
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

def clip(x): return min(1, max(0, x))

if __name__ == '__main__':    
    pwd = os.path.realpath(os.path.dirname(__file__))
    train_csv = pwd + '/data/Past_Students.csv'
    test_csv = pwd + '/data/Graduating_Class.csv'
    pred_csv = pwd + '/data/Antoine_Dangeard_Results.csv'

    X = []
    Y = []
    with open(train_csv, "r") as f:
        for entry in f.read().split('\n')[1:]:
            X.append(list(map(float, entry.split(",")))[1:-1])
            Y.append(list(map(float, entry.split(",")))[-1])
    X = StandardScaler().fit_transform(X)

    X_test = []
    X_test_fit = []
    with open(test_csv, "r") as f:
        for entry in f.read().split('\n')[1:]:
            X_test.append(entry)
            X_test_fit.append(list(map(float, entry.split(",")))[1:])
    X_test_fit = StandardScaler().fit_transform(X)

    model = SGDRegressor(alpha=0.0075, eta0=0.075, learning_rate='constant', loss='squared_error', penalty='l2', tol=0.007) 
    model.fit(X, Y)
    Y_pred = model.predict(X_test_fit)
    Y_pred = list(map(clip, Y_pred))
    out = "Serial_No,GRE_Score,TOEFL_Score,University_Rating,SOP,LOR,CGPA,Research\n"
    for x, y in zip(X_test,Y_pred):
        out += x + ","
        out += str(y) + "\n"
    with open(pred_csv, "w+") as f:
        f.write(out[:-1]) #remove trailing new line character
