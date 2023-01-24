import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def clip(x): return min(1, max(0, x))

if __name__ == '__main__':
    pwd = os.path.realpath(os.path.dirname(__file__))

    pred_csv = pwd + '/data/prediction.csv'

    y_csv = pwd + '/data/clean/data_y.csv'
    x_csv = pwd + '/data/clean/data_x.csv'
    
    x_test_csv = pwd + '/data/original/Graduating_Class.csv'
    
    X = []
    with open(x_csv, "r") as f:
        for entry in f.read().split('\n'):
            X.append(list(map(float, entry.split(","))))
    X = StandardScaler().fit_transform(X)
    Y = []
    with open(y_csv, "r") as f:
        for entry in f.read().split('\n'):
            Y.append(float(entry))
    X_test = []
    X_test_fit = []
    with open(x_test_csv, "r") as f:
        for entry in f.read().split('\n')[1:]:
            X_test.append(entry)
            X_test_fit.append(list(map(float, entry.split(","))))
    X_test_fit = StandardScaler().fit_transform(X)

    model = LinearRegression() 
    model.fit(X, Y)
    Y_pred = model.predict(X_test_fit)
    Y_pred = list(map(clip, Y_pred))
    out = "Serial_No,GRE_Score,TOEFL_Score,University_Rating,SOP,LOR,CGPA,Research\n"
    for x, y in zip(X_test,Y_pred):
        out += x + ","
        out += str(y) + "\n"
    with open(pred_csv, "w+") as f:
        f.write(out[:-1]) #remove trailing new line character
