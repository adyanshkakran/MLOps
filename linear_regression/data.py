import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(separateTest = False):
    if separateTest:
        data = pd.read_csv('data/data.csv')
        test = pd.read_csv('data/test.csv')

        x_train = data.iloc[:, 1:7]
        y_train = data.iloc[:, 7]

        x_test = test.iloc[:, 1:7]
        y_test = test.iloc[:, 7]
    else:
        data = pd.read_csv('data/data.csv')

        X = data.iloc[:, 1:7]
        Y = data.iloc[:, 7]

        x_train, x_test, y_train, y_test = train_test_split(X, Y)

    return x_train, x_test, y_train, y_test