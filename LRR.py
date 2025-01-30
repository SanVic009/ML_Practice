import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class LR:
    
    def __init__(self):
        self.b = None
        self.m = None
        
    def fit(self, x_train, y_train):

        num = 0
        den = 0
        for i in range(x_train.shape[0]):
            print(f"For iteration {i}, num = {num}, den = {den}")
            num = num + (x_train[i] - x_train.mean()) * (y_train[i] - y_train.mean())
            den = den + (x_train[i] - x_train.mean()) *  (x_train[i] - x_train.mean())

        
        self.m = num / den
        self.b = y_train.mean() - (self.m * x_train.mean())
        print(self.m)
        print(self.b)
        
    def predict(self, x_test):
        pred = []
        for i in range(x_test.shape[0]):
            pred.append((self.m * x_test[i]) + self.b)
        
        return pred


df = pd.read_csv('S:/Github/ML_Practice/datasets/package.csv') 
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 2)

lr = LR()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

# print(y_pred)
print(y_test)
# print(lr.b)
# print(lr.m)