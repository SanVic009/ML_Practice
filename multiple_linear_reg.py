import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# x, y = load_diabetes(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 2)

# lr = LinearRegression()

# lr.fit(x_train, y_train)

# y_pred = lr.predict(x_test)

# print(r2_score(y_test, y_pred))

# lr.coef_
# lr.intercept_

class MLR:
    def __init(self):
        self.coeff_ = None
        self.intercept_ = None

    def fit(x_train, y_train, self):
        x_train = np.insert(x_train, 0, 1, axis= 1)
        betas = np.linalg.inv(np.dot(x_train.T, x_train)).dot(x_train).dot(y_train)
        
        self.intercept_ = betas[0]
        self.coeff_ = betas[1:]

    def preditct(self, x_test):
        
        pass