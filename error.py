import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/package.csv')

x = df.iloc[:, 0:1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)

lr = LinearRegression()

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

sns.scatterplot(x= y_pred, y= y_test)