import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)



plt.scatter(X, Y, color ="red")
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.title("Level vs Salary (Polynomial Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


lin_reg.predict([[6.5]]) #It is used with linear regression
x = lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
print(x)
