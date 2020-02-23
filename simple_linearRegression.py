# -*- coding: utf-8 -*-
#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =1/3, random_state =0)

#fitting the linearRegression in our trainig set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Analyzing the test Set(Predicting)
y_pred = regressor.predict(X_test)

#Visualizing the prediction(Training_Set) in plots
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color ='blue')
plt.title('Salary vs Experience(Training_Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
 
#Visualizing the prediction(test_set) in plots
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color ='blue')
plt.title('Salary vs Experience(Test_Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()