import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import os

import statsmodels.api as sm

df = pd.read_csv(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/static/RainyDaysHero/data/LI/TI/dataset.csv")

X =df[['age','nb_payements','maturity','interest_rate','amount']]

y = df['target']


X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=5)

X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)

def get_r2_statsmodels(x, y, k=1):
    xpoly = np.column_stack([x**i for i in range(k+1)])    
    return sm.OLS(y, xpoly).fit().rsquared

def model_Polynomiale_learning(degree=1,X_train=X_train,y_train=y_train):
    polynomial_features= PolynomialFeatures(degree=degree)
    X_train_poly=polynomial_features.fit_transform(X_train)    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    return model

def learning_curve(degree=1):
     polynomial_features= PolynomialFeatures(degree)
     r2train=np.zeros(len(X_train)-50)
     r2test=np.zeros(len(X_train)-50)
     r2valid=np.zeros(len(X_train)-50)
     listei=np.zeros(len(X_train)-50)
     for i in range(50,len(X_train)):
         print(i)
         X_train_new=X_train[0:i]
         y_train_new=y_train[0:i]
         X_train_poly=polynomial_features.fit_transform(X_train_new)
         X_test_poly=polynomial_features.fit_transform(X_test)
         X_valid_poly=polynomial_features.fit_transform(X_valid)
         model = LinearRegression()
         model.fit(X_train_poly, y_train_new)         
         y_train_predict = model.predict(X_train_poly)

         r2train[i-50] = get_r2_statsmodels(y_train_new, y_train_predict)
 
         y_valid_predict = model.predict(X_valid_poly)
         r2valid[i-50] = get_r2_statsmodels(y_valid, y_valid_predict)
         
        
         y_test_predict = model.predict(X_test_poly)
         r2test[i-50] = get_r2_statsmodels(y_test, y_test_predict)

         listei[i-50]=i    
  
     return r2train,r2test, r2valid, listei  






def term_insurance_predicted_polynomiale_no_constraint(x,m,n,i,a,degree=1):
    big=model_Polynomiale_learning(degree,X_train,y_train)
    if (m>n):
        return('error')
    data=[[x,m,n,i,a]] 
    premium_to_predict=pd.DataFrame(data=data,columns=['age','nb_payements','maturity','interest_rate','amount'])
    polynomial_features=PolynomialFeatures(degree=degree)
    premium_to_predict = polynomial_features.fit_transform(premium_to_predict)
    final_premium = big.predict(premium_to_predict)  
    return f'{final_premium[0]:.2f}'   

