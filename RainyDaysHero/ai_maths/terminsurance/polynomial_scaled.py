import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
import os
from sklearn.neighbors import KNeighborsRegressor

import statsmodels.api as sm

df = pd.read_csv(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/static/RainyDaysHero/data/LI/TI/dataset.csv")

X =df[['age','nb_payements','maturity','interest_rate','amount']]

y = df['target']


X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=1)

X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)

def get_r2_statsmodels(x, y, k=1):
    xpoly = np.column_stack([x**i for i in range(k+1)])    
    return sm.OLS(y, xpoly).fit().rsquared


def Polynomial_scaled(degree=8,X_train=X_train,y_train=y_train):
    
    scaler = MinMaxScaler()
    X_train_scaled  = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X)    
    poly_scaled= PolynomialFeatures(degree=degree,include_bias=False)
    poly_scaled.fit(X_train_scaled)
    X_poly_train_scaled  = poly_scaled.transform(X_train_scaled)
    X_poly_test_scaled = poly_scaled.transform(X_test_scaled)
    model= LinearRegression().fit(X_poly_train_scaled, y_train)
    y_poly = model.predict(X_poly_test_scaled)
    y_poly=np.abs(y_poly)
    ##now we train knn
    model=KNeighborsRegressor(n_neighbors=20,weights="distance")
    model.fit(X, y_poly)    
    

    y_train_predict = model.predict( X_train) 
    y_valid_predict = model.predict(X_valid)     
    y_test_predict = model.predict(X_test)

    r2train = get_r2_statsmodels(y_train, y_train_predict)

    r2valid = get_r2_statsmodels(y_valid, y_valid_predict)

    r2test = get_r2_statsmodels(y_test, y_test_predict)

    return r2train,r2valid,r2test, y_test_predict, model    

def plot_polynomiale_scaled(degremax):
    liste_erreurs=np.zeros((4,degremax))
    for i in range(1,degremax):
        liste_erreurs[0,i]=(Polynomial_scaled(degree=i,X_train=X_train,y_train=y_train)[0])
        liste_erreurs[1,i]=(Polynomial_scaled(degree=i,X_train=X_train,y_train=y_train)[1])
        liste_erreurs[2,i]=(Polynomial_scaled(degree=i,X_train=X_train,y_train=y_train)[2])
        liste_erreurs[3,i]=i


def learning_curve_poly_scaled(degree=4):
     X_train_new=np.zeros(len(X_train)-50)
     r2train=np.zeros(len(X_train)-50)
     r2test=np.zeros(len(X_train)-50)
     r2valid=np.zeros(len(X_train)-50)     
     listei=np.zeros(len(X_train)-50)
     for i in range(50,len(X_train)):
         print(i)
         X_train_new=X_train[0:i]
         y_train_new=y_train[0:i]
         predictions = Polynomial_scaled(degree=8,X_train=X_train_new,y_train=y_train_new)        
         r2train[i-50] = predictions[0]  
         r2valid[i-50] = predictions[1]                           
         r2test[i-50] = predictions[2]   
         listei[i-50]=i
   
     return r2train,r2test, r2valid,listei     


def term_insurance_predicted_polynomiale_scaled(x,m,n,i,a,degree=8):
    if (m>n):
        return('error')
    data=[[x,m,n,i,a]] 
    premium_to_predict=pd.DataFrame(data=data,columns=['age','nb_payements','maturity','interest_rate','amount'])

    scaler = MinMaxScaler()
    X_train_scaled  = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X)    
    poly_scaled= PolynomialFeatures(degree=degree,include_bias=False)
    poly_scaled.fit(X_train_scaled)
    X_poly_train_scaled  = poly_scaled.transform(X_train_scaled)
    X_poly_test_scaled = poly_scaled.transform(X_test_scaled)
    model= LinearRegression().fit(X_poly_train_scaled, y_train)
    y_poly = model.predict(X_poly_test_scaled)
    y_poly=np.abs(y_poly)
    ##now we train knn
    
    model=KNeighborsRegressor(n_neighbors=20,weights="distance")
    model.fit(X, y_poly)

    y_test_predict = model.predict(premium_to_predict)

    return  f'{np.abs(y_test_predict[0]):.2f}' 

