import pandas as pd
import numpy as np
import mglearn as mg
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import csv
import os

import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/static/RainyDaysHero/data/LI/TI/dataset.csv")

X =df[['age','nb_payements','maturity','interest_rate','amount']]

y = df['target']


X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=5)

X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)

def get_r2_statsmodels(x, y, k=1):
    xpoly = np.column_stack([x**i for i in range(k+1)])    
    return sm.OLS(y, xpoly).fit().rsquared


def Polynomial_scaled(degree=8,X_train=X_train,y_train=y_train):
    
    scaler = MinMaxScaler()
    X_train_scaled  = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_valid_scaled=scaler.transform(X_valid)
    
    poly_scaled= PolynomialFeatures(degree=degree,include_bias=False)
    poly_scaled.fit(X_train_scaled)

    X_poly_train_scaled  = poly_scaled.transform(X_train_scaled)
    X_poly_test_scaled = poly_scaled.transform(X_test_scaled)
    X_poly_valid_scaled  = poly_scaled.transform(X_valid_scaled)

    
    model= LinearRegression().fit(X_poly_train_scaled, y_train)
    y_train_predict = model.predict( X_poly_train_scaled) 
    y_valid_predict = model.predict(X_poly_valid_scaled)     
    y_test_predict = model.predict(X_poly_test_scaled)

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
    p1,=plt.plot(liste_erreurs[3,],liste_erreurs[0,],label='train')
    p2,=plt.plot(liste_erreurs[3,],liste_erreurs[1,],label='validation')
    p3,=plt.plot(liste_erreurs[3,],liste_erreurs[2,],label='test')
    plt . xlabel ('degree', fontsize =20)
    plt . ylabel ('R²', fontsize =20)
    plt . title ('R² as a function of the degree',fontsize =16)
    plt . legend ( handles =[p1 , p2, p3],fontsize =16)
    plt.show()

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

     p1,=plt.plot(listei,r2train,label='train')
     p2,=plt.plot(listei,r2valid,label='valid')
     p3,=plt.plot(listei,r2test,label='test')
     plt . xlabel ('training set size', fontsize =20)
     plt . ylabel ('R²', fontsize =20)
     plt . title ('learning curve',fontsize =16)
     plt . legend ( handles =[p1 , p2,p3],fontsize =16)
     plt.show()    
     return r2train,r2test, r2valid,listei     

def term_insurance_predicted_polynomiale_scaled(x,m,n,i,a,degree=4):
    if (m>n):
        return('error')
    data=[[x,m,n,i,a]] 
    premium_to_predict=pd.DataFrame(data=data,columns=['age','nb_payements','maturity','interest_rate','amount'])
    
    scaler = MinMaxScaler()
    X_train_scaled  = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(premium_to_predict)
    
    poly_scaled= PolynomialFeatures(degree=degree,include_bias=False)
    poly_scaled.fit(X_train_scaled)

    X_poly_train_scaled  = poly_scaled.transform(X_train_scaled)
    X_poly_test_scaled = poly_scaled.transform(X_test_scaled)

    
    model= LinearRegression().fit(X_poly_train_scaled, y_train)
    y_test_predict = model.predict(X_poly_test_scaled)

    return  f'{np.abs(y_test_predict[0]):.2f}'     


def learning_curve_poly_scaled_3d(N):
     X_train_new=list()
     
     r2test=np.zeros((N,N))
     
     listei=list()
     degreei=list()     
     for degree in range(0,N):
         degreei.append(degree+1)         
         poly_scaled= PolynomialFeatures(degree=degree+1,include_bias=False)

         for j in range(1,N+1):
             i=len(X_train)//N*j
             X_train_new=X_train[0:i]
             y_train_new=y_train[0:i]
        
             scaler = MinMaxScaler()
             X_train_scaled  = scaler.fit_transform(X_train_new)
             X_test_scaled = scaler.transform(X_test)
            
             poly_scaled.fit(X_train_scaled)
        
             X_poly_train_scaled  = poly_scaled.transform(X_train_scaled)
             X_poly_test_scaled = poly_scaled.transform(X_test_scaled)


             model= LinearRegression().fit(X_poly_train_scaled, y_train_new)             
             y_predict=model.predict(X_poly_test_scaled)
             r2test[degree,j-1]= get_r2_statsmodels(y_predict, y_test, k=1)
             listei.append(i)        
     fig = figure()
     ax = Axes3D(fig)
     ax.set_xlabel('training set size')
     ax.set_ylabel('degree')   
     ax.set_zlabel('R²')       
     X, Y = np.meshgrid( np.arange(len(X_train)//N,len(X_train),len(X_train)//N),degreei)
     
     ax.plot_surface(X, Y, r2test , rstride=1, cstride=1, cmap='hot')
     show()
    
   
   
def profit_and_loss_age_scaled(x,m,n,i,a,degree):
    age=list()
    p_and_l=list()
    for x in range(1,90):
        p_and_l.append(profit_and_loss_scaled(x,m,n,i,a,degree))
        age.append(x)
    plt.plot(age,p_and_l)
    plt . xlabel ('age', fontsize =20)
    plt . ylabel ('profit_and_loss', fontsize =20)
    plt . title ('profit and loss as a function of the age',fontsize =16)
    plt.show()    
    
def profit_and_loss_interest_rate_scaled(x,m,n,i,a,degree):
    interest=list()
    p_and_l=list()
    for i in range(1,1000):
        p_and_l.append(profit_and_loss_scaled(x,m,n,i/1000,a,degree))
        interest.append(i/1000)
    plt.plot(interest,p_and_l)
    plt . xlabel ('interest_rate', fontsize =20)
    plt . ylabel ('profit_and_loss', fontsize =20)
    plt . title ('profit and loss as a function of the interest rate',fontsize =16)

def profit_and_loss_amount_scaled(x,m,n,i,a,degree):
    premium=list()
    p_and_l=list()
    for a in range(1,50000,10):
        p_and_l.append(profit_and_loss_scaled(x,m,n,i,a,degree))
        premium.append(a)
    plt.plot(premium,p_and_l)
    plt . xlabel ('insured amounts', fontsize =20)
    plt . ylabel ('profit_and_loss', fontsize =20)
    plt . title ('profit and loss as a function of the insured amount',fontsize =16)

def profit_and_loss_maturity_scaled(x,m,n,i,a,degree):
    maturity=list()
    p_and_l=list()
    for n in range(m+1,105-x):
        p_and_l.append(profit_and_loss_scaled(x,m,n,i,a,degree))
        maturity.append(n)
    plt.plot(maturity,p_and_l)
    plt . xlabel ('maturity', fontsize =20)
    plt . ylabel ('profit_and_loss', fontsize =20)
    plt . title ('profit and loss as a function of the maturity',fontsize =16)

def profit_and_loss_payements_scaled(x,m,n,i,a,degree):
    payements=list()
    p_and_l=list()
    for m in range(1,n):
        p_and_l.append(profit_and_loss_scaled(x,m,n,i,a,degree))
        payements.append(m)
    plt.plot(payements,p_and_l)
    plt . xlabel ('number of payements', fontsize =20)
    plt . ylabel ('profit_and_loss', fontsize =20)
    plt . title ('profit and loss as a function of the number of payements',fontsize =16)