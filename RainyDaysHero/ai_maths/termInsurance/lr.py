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
     p1,=plt.plot(listei,r2train,label='train')
     p2,=plt.plot(listei,r2valid,label='valid')
     p3,=plt.plot(listei,r2vtest,label='test')     
     plt . xlabel ('training set size', fontsize =20)
     plt . ylabel ('R²', fontsize =20)
     plt . title ('learning curve',fontsize =16)
     plt . legend ( handles =[p1 , p2, p3],fontsize =16)
     plt.show()    
     return r2train,r2test, r2valid, listei  


def plot_polynomiale(degremax):
    liste_erreurs=np.zeros((4,degremax))
    for i in range(1,degremax):
        liste_erreurs[0,i]=(Polynomiale(i)[0])
        liste_erreurs[1,i]=(Polynomiale(i)[1])
        liste_erreurs[2,i]=(Polynomiale(i)[2])
        liste_erreurs[3,i]=i

    p1,=plt.plot(liste_erreurs[3,],liste_erreurs[0,],label='train')
    p2,=plt.plot(liste_erreurs[3,],liste_erreurs[1,],label='validation')
    p3,=plt.plot(liste_erreurs[3,],liste_erreurs[2,],label='test')
    plt . xlabel ('degree', fontsize =20)
    plt . ylabel ('R²', fontsize =20)
    plt . title ('R² as a function of the degree',fontsize =16)
    plt . legend ( handles =[p1 , p2, p3],fontsize =16)
    plt.show()


def learning_curve_poly_3d(N):
     X_train_new=list()
     
     r2test=np.zeros((N,N))
     
     listei=list()
     degreei=list()     
     for degree in range(0,N):
         polynomial_features= PolynomialFeatures(degree+1)
         degreei.append(degree+1)
         for j in range(1,N+1):
             i=len(X_train)//N*j
             X_train_new=X_train[0:i]
             y_train_new=y_train[0:i]
             X_train_poly=polynomial_features.fit_transform(X_train_new)
             X_test_poly=polynomial_features.fit_transform(X_test)
             model = LinearRegression()
             model.fit(X_train_poly, y_train_new)         

             y_test_predict = model.predict(X_test_poly)  
             
             r2test[degree,j-1]=(get_r2_statsmodels(y_test_predict, y_test)) 
             listei.append(i)        
     fig = figure()
     ax = Axes3D(fig)
     ax.set_xlabel('training set size')
     ax.set_ylabel('degree')   
     ax.set_zlabel('R²')       
     X, Y = np.meshgrid( np.arange(len(X_train)//N,len(X_train),len(X_train)//N),degreei)
     
     ax.plot_surface(X, Y, r2test , rstride=1, cstride=1, cmap='hot')
     show()    
     



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

