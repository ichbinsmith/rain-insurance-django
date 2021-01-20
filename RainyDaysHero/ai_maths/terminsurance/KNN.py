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
from sklearn.neighbors import KNeighborsRegressor
import matplotlib
matplotlib.use('Agg')
import csv
import os
from mpl_toolkits.mplot3d import Axes3D
from pylab import show,figure
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/static/RainyDaysHero/data/LI/TI/dataset.csv")

X =df[['age','nb_payements','maturity','interest_rate','amount']]

y = df['target']

def best_knn():
    listnn=list()
    listR_train=list()
    listR_valid=list()
    listR_test=list()
    for nn in range(1,50):
        listnn.append(nn)
        reg = KNeighborsRegressor(n_neighbors=nn,weights="distance")
        reg.fit(X_train, y_train)
        listR_train.append(reg.score(X_train, y_train))
        listR_valid.append(reg.score(X_valid, y_valid))
        listR_test.append(reg.score(X_test, y_test))        
    p1,=plt.plot(listnn,listR_train,label='train')
    p2,=plt.plot(listnn,listR_valid,label='validation')
    p3,=plt.plot(listnn,listR_test,label='test')
    plt . xlabel ('number of neighbors', fontsize =20)
    plt . ylabel ('R²', fontsize =20)
    plt . title ('R² as a function of the number of neighbors',fontsize =16)
    plt . legend ( handles =[p1 , p2, p3],fontsize =16)
    plt.show()
        

def learning_curve_knn(nn):
     X_train_new=np.zeros(len(X_train)-50)
     r2train=np.zeros(len(X_train)-50)
     r2test=np.zeros(len(X_train)-50)
     r2valid=np.zeros(len(X_train)-50)     
     listei=np.zeros(len(X_train)-50)    
     for i in range(50,len(X_train)):
         X_train_new=X_train[0:i]
         y_train_new=y_train[0:i]
         reg = KNeighborsRegressor(n_neighbors=nn,weights="distance")
         reg.fit(X_train_new, y_train_new)         

         r2train[i-50] = reg.score(X_train, y_train)
         r2test[i-50] = reg.score(X_test, y_test)
         r2valid[i-50] = reg.score(X_valid, y_valid)         
         listei[i-50]=i
      
     p1,=plt.plot(listei,r2train,label='train')
     p2,=plt.plot(listei,r2test,label='test')
     p3,=plt.plot(listei,r2valid,label='valid')     
     plt . xlabel ('training set size', fontsize =20)
     plt . ylabel ('R²', fontsize =20)
     plt . title ('learning curve',fontsize =16)
     plt . legend ( handles =[p1 , p2, p3],fontsize =16)
     plt.show()    
     return r2train,r2test, r2valid

     

def learning_curve_knn_3d(N):
     X_train_new=list()    
     r2test=np.zeros((N,N))     
     listei=list()
     degreei=list()     
     for degree in range(1,N+1):
         degreei.append(degree+1)         
         reg = KNeighborsRegressor(n_neighbors=degree,weights="distance")

         for j in range(1,N+1):
             
             i=len(X_train)//N*j
             X_train_new=X_train[0:i]
             y_train_new=y_train[0:i]
             reg.fit(X_train_new, y_train_new) 
            
             r2test[degree-1,j-1]= reg.score(X_test,y_test)
             listei.append(i)        
     fig = figure()
     ax = Axes3D(fig)
     ax.set_xlabel('training set size')
     ax.set_ylabel('Number of neighbors')   
     ax.set_zlabel('R²')       
     X, Y = np.meshgrid( np.arange(len(X_train)//N,len(X_train),len(X_train)//N),degreei)
     
     ax.plot_surface(X, Y, r2test , rstride=1, cstride=1, cmap='hot')
     show() 
    

     
def term_insurance_predicted_knn(x,m,n,i,a,nn=10):
    if (m>n):
        return('error')
    premium_to_predict=[[x,m,n,i,a]] 
    model=KNeighborsRegressor(n_neighbors=nn,weights="distance")
    model.fit(X_train, y_train)      
    y_test_predict = model.predict(premium_to_predict)
    return(y_test_predict[0])    
    

def profit_and_loss_knn(x,m,n,i,a,nn=10):
    if (m>n):
        return('error')
  
    actuarial=TermInsuranceAnnual(x,m,n,i,a)
    machine_learning=term_insurance_predicted_knn(x,m,n,i,a,nn)
    print('premium computed with actuarial method : ' ,actuarial)
    print('premium computed with machine learning method: ' ,machine_learning)
    P_and_L=machine_learning-actuarial
    return( P_and_L)     
    

def knn_model(X_train,y_train,nn):
    model=KNeighborsRegressor(n_neighbors=nn,weights="distance")
    model.fit(X_train, y_train)      
    return model    


