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

def model_small_premium_learning(degree,X_train,y_train):
    smallX_train=X_train.copy()
    smally_train=y_train.copy()
    for i in y_train.index:
        if (y_train[i]>1000):
            smallX_train.drop([i],inplace=True)
            smally_train.drop([i],inplace=True)    
    polynomial_features= PolynomialFeatures(degree=degree)
    smallX_train_poly = polynomial_features.fit_transform(smallX_train)    
    modelsmall = LinearRegression()
    modelsmall.fit(smallX_train_poly, smally_train)
    return(modelsmall)

    
def model_Polynomiale_learning(degree,X_train=X_train,y_train=y_train):
    polynomial_features= PolynomialFeatures(degree=degree)
    X_train_poly=polynomial_features.fit_transform(X_train)    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    return model



def model_under0_premium_learning(degree,X_train,y_train):
    polynomial_features= PolynomialFeatures(degree=degree)
    smallX_train=X_train.copy()
    smally_train=y_train.copy()
    X_train_poly = polynomial_features.fit_transform(X_train)    
    y_train_predict = model_Polynomiale_learning(degree).predict(X_train_poly)
    j=0
    for i in y_train.index:
        if (y_train_predict[j]>0):
            smallX_train.drop([i],inplace=True)
            smally_train.drop([i],inplace=True) 
        j+=1
    smallX_train_poly = polynomial_features.fit_transform(smallX_train)            
    modelsmall = LinearRegression()
    modelsmall.fit(smallX_train_poly, smally_train)
    return(modelsmall)

def polyfinal_learning_under0(degree=6,X_train=X_train,y_train=y_train):
    small= model_small_premium_learning(degree,X_train,y_train)
    big=model_Polynomiale_learning(degree,X_train,y_train)
    under0=model_under0_premium_learning(degree,X_train,y_train)
    
    y_predictionfinale=[]
    y_predictiontrain=[]
    y_predictionvalid=[]
    
    polynomial_features=PolynomialFeatures(degree=degree)
    X_test_poly = polynomial_features.fit_transform(X_test)
    X_train_poly=polynomial_features.fit_transform(X_train)        
    X_valid_poly=polynomial_features.fit_transform(X_valid)        

    y_test_predict = big.predict(X_test_poly)
    y_train_predict = big.predict(X_train_poly) 
    y_valid_predict = big.predict(X_valid_poly) 
   
    for i in range(0,len(y_test_predict)):
        if (y_test_predict[i]<=1000 and 0<y_test_predict[i]):
            a=small.predict(X_test_poly[[i]])[0]
            if (a>0):
                y_predictionfinale.append(a)
            else:
                y_predictionfinale.append(big.predict(X_test_poly[[i]])[0])
        if (y_test_predict[i]>1000):
            y_predictionfinale.append(big.predict(X_test_poly[[i]])[0])
        if (y_test_predict[i]<0):
            y_predictionfinale.append(under0.predict(X_test_poly[[i]])[0])
        if (y_predictionfinale[i]<0):    
            y_predictionfinale[i]=10
            
    for i in range(0,len(y_train_predict)):
        if (y_train_predict[i]<=1000 and 0<y_train_predict[i] ):
            a=small.predict(X_train_poly[[i]])[0]
            if (a>0):
                y_predictiontrain.append(a)
            else:
                y_predictiontrain.append(big.predict(X_train_poly[[i]])[0])            

        if (y_train_predict[i]>1000):
            y_predictiontrain.append(big.predict(X_train_poly[[i]])[0])
        if (y_train_predict[i]<0):
            y_predictiontrain.append(under0.predict(X_train_poly[[i]])[0])
        if (y_predictiontrain[i]<0):    
            y_predictiontrain[i]=10
            
    for i in range(0,len(y_valid_predict)):
        if (y_valid_predict[i]<=1000 and 0<y_valid_predict[i]):
            a=small.predict(X_valid_poly[[i]])[0]
            if (a>0):
                y_predictionvalid.append(a)
            else:
                y_predictionvalid.append(big.predict(X_valid_poly[[i]])[0])            
        if (y_valid_predict[i]>1000):
            y_predictionvalid.append(big.predict(X_valid_poly[[i]])[0])
        if (y_valid_predict[i]<0):
            y_predictionvalid.append(under0.predict(X_valid_poly[[i]])[0])
        if (y_predictionvalid[i]<0):    
            y_predictionvalid[i]=10

    r2test = get_r2_statsmodels(y_test, y_predictionfinale)
    r2train = get_r2_statsmodels(y_train, y_predictiontrain)          
    r2valid = get_r2_statsmodels(y_valid, y_predictionvalid)              
    
    return(y_predictionfinale,r2test,r2train,r2valid,y_predictionvalid)      


def learning_curve_polyfinal_under0(degree=6):
     X_train_new=np.zeros(len(X_train)-100)
     r2train=np.zeros(len(X_train)-100)
     r2test=np.zeros(len(X_train)-100)
     r2valid=np.zeros(len(X_train)-100)     
     listei=np.zeros(len(X_train)-100)
     for i in range(100,len(X_train)):
         print(i)
         X_train_new=X_train[0:i]
         y_train_new=y_train[0:i]
         predictions = polyfinal_learning_under0(degree,X_train_new,y_train_new)         
         r2train[i-100] = predictions[2]         
         r2test[i-100] = predictions[1]   
         r2valid[i-100] = predictions[3]                  
         listei[i-100]=i

     p1,=plt.plot(listei,r2train,label='train')
     p2,=plt.plot(listei,r2valid,label='valid')
     p3,=plt.plot(listei,r2test,label='test')
     plt . xlabel ('training set size', fontsize =20)
     plt . ylabel ('R²', fontsize =20)
     plt . title ('learning curve',fontsize =16)
     plt . legend ( handles =[p1 , p2,p3],fontsize =16)
     #plt.show()    
     return r2train,r2test, r2valid,listei  


def plot_polyfinal_under0(degremax):
    liste_erreurs=np.zeros((4,degremax))
    for i in range(1,degremax):
        liste_erreurs[0,i]=(polyfinal_learning_under0(i)[2])
        liste_erreurs[1,i]=(polyfinal_learning_under0(i)[3])
        liste_erreurs[2,i]=(polyfinal_learning_under0(i)[1])
        liste_erreurs[3,i]=i

    p1,=plt.plot(liste_erreurs[3,],liste_erreurs[0,],label='train')
    p2,=plt.plot(liste_erreurs[3,],liste_erreurs[1,],label='validation')
    p3,=plt.plot(liste_erreurs[3,],liste_erreurs[2,],label='test')
    plt . xlabel ('degree', fontsize =20)
    plt . ylabel ('R²', fontsize =20)
    plt . title ('R² as a function of the degree',fontsize =16)
    plt . legend ( handles =[p1 , p2, p3],fontsize =16)
    #plt.show()





def term_insurance_predicted(x,m,n,i,a,degree):
    small= model_small_premium_learning(degree,X_train,y_train)
    big=model_Polynomiale_learning(degree,X_train,y_train)
    under0=model_under0_premium_learning(degree,X_train,y_train)
    if (m>n):
        return('error')
    data=[[x,m,n,i,a]] 
    premium_to_predict=pd.DataFrame(data=data,columns=['age','nb_payements','maturity','interest_rate','amount'])
    polynomial_features=PolynomialFeatures(degree=degree)
    premium_to_predict = polynomial_features.fit_transform(premium_to_predict)
    premium_predicted_vanilla = big.predict(premium_to_predict)
    
    if (premium_predicted_vanilla<=1000 and 0<premium_predicted_vanilla):
        final_premium=small.predict(premium_to_predict)
        if (final_premium<0):
            final_premium=big.predict(premium_to_predict)         
    if (premium_predicted_vanilla>1000):
        final_premium=big.predict(premium_to_predict)        
    if (premium_predicted_vanilla<0):
        final_premium=under0.predict(premium_to_predict) 
    ##On ramène finalement les valeurs à 0 à 10
    if (premium_predicted_vanilla<0):    
        final_premium=[10]    
    return f'{final_premium[0]:.2f}'

def profit_and_loss(x,m,n,i,a,degree):
    if (m>=n):
        return('error')
    actuarial=TermInsuranceAnnual(x,m,n,i,a)
    machine_learning=term_insurance_predicted(x,m,n,i,a,degree)
    print('premium computed with actuarial method : ' ,actuarial)
    print('premium computed with machine learning method: ' ,machine_learning)
    P_and_L=machine_learning-actuarial
    if (P_and_L>0):
        print('the profit while using Artificial intelligence is',  P_and_L)
        return( P_and_L)
    if (P_and_L<0):
        print('the loss while using Artificial intelligence is',  -P_and_L)
        return( P_and_L)        

def profit_and_loss_age(x,m,n,i,a,degree):
    age=list()
    p_and_l=list()
    for x in range(1,90):
        p_and_l.append(profit_and_loss(x,m,n,i,a,degree))
        age.append(x)
    plt.plot(age,p_and_l)
    plt . xlabel ('age', fontsize =20)
    plt . ylabel ('profit_and_loss', fontsize =20)
    plt . title ('profit and loss as a function of the age',fontsize =16)
    #plt.show()    
    
def profit_and_loss_interest_rate(x,m,n,i,a,degree):
    interest=list()
    p_and_l=list()
    for i in range(1,1000):
        p_and_l.append(profit_and_loss(x,m,n,i/1000,a,degree))
        interest.append(i/1000)
    plt.plot(interest,p_and_l)
    plt . xlabel ('interest_rate', fontsize =20)
    plt . ylabel ('profit_and_loss', fontsize =20)
    plt . title ('profit and loss as a function of the interest rate',fontsize =16)

def profit_and_loss_amount(x,m,n,i,a,degree):
    premium=list()
    p_and_l=list()
    for a in range(1,50000,10):
        p_and_l.append(profit_and_loss(x,m,n,i,a,degree))
        premium.append(a)
    plt.plot(premium,p_and_l)
    plt . xlabel ('insured amounts', fontsize =20)
    plt . ylabel ('profit_and_loss', fontsize =20)
    plt . title ('profit and loss as a function of the insured amount',fontsize =16)

def profit_and_loss_maturity(x,m,n,i,a,degree):
    maturity=list()
    p_and_l=list()
    for n in range(m+1,105-x):
        p_and_l.append(profit_and_loss(x,m,n,i,a,degree))
        maturity.append(n)
    plt.plot(maturity,p_and_l)
    plt . xlabel ('maturity', fontsize =20)
    plt . ylabel ('profit_and_loss', fontsize =20)
    plt . title ('profit and loss as a function of the maturity',fontsize =16)

def profit_and_loss_payements(x,m,n,i,a,degree):
    payements=list()
    p_and_l=list()
    for m in range(1,n):
        p_and_l.append(profit_and_loss(x,m,n,i,a,degree))
        payements.append(m)
    plt.plot(payements,p_and_l)
    plt . xlabel ('number of payements', fontsize =20)
    plt . ylabel ('profit_and_loss', fontsize =20)
    plt . title ('profit and loss as a function of the number of payements',fontsize =16)

def learning_curve_poly_under0_3d(N):
     X_train_new=list()
     r2test=np.zeros((N,N))
          
     listei=list()
     degreei=list()     
     for degree in range(0,N):
         degreei.append(degree+1)         
         for j in range(1,N+1):
             i=len(X_train)//N*j
             X_train_new=X_train[0:i]
             y_train_new=y_train[0:i]

             r2test[degree,j-1]=polyfinal_learning_under0(degree+1,X_train_new,y_train_new)[1]       
             listei.append(i)        
     fig = figure()
     ax = Axes3D(fig)
     ax.set_xlabel('training set size')
     ax.set_ylabel('degree')   
     ax.set_zlabel('R²')       
     X, Y = np.meshgrid( np.arange(len(X_train)//N,len(X_train),len(X_train)//N),degreei)
     ax.plot_surface(X, Y, r2test , rstride=1, cstride=1, cmap='hot')
     show()  
