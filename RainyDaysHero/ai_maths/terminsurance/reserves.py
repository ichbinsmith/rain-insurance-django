from RainyDaysHero.ai_maths.terminsurance import KNN,stresstest

TH = [100000,99511,99473,99446,99424,99406,99390,99376,99363,99350,99338,99325,99312,99296,99276,99250,99213,99163,99097,99015,98921,98820,98716,98612,98509,98406,98303,98198,98091,97982,97870,97756,97639,97517,97388,97249,97100,96939,96765,96576,96369,96141,95887,95606,95295,94952,94575,94164,93720,93244,92736,92196,91621,91009,90358,89665,88929,88151,87329,86460,85538,84558,83514,82399,81206,79926,78552,77078,75501,73816,72019,70105,68070,65914,63637,61239,58718,56072,53303,50411,47390,44234,40946,37546,34072,30575,27104,23707,20435,17338,14464,11852,9526,7498,5769,4331,3166,2249,1549,1032,663,410,244,139,75,39,19,9,4,2,1]

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

#lx - table
lx = TH

df = pd.read_csv(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/static/RainyDaysHero/data/LI/TI/dataset.csv")

X =df[['age','nb_payements','maturity','interest_rate','amount']]

y = df['target']

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=5)

X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)


#Term Insurance Single Premium
def TermInsurance(x,n,i,a):
    NA = 0
    for j in range(1,n+1): NA+=MNQX(x,1,j-1) * TechDF(j,i)
    return NA * a


#Annuity : from 0 - to M-1 --> M values
def AnnuityFromZeroToM(x,i,m):
    A=0
    for j in range(m): A+= NPX(x,j)*TechDF(j,i)
    return A

#Term Insurance Annual Premium 
def TermInsuranceAnnual(x,n,i,a,m):
    NA = 0
    for j in range(1,n+1):
        NA+=MNQX(x,1,j-1) * TechDF(j,i)
    return (NA / AnnuityFromZeroToM(x,i,m) )* a

'''utils'''
def Lx(x):
    return lx[x]
    
def LxOffset(x, offset):
    if x == 0:
        return lx[0]
    return max(0,lx[int(x)] - offset)

#dx, ndx, qx
def Dx(x):
    if x+1 == len(lx):
        return lx[x]
    return lx[x]-lx[x+1]

def NDX(x,n):
    return Lx(x) - Lx(int(x+n))

def Qx(x):
    #lx = []
    #lx = readLxInputFile(lx)
    if Lx(x) == 0:
        return 1
    return Dx(x)/Lx(x)

#Ex
def Ex(x):
    if x+1 == len(lx):
        return 0
    return sum(lx[x+1:])/lx[x]
#npx 
def NPX(x,n):
    return Lx(int(x+n))/Lx(x)

#nqx 
def NQX(x,n):
    return (Lx(x) - Lx(x+n) ) / Lx(x)


#mnqx 
def MNQX(x,n,m):
    return NPX(x,m) * NQX(x+m,n)


#techDF - actualization factor
def TechDF(n,i):
    return 1 / ((1+i)**n)

        
def reserves_predicted(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):
    if adapt==True:
        ## First, We compute the best model
        model= best_model_scale_knn(stress_MT,stress_interest_rates,X=X)
        stress=stress_MT
        stress_i=stress_interest_rates
        newTH=stresstest.StressTest_table(TH,stress)[0]
        i=i+stress_i
        listcontract=np.zeros((1,n+1))
        list_annual_premium1=np.zeros((1,n+1))
        list_natural_premium1=np.zeros((1,n+1))        
        annual_premium= model.predict([[x,m,n,i,a]])
        for term in range(1,n+1):
            qx=stresstest.Qx(x+term-1,newTH)
            down=1
            if (term<=m):
                left=listcontract[0][term-1]+annual_premium
                list_annual_premium1[0][term-1]=annual_premium
            else:
                left=listcontract[0][term-1]
            right=a*qx
            listcontract[0][term]=(left-right)/down
            list_natural_premium1[0][term-1]=a*qx
        recurrence1=listcontract[0]   
    else:        
        model= best_model_scale_knn(stress_MT=0,stress_interest=0,X=X)       
        list_annual_premium1=np.zeros((1,n+1))
        list_natural_premium1=np.zeros((1,n+1))        
        stress=stress_MT
        stress_i=stress_interest_rates
        newTH=stresstest.StressTest_table(TH,stress)[0]
        listcontract=np.zeros((1,n+1))
        annual_premium= model.predict([[x,m,n,i,a]])
        for term in range(1,n+1):
            qx=stresstest.Qx(x+term-1,newTH)
            down=1
            if (term<=m):
                left=listcontract[0][term-1]+annual_premium
                list_annual_premium1[0][term-1]=annual_premium                
            else:
                left=listcontract[0][term-1]
            right=a*qx
            listcontract[0][term]=(left-right)/down
            list_natural_premium1[0][term]=a*qx            
        recurrence1=listcontract[0]    

    #put stress in %
    if adapt==True:
        stress=stress_MT
        stress_i=stress_interest_rates
        newTH=stresstest.StressTest_table(TH,stress)[0]
        listcontract=np.zeros((1,n+1))
        list_annual_premium2=np.zeros((1,n+1))
        list_natural_premium2=np.zeros((1,n+1))           
        annual_premium=stresstest.TermInsuranceAnnual(x,n,i,a,m,newTH) 
        for term in range(1,n+1):
            qx=stresstest.Qx(x+term-1,newTH)
            down=1/(1+i)*(1-qx)
            if (term<=m):
                left=listcontract[0][term-1]+annual_premium
                list_annual_premium2[0][term-1]=annual_premium
            else:
                left=listcontract[0][term-1]
            right=a*(1/(1+i))*qx
            list_natural_premium2[0][term-1]=a*(1/(1+i))*qx
            listcontract[0][term]=(left-right)/down
        recurrence2=listcontract[0]

    else:
        stress=stress_MT
        stress_i=stress_interest_rates
        listcontract=np.zeros((1,n+1))
        annual_premium=stresstest.TermInsuranceAnnual(x,n,i,a,m,TH) 
        list_annual_premium2=np.zeros((1,n+1))
        list_natural_premium2=np.zeros((1,n+1))         
        for term in range(1,n+1):
            qx=stresstest.Qx(x+term-1,newTH)
            down=1/(1+i+stress_i)*(1-qx)
            if (term<=m):
                left=listcontract[0][term-1]+annual_premium
                list_annual_premium2[0][term-1]=annual_premium                
            else:
                left=listcontract[0][term-1]
            right=a*(1/(1+i+stress_i))*qx
            list_natural_premium2[0][term-1]=a*(1/(1+i+stress_i))*qx            
            listcontract[0][term]=(left-right)/down
            
        recurrence2=listcontract[0]     
        
    p1,=plt.plot(np.arange(0,n+1,1),recurrence1,label='KNN model reserves')
    p2,=plt.plot(np.arange(0,n+1,1),recurrence2,label='real reserves')
    p3,=plt.plot(np.arange(0,n,1),list_natural_premium2[0][0:n],label='real natural premiums')
    p4,=plt.plot(np.arange(0,n,1),list_annual_premium2[0][0:n],label='real annual premiums')    
    p5,=plt.plot(np.arange(0,n,1),list_natural_premium1[0][0:n],label='KNN model natural premiums')
    p6,=plt.plot(np.arange(0,n,1),list_annual_premium1[0][0:n],label='KNN annual premiums')  

    plt . xlabel ('Years', fontsize =20)
    plt . ylabel ('Reserves', fontsize =20)
    plt . title ('Reserves with stress on mortality table={}'.format(stress*100)+"%"+" and stress on interest rates={}".format(stress_i*100)+"%",fontsize =16)
    plt . legend ( handles =[p1,p2,p3,p4,p4,p5],fontsize =16)
    #plt.show()          
    return([i for i in range(1,n+2)],recurrence1,recurrence2,list_annual_premium2[0],list_annual_premium1[0],list_natural_premium2[0],list_natural_premium1[0])    
        


  
def reserves_sum(stress_MT=0,stress_interest_rates=0,adapt=False):
    #put stress in %
     level_annual_premium=np.zeros((len(X),41))
     natural_premium=np.zeros((len(X),41))
     if adapt==True:
        stress=stress_MT
        stress_i=stress_interest_rates
        newTH=stresstest.StressTest_table(TH,stress)[0]
        listcontract=np.zeros((len(X),41))
        for contract in range(0,len(X)):
           x=int(X.iloc[contract].age)
           m=int(X.iloc[contract].nb_payements)
           n=int(X.iloc[contract].maturity)
           i=X.iloc[contract].interest_rate+stress_i
           a=X.iloc[contract].amount
           annual_premium=stresstest.TermInsuranceAnnual(x,n,i,a,m,newTH)
           for term in range(1,n+1):
               qx=stresstest.Qx(x+term-1,newTH)               
               natural_premium[contract][term-1]=a*(1/(1+i))*qx
               down=1/(1+i)*(1-qx)
               if (term<=m):
                   level_annual_premium[contract][term-1]=annual_premium                   
                   left=listcontract[contract][term-1]+annual_premium
               else:
                   left=listcontract[contract][term-1]
               right=a*(1/(1+i))*qx
               listcontract[contract][term]=(left-right)/down
        reserve_total=list()      
        level_annual_premium_total=list()
        natural_premium_total=list()
        for term in range(0,41):       
           ## print(np.sum(listcontract[:,40]))
             level_annual_premium_total.append(np.sum(level_annual_premium[:,term]))
             natural_premium_total.append(np.sum(natural_premium[:,term]))
             reserve_total.append(np.sum(listcontract[:,term]))
     else:
        stress=stress_MT
        stress_i=stress_interest_rates
        newTH=stresstest.StressTest_table(TH,stress)[0]
        listcontract=np.zeros((len(X),41))
        for contract in range(0,len(X)):
           x=int(X.iloc[contract].age)
           m=int(X.iloc[contract].nb_payements)
           n=int(X.iloc[contract].maturity)
           i=X.iloc[contract].interest_rate
           a=X.iloc[contract].amount
           annual_premium=stresstest.TermInsuranceAnnual(x,n,i,a,m,TH) 
           for term in range(1,n+1):
               qx=stresstest.Qx(x+term-1,newTH)
               down=1/(1+i+stress_i)*(1-qx)
               natural_premium[contract][term-1]=a*(1/(1+i+stress_i))*qx   
               if (term<=m):
                   level_annual_premium[contract][term-1]=annual_premium                   
                   left=listcontract[contract][term-1]+annual_premium
               else:
                   left=listcontract[contract][term-1]
               right=a*(1/(1+i+stress_i))*qx
               listcontract[contract][term]=(left-right)/down
        reserve_total=list()   
        level_annual_premium_total=list()
        natural_premium_total=list()        
        for term in range(0,41):       
             level_annual_premium_total.append(np.sum(level_annual_premium[:,term]))
             natural_premium_total.append(np.sum(natural_premium[:,term]))
             reserve_total.append(np.sum(listcontract[:,term]))
     p1,=plt.plot(np.arange(0,41,1),level_annual_premium_total,label='level annual premiums')    
     p2,=plt.plot(np.arange(0,41,1),reserve_total,label='real reserves')
     p3,=plt.plot(np.arange(0,41,1),natural_premium_total,label='natural premium')
     plt . xlabel ('Years', fontsize =20)
     plt . ylabel ('Reserves', fontsize =20)
     plt . title ('Reserves with stress on mortality table={}'.format(stress*100)+"%"+" and stress on interest rates={}".format(stress_i*100)+"%",fontsize =16)
     plt . legend ( handles =[p1 , p2,p3],fontsize =16)
     #plt.show()
     return ([i for i in range(1,42)],reserve_total,natural_premium_total,level_annual_premium_total)          


def reserves_sum_knn(stress_MT=0,stress_interest_rates=0,adapt=True): 
    level_annual_premium=np.zeros((len(X),41))
    natural_premium=np.zeros((len(X),41))    
    if adapt==True:
        ## First, We compute the best model
        model= best_model_scale_knn(stress_MT,stress_interest_rates,X)
      #  model=stresstest. best_model_scale_knn(stress_MT,stress_interest_rates,X)[0]    
        stress=stress_MT#/100
        stress_i=stress_interest_rates#/100
        newTH=stresstest.StressTest_table(TH,stress)[0]
        listcontract=np.zeros((len(X),41))
        for contract in range(0,len(X)):
           x=int(X.iloc[contract].age)
           m=int(X.iloc[contract].nb_payements)
           n=int(X.iloc[contract].maturity)
           i=X.iloc[contract].interest_rate+stress_i
           a=X.iloc[contract].amount
           annual_premium= model.predict([[x,m,n,i,a]])
           for term in range(1,n+1):
               qx=stresstest.Qx(x+term-1,newTH)
               down=1/(1+i)*(1-qx)
               if (term<=m):
                   left=listcontract[contract][term-1]+annual_premium
                   level_annual_premium[contract][term-1]=annual_premium                               
               else:
                   left=listcontract[contract][term-1]
               right=a*(1/(1+i))*qx    
               natural_premium[contract][term-1]=a*qx*(1/(1+i))               
               listcontract[contract][term]=(left-right)/down
               
        recurrence2=list()  
        level_annual_premium_total=list()
        natural_premium_total=list()         
        for term in range(0,41):       
           ## print(np.sum(listcontract[:,40]))
            recurrence2.append(np.sum(listcontract[:,term]))
            natural_premium_total.append(np.sum(natural_premium[:,term]))
            level_annual_premium_total.append(np.sum( level_annual_premium[:,term]))

    else:
        model= best_model_scale_knn(0,0,X)        
     #   model=stresstest. best_model_stress(0,0,X)[0]        
        stress=stress_MT#/100
        stress_i=stress_interest_rates#/100
        newTH=stresstest.StressTest_table(TH,stress)[0]
        listcontract=np.zeros((len(X),41))
        for contract in range(0,len(X)):
           x=int(X.iloc[contract].age)
           m=int(X.iloc[contract].nb_payements)
           n=int(X.iloc[contract].maturity)
           i=X.iloc[contract].interest_rate
           a=X.iloc[contract].amount
           annual_premium= model.predict([[x,m,n,i,a]]) 
           for term in range(1,n+1):
               qx=stresstest.Qx(x+term-1,newTH)
               down=1/(1+i+stress_i)*(1-qx)
               if (term<=m):
                   left=listcontract[contract][term-1]+annual_premium
                   level_annual_premium[contract][term-1]=annual_premium                                                     
               else:
                   left=listcontract[contract][term-1]
                   
               right=a*qx*(1/(1+i+stress_i))
               natural_premium[contract][term-1]=a*qx*(1/(1+i+stress_i))                    
               listcontract[contract][term]=(left-right)/down
        recurrence2=list()  
        level_annual_premium_total=list()
        natural_premium_total=list()         
        for term in range(0,41):       
            recurrence2.append(np.sum(listcontract[:,term]))
            natural_premium_total.append(np.sum(natural_premium[:,term]))
            level_annual_premium_total.append(np.sum( level_annual_premium[:,term]))

    # p1,=plt.plot(np.arange(0,41,1),level_annual_premium_total,label='level annual premiums')    
    # p3,=plt.plot(np.arange(0,41,1),natural_premium_total,label='natural premium')
    # p2,=plt.plot(np.arange(0,41,1),recurrence2,label='KNN reserves')
    # plt . xlabel ('Years', fontsize =20)
    # plt . ylabel ('Reserves', fontsize =20)
    # plt . title ('KNN reserves with stress on mortality table={}'.format(stress*100)+"%"+" and stress on interest rates={}".format(stress_i*100)+"%",fontsize =16)
    # plt . legend ( handles =[p1 , p2,p3],fontsize =16)
    # plt.show()
    reserve_total=recurrence2

    return([i for i in range(1,42)], reserve_total,natural_premium_total,level_annual_premium_total)
    
       

#################
 ## Warning % ## 
 ###############

def best_model_scale_knn(stress_MT=0,stress_interest=0,X=X):
    ##put the stress in %
    y_stressed=list()
    stress=stress_MT#/100
    stress_i=stress_interest#/100
    TH_stressed=stresstest.StressTest_table(TH,stress)[0]
    X_new=X.copy()
    X_new.interest_rate=X_new.interest_rate+stress_i
    for contract in range(0,len(X)):
        y_stressed.append(stresstest.TermInsuranceAnnual(int(X_new.iloc[contract].age),int(X_new.iloc[contract].maturity),X_new.iloc[contract].interest_rate,X_new.iloc[contract].amount,int(X_new.iloc[contract].nb_payements),TH_stressed))    

    X_trainval, X_test, y_trainval, y_test = train_test_split(X_new, y_stressed, random_state=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)
    test_scores_scaled=list()
    train_scores_scaled=list()
    nn_list_scaled=list()
    for nn in range (7,9):
            
        nn_list_scaled.append(nn)            
        scaler = MinMaxScaler()
        X_train_scaled  = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)    
        poly_scaled= PolynomialFeatures(degree=nn,include_bias=False)
        poly_scaled.fit(X_train_scaled)
        X_poly_train_scaled  = poly_scaled.transform(X_train_scaled)
        X_poly_test_scaled = poly_scaled.transform(X_test_scaled)
        model= LinearRegression().fit(X_poly_train_scaled, y_train)
        test_scores_scaled.append(model.score(X_poly_test_scaled,y_test))
        train_scores_scaled.append(model.score(X_poly_train_scaled,y_train))  
          
    m = max(test_scores_scaled)
    max_nn_scaled=0
    for nn in  range (7,9):
        if test_scores_scaled[nn-7]==m:
            max_nn_scaled=nn             
    scaler = MinMaxScaler()
    X_train_scaled  = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_new)    
    poly_scaled= PolynomialFeatures(degree=max_nn_scaled,include_bias=False)
    poly_scaled.fit(X_train_scaled)
    X_poly_train_scaled  = poly_scaled.transform(X_train_scaled)
    X_poly_test_scaled = poly_scaled.transform(X_test_scaled)
    model= LinearRegression().fit(X_poly_train_scaled, y_train)
    y_poly = model.predict(X_poly_test_scaled)
    y_poly=np.abs(y_poly)
    ##now we train knn
    
    model=KNeighborsRegressor(n_neighbors=20,weights="distance")
    model.fit(X_new, y_poly)
    return model


# def term_insurance_predicted_polynomiale_scaled(x,m,n,i,a,stress,stress_i,degree=8):
#     if (m>n):
#         return('error')
#     y_stressed=list()
#     TH_stressed=stresstest.StressTest_table(TH,stress)[0]
#     X_new=X.copy()
#     X_new.interest_rate=X_new.interest_rate+stress_i
#     for contract in range(0,len(X)):
#         y_stressed.append(stresstest.TermInsuranceAnnual(int(X_new.iloc[contract].age),int(X_new.iloc[contract].maturity),X_new.iloc[contract].interest_rate,X_new.iloc[contract].amount,int(X_new.iloc[contract].nb_payements),TH_stressed))    

#     X_trainval, X_test, y_trainval, y_test = train_test_split(X_new, y_stressed, random_state=1)

#     X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)
        
#     data=[[x,m,n,i+stress_i,a]] 
#     premium_to_predict=pd.DataFrame(data=data,columns=['age','nb_payements','maturite','taux_interet','montant_garanti'])
    
#     scaler = MinMaxScaler()
#     X_train_scaled  = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(premium_to_predict)
    
    
#     poly_scaled= PolynomialFeatures(degree=degree,include_bias=False)
#     poly_scaled.fit(X_train_scaled)

#     X_poly_train_scaled  = poly_scaled.transform(X_train_scaled)
#     X_poly_test_scaled = poly_scaled.transform(X_test_scaled)

    
#     model= LinearRegression().fit(X_poly_train_scaled, y_train)
#     y_test_predict = model.predict(X_poly_test_scaled)

#     return(np.abs(y_test_predict[0])) 




def reserves_predicted_scale_knn(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):
    if adapt==True:
        ## First, We compute the best model
        model= best_model_scale_knn(stress_MT,stress_interest_rates,X=X)
        stress=stress_MT
        stress_i=stress_interest_rates
        newTH=stresstest.StressTest_table(TH,stress)[0]
        i=i+stress_i
        listcontract=np.zeros((1,n+1))
        list_annual_premium1=np.zeros((1,n+1))
        list_natural_premium1=np.zeros((1,n+1))        
        annual_premium= model.predict([[x,m,n,i,a]])
        for term in range(1,n+1):
            qx=stresstest.Qx(x+term-1,newTH)
            down=1/(1+i)*(1-qx)
            if (term<=m):
                left=listcontract[0][term-1]+annual_premium
                list_annual_premium1[0][term-1]=annual_premium
            else:
                left=listcontract[0][term-1]
            right=a*(1/(1+i))*qx
            listcontract[0][term]=(left-right)/down
            list_natural_premium1[0][term-1]=a*qx
        recurrence1=listcontract[0]   
    else:        
        model= best_model_scale_knn(stress_MT=0,stress_interest=0,X=X)       
        list_annual_premium1=np.zeros((1,n+1))
        list_natural_premium1=np.zeros((1,n+1))        
        stress=stress_MT
        stress_i=stress_interest_rates
        newTH=stresstest.StressTest_table(TH,stress)[0]
        listcontract=np.zeros((1,n+1))
        annual_premium= model.predict([[x,m,n,i,a]])
        for term in range(1,n+1):
            qx=stresstest.Qx(x+term-1,newTH)
            down=1
            if (term<=m):
                left=listcontract[0][term-1]+annual_premium
                list_annual_premium1[0][term-1]=annual_premium                
            else:
                left=listcontract[0][term-1]
            right=a*qx
            listcontract[0][term]=(left-right)/down
            list_natural_premium1[0][term]=a*qx            
        recurrence1=listcontract[0]    

    # p1,=plt.plot(np.arange(0,n+1,1),recurrence1,label='AI model reserves')   
    # p5,=plt.plot(np.arange(0,n,1),list_natural_premium1[0][0:n],label='KNN model natural premiums')
    # p6,=plt.plot(np.arange(0,n,1),list_annual_premium1[0][0:n],label='KNN model annual premiums')  

    # plt . xlabel ('Years', fontsize =20)
    # plt . ylabel ('Reserves', fontsize =20)
    # plt . title ('Reserves with stress on mortality table={}'.format(stress*100)+"%"+" and stress on interest rates={}".format(stress_i*100)+"%",fontsize =16)
    # plt . legend ( handles =[p1,p5,p6],fontsize =16)
    # plt.show()          
    return([i for i in range(1,n+2)],recurrence1,list_annual_premium1[0],list_natural_premium1[0])    
        


def reserves_true(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):
    if adapt==True:
        stress=stress_MT
        stress_i=stress_interest_rates
        i=i+stress_i
        newTH=stresstest.StressTest_table(TH,stress)[0]
        listcontract=np.zeros((1,n+1))
        list_annual_premium2=np.zeros((1,n+1))
        list_natural_premium2=np.zeros((1,n+1))           
        annual_premium=stresstest.TermInsuranceAnnual(x,n,i,a,m,newTH) 
        for term in range(1,n+1):
            qx=stresstest.Qx(x+term-1,newTH)
            down=1/(1+i)*(1-qx)
            if (term<=m):
                left=listcontract[0][term-1]+annual_premium
                list_annual_premium2[0][term-1]=annual_premium
            else:
                left=listcontract[0][term-1]
            right=a*(1/(1+i))*qx
            list_natural_premium2[0][term-1]=a*(1/(1+i))*qx
            listcontract[0][term]=(left-right)/down
        recurrence2=listcontract[0]

    else:
        stress=stress_MT
        stress_i=stress_interest_rates
        newTH=stresstest.StressTest_table(TH,stress)[0]        
        listcontract=np.zeros((1,n+1))
        annual_premium=stresstest.TermInsuranceAnnual(x,n,i,a,m,TH) 
        list_annual_premium2=np.zeros((1,n+1))
        list_natural_premium2=np.zeros((1,n+1))         
        for term in range(1,n+1):
            qx=stresstest.Qx(x+term-1,newTH)
            down=1/(1+i+stress_i)*(1-qx)
            if (term<=m):
                left=listcontract[0][term-1]+annual_premium
                list_annual_premium2[0][term-1]=annual_premium                
            else:
                left=listcontract[0][term-1]
            right=a*(1/(1+i+stress_i))*qx
            list_natural_premium2[0][term-1]=a*(1/(1+i+stress_i))*qx            
            listcontract[0][term]=(left-right)/down
        recurrence2=listcontract[0]            
#    p2,=plt.plot(np.arange(0,n+1,1),recurrence2,label='real reserves')
#    p3,=plt.plot(np.arange(0,n,1),list_natural_premium2[0][0:n],label='real natural premiums')
#    p4,=plt.plot(np.arange(0,n,1),list_annual_premium2[0][0:n],label='real annual premiums')    
#    plt . xlabel ('Years', fontsize =20)
#    plt . ylabel ('Reserves', fontsize =20)
#    plt . title ('Reserves with stress on mortality table={}'.format(stress*100)+"%"+" and stress on interest rates={}".format(stress_i*100)+"%",fontsize =16)
#    plt . legend ( handles =[p2,p3,p4],fontsize =16)
#    plt.show()          
    return([i for i in range(1,n+2)],recurrence2,list_annual_premium2[0],list_natural_premium2[0])    
 





