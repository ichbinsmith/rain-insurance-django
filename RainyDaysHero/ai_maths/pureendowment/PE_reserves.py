# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:28:26 2021

@author: Mon PC
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

import csv
import os
matplotlib.use('Agg')

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/static/RainyDaysHero/data/LI/TI/dataset.csv")

X =df[['age','nb_payements','maturity','interest_rate','amount']]

y = df['target']

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=5)

X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)


TF = [100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,99987.943454803000000,99976.891621705800000,99966.844500708300000,99956.797379710800000,99946.750258713400000,99935.698425616100000,99924.646592518900000,99912.590047321900000,99899.528790025200000,99883.453396429200000,99863.359154434300000,99839.246064040300000,99810.109413147700000,99776.953913856000000,99741.788990364800000,99706.624066873600000,99672.463855482200000,99637.262739119000000,99602.061622755800000,99562.808587152700000,99520.536087272300000,99474.237635022500000,99422.906742310700000,99365.536921044500000,99302.128171224000000,99232.680492849200000,99157.193885920100000,99073.655374251900000,98981.058469752200000,98879.403172420900000,98766.676506073400000,98642.878470709600000,98507.002578237200000,98358.042340563700000,98195.997757689200000,98004.419419985400000,97798.724783713900000,97579.922155915400000,97348.011536589700000,97104.001232777500000,96846.882937438200000,96573.631729450100000,96282.230994732200000,95970.664119203400000,95638.931102863600000,95264.475758390500000,94869.779584486300000,94452.818498208100000,94010.556375141100000,93537.933007927700000,93029.888189210000000,92482.373753101900000,91837.624008223600000,91136.853353442800000,90372.931854771900000,89538.729578223200000,88625.079465812800000,87620.777335560000000,86517.674691479100000,85305.585913587900000,83972.288257907500000,82501.484732466400000,80663.431201173300000,78624.843847641900000,76364.941964904500000,73862.944845993400000,71093.915642547400000,68033.956541553500000,64662.286836044200000,60970.594243232300000,56968.230081253200000,52685.326375210200000,48171.756821825700000,43502.331966183500000,38775.760166380100000,33634.170688786900000,28741.863237795800000,24153.183658068700000,19508.781438646500000,15400.161585067600000,11857.408133916700000,8887.683951047370000,6469.500438899080000,4562.745548692110000,3110.116353598300000,2042.849341423850000,1289.315853633560000,779.319804862954000,449.827975601043000,246.402585361081000,127.499012192652000,63.033219510974200,28.651463414079200,12.893158536335600,5.730292682815840,1.432573170703960]

omega = 110
#omega = 112

#lx - table
lx = TF



def SinglePremiumPE(x,n,i,a,lx):
    return (1/(1+i)**n * (lx[n+x]/lx[x]) )*a

def PEAnnual(x,p,n,i,a,lx):
    AP = 0
    for j in range(p): AP+= SinglePremiumPE(x,j,i,a,lx)
    return (SinglePremiumPE(x,n,i,a,lx) / AP )*a


def Lx(x):
    return lx[x]
    
def LxOffset(x, offset):
    if x == 0:
        return lx[0]
    return max(0,lx[int(x)] - offset)

#dx, ndx, qx
def Dx(x,lx):
    if x+1 == len(lx):
        return lx[x]
    return lx[x]-lx[x+1]

def NDX(x,n,lx):
    return Lx[x] - Lx[int(x+n)]

def Qx(x,lx):
    #lx = []
    #lx = readLxInputFile(lx)
    if lx[x] < 1:
        return 1
    return Dx(x,lx)/lx[x]



def A_x_to_n(x,i,n,lx):
    s=0
    for j in range(0,n):
        s+=lx[x+j]/lx[x]/(1+i)**j
    return(s)    




def StressTest_table(TF,stress):
    tablelx=TF.copy()
    tablelx.append(0)
    new_Qx=list()
    tablelx[0]=100000
    for i in range(0,len(TF)):
        new_Qx.append(Qx(i,TF)*(1-stress))

        tablelx[i+1]=(tablelx[i]*(1-new_Qx[i]))
    return(tablelx,new_Qx)
 



def reserves_sum(stress_MT=0,stress_interest_rates=0,adapt=True):
    #put stress in %
     level_annual_premium=np.zeros((len(X),41))
     natural_premium=np.zeros((len(X),41))
     if adapt==True:
        stress=stress_MT
        stress_i=stress_interest_rates
        newTF=StressTest_table(TF,stress)[0]
        listcontract=np.zeros((len(X),41))
        for contract in range(0,len(X)):
           x=int(X.iloc[contract].age)
           m=int(X.iloc[contract].nb_payements)
           n=int(X.iloc[contract].maturity)
           i=X.iloc[contract].interest_rate+stress_i
           a=X.iloc[contract].amount
           annual_premium=PEAnnual(x,m,n,i,a,newTF)
           for term in range(1,n+1):
               qx=Qx(x+term-1,newTF)               
               down=1/(1+i)*(1-qx)
               if (term<=m):
                   level_annual_premium[contract][term-1]=annual_premium                   
                   left=listcontract[contract][term-1]+annual_premium
               else:
                   left=listcontract[contract][term-1]
               
               if (term==n):
                   right=a*(1/(1+i))*(1-qx)
                   natural_premium[contract][term-1]=a*(1/(1+i))*(1-qx)
                   listcontract[contract][term]=(left-right)/down
               else:
                   listcontract[contract][term]=(left)/down
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
        newTF=StressTest_table(TF,stress)[0]
        listcontract=np.zeros((len(X),41))
        for contract in range(0,len(X)):
           x=int(X.iloc[contract].age)
           m=int(X.iloc[contract].nb_payements)
           n=int(X.iloc[contract].maturity)
           i=X.iloc[contract].interest_rate
           a=X.iloc[contract].amount
           annual_premium=PEAnnual(x,m,n,i,a,TF)
           for term in range(1,n+1):
                   qx=Qx(x+term-1,newTF)               
                   down=1/(1+i+stress_i)*(1-qx)
                   if (term<=m):
                       level_annual_premium[contract][term-1]=annual_premium                   
                       left=listcontract[contract][term-1]+annual_premium
                   else:
                       left=listcontract[contract][term-1]
                   
                   if (term==n):
                       right=a*(1/(1+i+stress_i))*(1-qx)
                       natural_premium[contract][term-1]=a*(1/(1+i+stress_i))*(1-qx)
                       listcontract[contract][term]=(left-right)/down
                   else:
                       listcontract[contract][term]=(left)/down
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
     plt.show()
     return([i for i in range(1,42)],reserve_total,natural_premium_total,level_annual_premium_total)          




#model=stresstest. best_model_stress(0,0,X)[0]
#model.predict([[60,40,40,2.5/100,8000]])
#stresstest.TermInsuranceAnnual(60,40,2.5/100,8000,40,TH) 

def reserves_sum_knn(stress_MT=0,stress_interest_rates=0,adapt=True): 
    level_annual_premium=np.zeros((len(X),41))
    natural_premium=np.zeros((len(X),41))    
    if adapt==True:
        ## First, We compute the best model
        model= best_model_scale_knn(stress_MT,stress_interest_rates,X)
      #  model=stresstest. best_model_scale_knn(stress_MT,stress_interest_rates,X)[0]    
        stress=stress_MT
        stress_i=stress_interest_rates
        newTF=StressTest_table(TF,stress)[0]
        listcontract=np.zeros((len(X),41))
        for contract in range(0,len(X)):
           x=int(X.iloc[contract].age)
           m=int(X.iloc[contract].nb_payements)
           n=int(X.iloc[contract].maturity)
           i=X.iloc[contract].interest_rate+stress_i
           a=X.iloc[contract].amount
           annual_premium= model.predict([[x,m,n,i,a]])
           for term in range(1,n+1):
               qx=Qx(x+term-1,newTF)               
               down=1/(1+i)*(1-qx)
               if (term<=m):
                   level_annual_premium[contract][term-1]=annual_premium                   
                   left=listcontract[contract][term-1]+annual_premium
               else:
                   left=listcontract[contract][term-1]
               
               if (term==n):
                   right=a*(1/(1+i))*(1-qx)
                   natural_premium[contract][term-1]=a*(1/(1+i))*(1-qx)
                   listcontract[contract][term]=(left-right)/down
               else:
                   listcontract[contract][term]=(left)/down
              
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
        stress=stress_MT
        stress_i=stress_interest_rates
        newTF=StressTest_table(TF,stress)[0]
        listcontract=np.zeros((len(X),41))
        for contract in range(0,len(X)):
           x=int(X.iloc[contract].age)
           m=int(X.iloc[contract].nb_payements)
           n=int(X.iloc[contract].maturity)
           i=X.iloc[contract].interest_rate
           a=X.iloc[contract].amount
           annual_premium= model.predict([[x,m,n,i,a]]) 
           for term in range(1,n+1):
                   qx=Qx(x+term-1,newTF)               
                   down=1/(1+i+stress_i)*(1-qx)
                   if (term<=m):
                       level_annual_premium[contract][term-1]=annual_premium                   
                       left=listcontract[contract][term-1]+annual_premium
                   else:
                       left=listcontract[contract][term-1]
                   
                   if (term==n):
                       right=a*(1/(1+i+stress_i))*(1-qx)
                       natural_premium[contract][term-1]=a*(1/(1+i+stress_i))*(1-qx)
                       listcontract[contract][term]=(left-right)/down
                   else:
                       listcontract[contract][term]=(left)/down
        recurrence2=list()  
        level_annual_premium_total=list()
        natural_premium_total=list()         
        for term in range(0,41):       
            recurrence2.append(np.sum(listcontract[:,term]))
            natural_premium_total.append(np.sum(natural_premium[:,term]))
            level_annual_premium_total.append(np.sum( level_annual_premium[:,term]))

    p1,=plt.plot(np.arange(0,41,1),level_annual_premium_total,label='level annual premiums')    
    p3,=plt.plot(np.arange(0,41,1),natural_premium_total,label='natural premium')
    p2,=plt.plot(np.arange(0,41,1),recurrence2,label='KNN reserves')
    plt . xlabel ('Years', fontsize =20)
    plt . ylabel ('Reserves', fontsize =20)
    plt . title ('KNN reserves with stress on mortality table={}'.format(stress*100)+"%"+" and stress on interest rates={}".format(stress_i*100)+"%",fontsize =16)
    plt . legend ( handles =[p1 , p2,p3],fontsize =16)
    plt.show()
    reserve_total=recurrence2

    return([i for i in range(1,42)],reserve_total,natural_premium_total,level_annual_premium_total)
    

       
def best_model_scale_knn(stress_MT=0,stress_interest=0,X=X):
    ##put the stress in %
    y_stressed=list()
    stress=stress_MT
    stress_i=stress_interest
    TF_stressed=StressTest_table(TF,stress)[0]
    X_new=X.copy()
    X_new.interest_rate=X_new.interest_rate+stress_i
    for contract in range(0,len(X)):
        y_stressed.append( PEAnnual(int(X_new.iloc[contract].age),int(X_new.iloc[contract].nb_payements),int(X_new.iloc[contract].maturity), X_new.iloc[contract].interest_rate,X_new.iloc[contract].amount,TF_stressed))

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



## we split the main function to decrease the computing time



def reserves_predicted_scale_knn(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):
    if adapt==True:
        ## First, We compute the best model
        model= best_model_scale_knn(stress_MT,stress_interest_rates,X=X)
        stress=stress_MT
        stress_i=stress_interest_rates
        newTF=StressTest_table(TF,stress)[0]
        i=i+stress_i
        listcontract=np.zeros((1,n+1))
        list_annual_premium1=np.zeros((1,n+1))
        list_natural_premium1=np.zeros((1,n+1))        
        annual_premium= model.predict([[x,m,n,i,a]])
        for term in range(1,n+1):
            qx=Qx(x+term-1,newTF)
            down=1/(1+i)*(1-qx)
            if (term<=m):
                left=listcontract[0][term-1]+annual_premium
                list_annual_premium1[0][term-1]=annual_premium
            else:
                left=listcontract[0][term-1]
            if (term==n):    
                right=a*(1/(1+i))*(1-qx)
                list_natural_premium1[0][term-1]=a*(1/(1+i))*(1-qx)
                listcontract[0][term]=(left-right)/down
            else:
                listcontract[0][term]=(left)/down
        recurrence1=listcontract[0]   
    else:        
        model= best_model_scale_knn(stress_MT=0,stress_interest=0,X=X)       
        list_annual_premium1=np.zeros((1,n+1))
        list_natural_premium1=np.zeros((1,n+1))        
        stress=stress_MT
        stress_i=stress_interest_rates
        newTF=StressTest_table(TF,stress)[0]
        listcontract=np.zeros((1,n+1))
        annual_premium= model.predict([[x,m,n,i,a]])
        for term in range(1,n+1):
            qx=Qx(x+term-1,newTF)
            down=1/(1+i+stress_i)*(1-qx)
            if (term<=m):
                left=listcontract[0][term-1]+annual_premium
                list_annual_premium1[0][term-1]=annual_premium
            else:
                left=listcontract[0][term-1]
            if (term==n):    
                right=a*(1/(1+i+stress_i))*(1-qx)
                list_natural_premium1[0][term-1]=a*(1/(1+i+stress_i))*(1-qx)
                listcontract[0][term]=(left-right)/down
            else:
                listcontract[0][term]=(left)/down
        recurrence1=listcontract[0]   
    p1,=plt.plot(np.arange(0,n+1,1),recurrence1,label='AI model reserves')   
    p5,=plt.plot(np.arange(0,n,1),list_natural_premium1[0][0:n],label='KNN model natural premiums')
    p6,=plt.plot(np.arange(0,n,1),list_annual_premium1[0][0:n],label='KNN model annual premiums')  

    plt . xlabel ('Years', fontsize =20)
    plt . ylabel ('Reserves', fontsize =20)
    plt . title ('Reserves with stress on mortality table={}'.format(stress*100)+"%"+" and stress on interest rates={}".format(stress_i*100)+"%",fontsize =16)
    plt . legend ( handles =[p1,p5,p6],fontsize =16)
    plt.show()          
    return([i for i in range(1,n+2)],recurrence1,list_annual_premium1[0],list_natural_premium1[0])    
        


def reserves_true(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):
    if adapt==True:
        stress=stress_MT
        stress_i=stress_interest_rates
        i=i+stress_i
        newTF=StressTest_table(TF,stress)[0]
        listcontract=np.zeros((1,n+1))
        list_annual_premium2=np.zeros((1,n+1))
        list_natural_premium2=np.zeros((1,n+1))           
        annual_premium=PEAnnual(x,m,n,i,a,newTF)
        for term in range(1,n+1):
            qx=Qx(x+term-1,newTF)
            down=1/(1+i)*(1-qx)
            if (term<=m):
                left=listcontract[0][term-1]+annual_premium
                list_annual_premium2[0][term-1]=annual_premium
            else:
                left=listcontract[0][term-1]
            if (term==n):    
                right=a*(1/(1+i))*(1-qx)
                list_natural_premium2[0][term-1]=a*(1/(1+i))*(1-qx)
                listcontract[0][term]=(left-right)/down
            else:
                listcontract[0][term]=(left)/down
        recurrence2=listcontract[0]

    else:
        stress=stress_MT
        stress_i=stress_interest_rates
        newTF=StressTest_table(TF,stress)[0] 
        listcontract=np.zeros((1,n+1))
        annual_premium=PEAnnual(x,m,n,i,a,TF)
        list_annual_premium2=np.zeros((1,n+1))
        list_natural_premium2=np.zeros((1,n+1))         
        for term in range(1,n+1):
            qx=Qx(x+term-1,newTF)
            down=1/(1+i+stress_i)*(1-qx)
            if (term<=m):
                left=listcontract[0][term-1]+annual_premium
                list_annual_premium2[0][term-1]=annual_premium
            else:
                left=listcontract[0][term-1]
            if (term==n):    
                right=a*(1/(1+i+stress_i))*(1-qx)
                list_natural_premium2[0][term-1]=a*(1/(1+i+stress_i))*(1-qx)
                listcontract[0][term]=(left-right)/down
            else:
                listcontract[0][term]=(left)/down
        recurrence2=listcontract[0]
        
    p2,=plt.plot([i for i in range(1,n+2)],recurrence2,label='real reserves')
    p3,=plt.plot(np.arange(1,n+1,1),list_natural_premium2[0][0:n],label='real natural premiums')
    p4,=plt.plot(np.arange(1,n+1,1),list_annual_premium2[0][0:n],label='real annual premiums')    
    plt . xlabel ('Years', fontsize =20)
    plt . ylabel ('Reserves', fontsize =20)
    plt . title ('Reserves with stress on mortality table={}'.format(stress*100)+"%"+" and stress on interest rates={}".format(stress_i*100)+"%",fontsize =16)
    plt . legend ( handles =[p2,p3,p4],fontsize =16)
    plt.show()          
    return([i for i in range(1,n+2)],recurrence2,list_annual_premium2[0],list_natural_premium2[0])    
 

























