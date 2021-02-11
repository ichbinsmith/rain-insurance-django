

import pandas as pd
import numpy as np
import mglearn as mg
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

import matplotlib
matplotlib.use('Agg')
import csv
import os

from sklearn.preprocessing import MinMaxScaler

#lx - table
lx = TF

df = pd.read_csv(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/static/RainyDaysHero/data/LI/PE/dataset.csv")

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
 


def PE_evolution(TH,x,i,n,m,a):
    TI=list()
    listi=list()
    for j in range(1,100):
        tablelx=StressTest_table(TH,j/100)[0]
        TI.append(PEAnnual(x,m,n,i,a,tablelx))
        listi.append(j)
    plt.plot(listi,TI)
    # plt . xlabel ('discount (%)', fontsize =20)
    # plt . ylabel ('Annual premium', fontsize =20)
    # plt . title ('Evolution of the Annual premium of a Pure endowment with respect to the discount',fontsize =16)
     return(TI)    


##############################################

#Here begins the functions we will use to plot#

##############################################


  
def plot_p_and_l_point_interest(TF,x,i,n,m,a):
    P_and_L=list()
    TI=PEAnnual(x,m,n,i,a,TF)
    stresslist=list()
    for stress in range(-15,25):
        stress=stress/1000
        stresslist.append(stress)        
        P_and_L.append(TI-PEAnnual(x,m,n,i+stress,a,TF))
    # plt.xlabel('Interest rate increase', fontsize=20)    
    # plt . ylabel ('Profit and loss', fontsize =20)
    # plt . title ('P & L as a function of the interest rate stress ',fontsize =16)
    # p1,=plt.plot(stresslist,P_and_L,label='Profit_and_loss',color='red')
    # plt . legend ( handles =[p1],fontsize =16)

    # plt.plot()         
    return stresslist, P_and_L
    


def plot_p_and_l_point(TF,x,i,n,m,a):
    P_and_L=list()
    TI=PEAnnual(x,m,n,i,a,TF)
    stresslist=list()
    for stress in range(-10,10):
        stress=stress/100
        stresslist.append(stress)        
        TF_stressed=StressTest_table(TF,stress)[0]        
        P_and_L.append(TI-PEAnnual(x,m,n,i,a,TF_stressed))
    # plt.xlabel('Stress ', fontsize=20)    
    # plt . ylabel ('Profit and loss', fontsize =20)
    # plt . title ('P & L with stress on mortality table',fontsize =16)
    # p1,=plt.plot(stresslist,P_and_L,label='Profit_and_loss',color='red')
    # plt . legend ( handles =[p1],fontsize =16)
    # plt.plot()     
    return stresslist, P_and_L

def plot_p_and_l_point_knn(TF,x,i,n,m,a,degree=8):
    P_and_L=list()
    TI=Pure_endowment_predicted(x,m,n,i,a,degree)
    stresslist=list()
    for stress in range(-10,10):
        stress=stress/100
        stresslist.append(stress)        
        TF_stressed=StressTest_table(TF,stress)[0]        
        P_and_L.append(TI-PEAnnual(x,m,n,i,a,TF_stressed))
    # plt.xlabel('Stress ', fontsize=20)    
    # plt . ylabel ('Profit and loss', fontsize =20)
    # plt . title ('P & L as a function of the stress ',fontsize =16)
    # p1,=plt.plot(stresslist,P_and_L,label='Profit_and_loss',color='red')
    # plt . legend ( handles =[p1],fontsize =16)

    # plt.plot()        
    return stresslist, P_and_L

def plot_p_and_l_point_interest_knn(TF,x,i,n,m,a,degree=8):
    P_and_L=list()
    TI=Pure_endowment_predicted(x,m,n,i,a,degree)
    stresslist=list()
    for stress in range(-15,25):
        stress=stress/1000
        stresslist.append(stress)        
        P_and_L.append(TI-PEAnnual(x,m,n,i+stress,a,TF))
    # plt.xlabel('Interest rate increase', fontsize=20)    
    # plt . ylabel ('Profit and loss', fontsize =20)
    # plt . title ('P & L as a function of the interest rate stress ',fontsize =16)
    # p1,=plt.plot(stresslist,P_and_L,label='Profit_and_loss',color='red')
    # plt . legend ( handles =[p1],fontsize =16)
    # plt.plot() 
    return stresslist, P_and_L


def plot_p_and_l_point_new(x,m,n,i,a,stress_MT=True):
    P_and_L=list()
    stresslist=list()  
    if (stress_MT==True): 
        for stress in range(-10,10):
            stress=stress/100
            new_knn=new_p_and_l_point(x,m,n,i,a,stress,stress_interest=0)
            
            stresslist.append(stress)        
            P_and_L.append(new_knn)
           
            # plt.xlabel('Stress ', fontsize=20)    
            # plt . ylabel ('Profit and loss', fontsize =20)
            # plt . title ('P & L as a function of the stress ',fontsize =16)
            # p1,=plt.plot(stresslist,P_and_L,label='Profit_and_loss',color='red')
            # plt . legend ( handles =[p1],fontsize =16)
            # plt.plot()
    else:
        for stress in range(-15,25):
            stress=stress/1000
            new_knn=new_p_and_l_point(x,m,n,i,a,0,stress)
            
            stresslist.append(stress)        
            P_and_L.append(new_knn)            
    
            # plt.xlabel('Interest rate increase', fontsize=20)    
            # plt . ylabel ('Profit and loss', fontsize =20)
            # plt . title ('P & L as a function of the interest rate stress ',fontsize =16)
            # p1,=plt.plot(stresslist,P_and_L,label='Profit_and_loss',color='red')
            # plt . legend ( handles =[p1],fontsize =16)
            # plt.plot()        
   
    return stresslist,P_and_L




################################################

#After  this point we won't plot the functions #

################################################




    
def new_p_and_l_point(x,m,n,i,a,stress=0,stress_interest=0):
    rec= best_model_scale_knn(stress,stress_interest,X=X)
    stress=stress
    stress_i=stress_interest
    premium_to_predict=[[x,m,n,i+stress_i,a]] 
    prediction = rec[0].predict(premium_to_predict)
    stress=stress
    stress_i=stress_interest
    TF_stressed=StressTest_table(TF,stress)[0]        
    P_and_L= prediction-PEAnnual(x,m,n,i+stress_i,a,TF_stressed)
    return P_and_L[0]               

def best_model_scale_knn(stress_MT=0,stress_interest=0,X=X):
    ##put the stress in %
    y_stressed=list()
    stress=stress_MT
    stress_i=stress_interest
    TF_stressed=StressTest_table(TF,stress)[0]
    X_new=X.copy()
    X_new.interest_rate=X_new.interest_rate+stress_i
    for contract in range(0,len(X)):
        y_stressed.append(PEAnnual(int(X_new.iloc[contract].age),int(X_new.iloc[contract].nb_payements),int(X_new.iloc[contract].maturity), X_new.iloc[contract].interest_rate,X_new.iloc[contract].amount,TF_stressed))   

    X_trainval, X_test, y_trainval, y_test = train_test_split(X_new, y_stressed, random_state=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)
    test_scores_scaled=list()
    train_scores_scaled=list()
    valid_scores_scaled=list()    
    nn_list_scaled=list()
    valid_scores=list()
    test_scores=list()
    train_scores=list()    
    
    for nn in range (7,9):
            
        nn_list_scaled.append(nn)            
        scaler = MinMaxScaler()
        X_train_scaled  = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)    
        X_valid_scaled = scaler.transform(X_valid)            
        poly_scaled= PolynomialFeatures(degree=nn,include_bias=False)
        poly_scaled.fit(X_train_scaled)
        X_poly_train_scaled  = poly_scaled.transform(X_train_scaled)
        X_poly_test_scaled = poly_scaled.transform(X_test_scaled)
        X_poly_valid_scaled = poly_scaled.transform(X_valid_scaled)        
        model= LinearRegression().fit(X_poly_train_scaled, y_train)
        test_scores_scaled.append(model.score(X_poly_test_scaled,y_test))
        train_scores_scaled.append(model.score(X_poly_train_scaled,y_train))  
        valid_scores_scaled.append(model.score(X_poly_valid_scaled,y_valid))
        X_test_scaled = scaler.transform(X_new)    
        X_poly_test_scaled = poly_scaled.transform(X_test_scaled)
        y_poly = model.predict(X_poly_test_scaled)
        y_poly=np.abs(y_poly)        
        model=KNeighborsRegressor(n_neighbors=20,weights="distance")
        model.fit(X_new, y_poly)     
        train_scores.append(model.score(X_train,y_train))
        valid_scores.append(model.score(X_valid,y_valid))
        test_scores.append(model.score(X_test,y_test))
                      
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
    
    return model, nn_list_scaled, train_scores, valid_scores, test_scores,y_stressed  ,test_scores_scaled  

    
def plot_p_and_l_knn_sum_stress():
    p_and_l=list()
    stresslist=list()
    knn_sum=sum(best_model_scale_knn(stress_MT=0,stress_interest=0,X=X)[0].predict(X))
    for stress in range(-10,10):
        TI_stressed=list()
        stress=stress/100
        TF_stressed=StressTest_table(TF,stress)[0]
        stresslist.append(stress)
        for contract in range(0,len(X)):
            TI_stressed.append(PEAnnual(int(X.iloc[contract].age),int(X.iloc[contract].nb_payements),int(X.iloc[contract].maturity), X.iloc[contract].interest_rate,X.iloc[contract].amount,TF_stressed))
        p_and_l.append(knn_sum-sum(TI_stressed))

    # plt.xlabel('Stress ', fontsize=20)    
    # plt . ylabel ('Profit and loss', fontsize =20)
    # plt . title ('Profit and loss with stress on the mortality table ',fontsize =16)
    # p1,=plt.plot(stresslist, p_and_l,label='Profit_and_loss',color='red')
    # plt . legend ( handles =[p1],fontsize =16)

    # plt.plot()
    return stresslist, p_and_l

def plot_p_and_l_sum_interest_knn():
    p_and_l=list()
    stresslist=list()
    knn_sum=sum(best_model_scale_knn(stress_MT=0,stress_interest=0,X=X)[0].predict(X))    
    for stress in range(-15,25):
        TI_stressed=list()
        stress=stress/1000
        stresslist.append(stress)
        for contract in range(0,len(X)):
            TI_stressed.append(PEAnnual(int(X.iloc[contract].age),int(X.iloc[contract].nb_payements),int(X.iloc[contract].maturity), X.iloc[contract].interest_rate+stress,X.iloc[contract].amount,TF))
        p_and_l.append(knn_sum-sum(TI_stressed))

    # plt.xlabel('Interest rate increase', fontsize=20)    
    # plt . ylabel ('Profit and loss', fontsize =20)
    # plt . title ('Profit and loss with stress on the interest rates',fontsize =16)
    # p1,=plt.plot(stresslist, p_and_l,label='Profit_and_loss',color='red')
    # plt . legend ( handles =[p1],fontsize =16)

    # plt.plot() 
    return stresslist, p_and_l

         
    

        
def plot_p_and_l_knn_sum_stress_new():
    p_and_l=list()
    stresslist=list()
    for stress in range(-10,10):
        p_and_l.append(new_p_and_l_sum(stress=stress/100,stress_interest=0))
        stresslist.append(stress/100)
    # plt.xlabel('Stress ', fontsize=20)    
    # plt . ylabel ('Profit and loss', fontsize =20)
    # plt . title ('Profit and loss for the AI method with stress on the mortality table ',fontsize =16)
    # p1,=plt.plot(stresslist, p_and_l,label='Profit_and_loss',color='red')
    # plt . legend ( handles =[p1],fontsize =16)

    # plt.plot()        
    return stresslist, p_and_l
        
def plot_p_and_l_knn_sum_stress_interest_new():
    p_and_l=list()
    stresslist=list()
    for stress in range(-15,25):
        p_and_l.append(new_p_and_l_sum(stress=0,stress_interest=stress/1000))
        stresslist.append(stress/1000)
    # plt.xlabel('Stress ', fontsize=20)    
    # plt . ylabel ('Profit and loss', fontsize =20)
    # plt . title ('Profit and loss for the AI method with stress on the interest rates ',fontsize =16)
    # p1,=plt.plot(stresslist, p_and_l,label='Profit_and_loss',color='red')
    # plt . legend ( handles =[p1],fontsize =16)

    # plt.plot()         
    return stresslist, p_and_l


def new_p_and_l_sum(stress=0,stress_interest=0):
    rec= best_model_scale_knn(stress,stress_interest,X=X)
    X_new=X.copy()
    X_new.interest_rate=X_new.interest_rate+stress_interest
    return sum((rec[0]).predict(X_new))-sum(rec[5])


def plot_p_and_l_sum():
    p_and_l=list()
    stresslist=list()
    for stress in range(-10,10):
        TI=list()
        TI_stressed=list()
        stress=stress/100
        TF_stressed=StressTest_table(TF,stress)[0]
        stresslist.append(stress)
        for contract in range(0,len(X)):
            TI.append(PEAnnual(int(X.iloc[contract].age),int(X.iloc[contract].nb_payements),int(X.iloc[contract].maturity), X.iloc[contract].interest_rate,X.iloc[contract].amount,TF))
            TI_stressed.append(PEAnnual(int(X.iloc[contract].age),int(X.iloc[contract].nb_payements),int(X.iloc[contract].maturity), X.iloc[contract].interest_rate,X.iloc[contract].amount,TF_stressed))
        p_and_l.append(sum(TI)-sum(TI_stressed))

    # plt.xlabel('Stress ', fontsize=20)    
    # plt . ylabel ('Profit and loss', fontsize =20)
    # plt . title ('Profit and loss with a stress on the mortality table ',fontsize =16)
    # p1,=plt.plot(stresslist, p_and_l,label='Profit_and_loss',color='red')
    # plt . legend ( handles =[p1],fontsize =16)

    # plt.plot()
    return stresslist, p_and_l



def plot_p_and_l_sum_interest():
    p_and_l=list()
    stresslist=list()
    for stress in range(-15,25):
        TI=list()
        TI_stressed=list()
        stress=stress/1000
        stresslist.append(stress)
        for contract in range(0,len(X)):
            TI.append(PEAnnual(int(X.iloc[contract].age),int(X.iloc[contract].nb_payements),int(X.iloc[contract].maturity), X.iloc[contract].interest_rate,X.iloc[contract].amount,TF))
            TI_stressed.append(PEAnnual(int(X.iloc[contract].age),int(X.iloc[contract].nb_payements),int(X.iloc[contract].maturity), X.iloc[contract].interest_rate+stress,X.iloc[contract].amount,TF))
        p_and_l.append(sum(TI)-sum(TI_stressed))

    # plt.xlabel('Interest rate increase', fontsize=20)    
    # plt . ylabel ('Profit and loss', fontsize =20)
    # plt . title ('Profit and loss with stress on the interest rates ',fontsize =16)
    # p1,=plt.plot(stresslist, p_and_l,label='Profit_and_loss',color='red')
    # plt . legend ( handles =[p1],fontsize =16)

    # plt.plot() 
    return stresslist, p_and_l


def Pure_endowment_predicted(x,m,n,i,a,degree=8):
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

    return np.abs(y_test_predict[0])

def profit_and_loss(x,m,n,i,a):
    actuarial=PEAnnual(x,m,n,i,a,lx)
    machine_learning=best_model_scale_knn(stress_MT=0,stress_interest=0,X=X)[0].predict([(x,m,n,i,a)])
    # print('premium computed with actuarial method : ' ,actuarial)
    # print('premium computed with machine learning method: ' ,machine_learning)
    P_and_L=machine_learning-actuarial
    return( P_and_L)        


def profit_and_loss_age():
    ML=best_model_scale_knn(stress_MT=0,stress_interest=0,X=X)[0].predict(X)
    p_and_l=ML-y        
    age=[20,30,40,50,60]
    pl=np.zeros((1,5))
    
    for smash in range(0,len(X.age)):
        for smashos in range(0,len(age)):
            if X.age[smash]==age[smashos]:
                pl[0][smashos]+=p_and_l[smash]                

    # plt . xlabel ('age', fontsize =20)
    # plt . ylabel ('profit_and_loss', fontsize =20)
    # plt . title ('profit and loss as a function of the age',fontsize =16)
    # plt.bar(age, pl[0], width=1.0, color='b' )
    # plt.show()    
    
def profit_and_loss_interest():
    ML=best_model_scale_knn(stress_MT=0,stress_interest=0,X=X)[0].predict(X)
    p_and_l=ML-y        
    age=[0,0.5,1,1.5,2,2.5]
    pl=np.zeros((1,6))
    
    for smash in range(0,len(X.interest_rate)):
        for smashos in range(0,len(age)):
            if X.interest_rate[smash]==age[smashos]/100:
                pl[0][smashos]+=p_and_l[smash]                

    # plt . xlabel ('interest rate (%)', fontsize =20)
    # plt . ylabel ('profit_and_loss', fontsize =20)
    # plt . title ('profit and loss as a function of the interest rate',fontsize =16)
    # plt.bar(age, pl[0], width=0.1, color='b' )
    # plt.show()  

def profit_and_loss_payments():
    ML=best_model_scale_knn(stress_MT=0,stress_interest=0,X=X)[0].predict(X)
    p_and_l=ML-y        
    age=[1,5,10,20,30,40]
    pl=np.zeros((1,6))
    
    for smash in range(0,len(X.nb_payements)):
        for smashos in range(0,len(age)):
            if X.nb_payements[smash]==age[smashos]:
                pl[0][smashos]+=p_and_l[smash]                

    # plt . xlabel ('number of payments', fontsize =20)
    # plt . ylabel ('profit_and_loss', fontsize =20)
    # plt . title ('profit and loss as a function of the number of payments',fontsize =16)
    # plt.bar(age, pl[0], width=1, color='b' )
    # plt.show() 

def profit_and_loss_maturity():
    ML=best_model_scale_knn(stress_MT=0,stress_interest=0,X=X)[0].predict(X)
    p_and_l=ML-y        
    age=[1,5,10,20,30,40]
    pl=np.zeros((1,6))
    
    for smash in range(0,len(X.maturity)):
        for smashos in range(0,len(age)):
            if X.maturity[smash]==age[smashos]:
                pl[0][smashos]+=p_and_l[smash]                

    # plt . xlabel ('Maturity', fontsize =20)
    # plt . ylabel ('profit_and_loss', fontsize =20)
    # plt . title ('profit and loss as a function of the maturity',fontsize =16)
    # plt.bar(age, pl[0], width=1, color='b' )
    # plt.show() 

def profit_and_loss_amount():
    ML=best_model_scale_knn(stress_MT=0,stress_interest=0,X=X)[0].predict(X)
    p_and_l=ML-y        
    age=[500,1000,2000,4000,8000]
    pl=np.zeros((1,5))
    for smash in range(0,len(X.amount)):
        for smashos in range(0,len(age)):
            if X.amount[smash]==age[smashos]:
                pl[0][smashos]+=p_and_l[smash]                

    # plt . xlabel ('Amount', fontsize =20)
    # plt . ylabel ('profit_and_loss', fontsize =20)
    # plt . title ('profit and loss as a function of the insured amount',fontsize =16)
    # plt.bar(age, pl[0], width=500, color='b' )
    # plt.show() 





    