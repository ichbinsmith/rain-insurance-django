

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


def TermInsurance(x,n,i,a):
    NA = 0
    for j in range(1,n+1): NA+=MNQX(x,1,j-1) * TechDF(j,i)
    return NA * a


#Annuity : from 0 - to M-1 --> M values
def AnnuityFromZeroToM(x,i,m,lx):
    A=0
    for j in range(m):
#        print(j)
        A+= NPX(x,j,lx)*TechDF(j,i)
    return A

#Term Insurance Annual Premium 
def TermInsuranceAnnual(x,n,i,a,m,lx):
    NA = 0
    for j in range(1,n+1):
        NA+=MNQX(x,1,j-1,lx) * TechDF(j,i)
    return (NA / AnnuityFromZeroToM(x,i,m,lx) )* a


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

#Ex
def Ex(x,lx):
    if x+1 == len(lx):
        return 0
    return sum(lx[x+1:])/lx[x]
#npx 
def NPX(x,n,lx):

#    print(n)
#    print(lx[int(x+n)]/lx[x])
    return lx[int(x+n)]/lx[x]

#nqx 
def NQX(x,n,lx):
    return (lx[x] - lx[x+n] ) / lx[x]


#mnqx 
def MNQX(x,n,m,lx):
    return NPX(x,m,lx) * NQX(x+m,n,lx)


#techDF - actualization factor
def TechDF(n,i):
    return 1 / ((1+i)**n)


def StressTest_table(TH,stress):
    tablelx=TH.copy()
    tablelx.append(0)
    new_Qx=list()
    tablelx[0]=100000
    for i in range(0,len(TH)):
        new_Qx.append(Qx(i,TH)*(1-stress))

        tablelx[i+1]=(tablelx[i]*(1-new_Qx[i]))
    return(tablelx,new_Qx)

def TI_evolution(TH,x,i,n,m,a):
    TI=list()
    listi=list()
    for j in range(1,100):
        tablelx=StressTest_table(TH,j/100)[0]
        TI.append(TermInsuranceAnnual(x,n,i,a,m,tablelx))
        listi.append(j)
#    plt.plot(listi,TI)
#    plt . xlabel ('discount (%)', fontsize =20)
#    plt . ylabel ('Annual premium', fontsize =20)
#    plt . title ('Evolution of the Annual premium of a term insurance with respect to the discount',fontsize =16)
    return(TI)    


##############################################

#Here begins the function we will use to plot#

##############################################

def lx_evolution(stress=0):
    TH_bis=TH.copy()    
    TH_bis.append(0)
    return [i for i in range(0,len(StressTest_table(TH,stress)[0]))],StressTest_table(TH,stress)[0], TH_bis

def qx_evolution(stress=0):
    return [i for i in range(0,len(StressTest_table(TH,stress)[0]))], StressTest_table(TH,stress)[1],StressTest_table(TH,0)[1]

  
def plot_p_and_l_point_interest(TH,x,i,n,m,a):
    P_and_L=list()
    TI=TermInsuranceAnnual(x,n,i,a,m,TH) 
    stresslist=list()
    for stress in range(-15,25):
        stress=stress/100
        stresslist.append(stress)        
        P_and_L.append(TI-TermInsuranceAnnual(x,n,i+stress,a,m,TH))
    # plt.xlabel('Interest rate increase', fontsize=20)    
    # plt . ylabel ('Profit and loss', fontsize =20)
    # plt . title ('P & L as a function of the interest rate stress ',fontsize =16)
    # p1,=plt.plot(stresslist,P_and_L,label='Profit_and_loss',color='red')
    # plt . legend ( handles =[p1],fontsize =16)

    # plt.plot()         
    return stresslist, P_and_L
    


def plot_p_and_l_point(TH,x,i,n,m,a):
    P_and_L=list()
    TI=TermInsuranceAnnual(x,n,i,a,m,TH) 
    stresslist=list()
    for stress in range(-10,10):
        stress=stress/100
        stresslist.append(stress)        
        TH_stressed=StressTest_table(TH,stress)[0]        
        P_and_L.append(TI-TermInsuranceAnnual(x,n,i,a,m,TH_stressed))
    # plt.xlabel('Stress ', fontsize=20)    
    # plt . ylabel ('Profit and loss', fontsize =20)
    # plt . title ('P & L with stress on mortality table',fontsize =16)
    # p1,=plt.plot(stresslist,P_and_L,label='Profit_and_loss',color='red')
    # plt . legend ( handles =[p1],fontsize =16)
    # plt.plot()     
    return stresslist, P_and_L

def plot_p_and_l_point_knn(TH,x,i,n,m,a,degree=8):
    P_and_L=list()
    TI=term_insurance_predicted_polynomiale_scaled(x,m,n,i,a,degree)
    stresslist=list()
    for stress in range(-15,15):
        stress=stress/100
        stresslist.append(stress)        
        TH_stressed=StressTest_table(TH,stress)[0]        
        P_and_L.append(TI-TermInsuranceAnnual(x,n,i,a,m,TH_stressed))
    # plt.xlabel('Stress ', fontsize=20)    
    # plt . ylabel ('Profit and loss', fontsize =20)
    # plt . title ('P & L as a function of the stress ',fontsize =16)
    # p1,=plt.plot(stresslist,P_and_L,label='Profit_and_loss',color='red')
    # plt . legend ( handles =[p1],fontsize =16)

    # plt.plot()        
    return stresslist, P_and_L

def plot_p_and_l_point_interest_knn(TH,x,i,n,m,a,degree=8):
    P_and_L=list()
    TI=term_insurance_predicted_polynomiale_scaled(x,m,n,i,a,degree)
    stresslist=list()
    for stress in range(-15,25):
        stress=stress/1000
        stresslist.append(stress)        
        P_and_L.append(TI-TermInsuranceAnnual(x,n,i+stress,a,m,TH))
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
        for stress in range(-15,15):
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

    return  y_test_predict     
    
    




    
def new_p_and_l_point(x,m,n,i,a,stress=0,stress_interest=0):
    rec= best_model_scale_knn(stress,stress_interest,X=X)
    stress=stress/100
    stress_i=stress_interest/100
    premium_to_predict=[[x,m,n,i+stress_i,a]] 
    prediction = rec[0].predict(premium_to_predict)
    stress=stress/100
    stress_i=stress_interest/100
    TH_stressed=StressTest_table(TH,stress)[0]        
    P_and_L= prediction-TermInsuranceAnnual(x,n,i+stress_i,a,m,TH_stressed)
    return P_and_L[0]               
    
def best_model_scale_knn(stress_MT=0,stress_interest=0,X=X):
    ##put the stress in %
    y_stressed=list()
    stress=stress_MT/100
    stress_i=stress_interest/100
    TH_stressed=StressTest_table(TH,stress)[0]
    X_new=X.copy()
    X_new.interest_rate=X_new.interest_rate+stress_i
    for contract in range(0,len(X)):
        y_stressed.append(TermInsuranceAnnual(int(X_new.iloc[contract].age),int(X_new.iloc[contract].maturity),X_new.iloc[contract].interest_rate,X_new.iloc[contract].amount,int(X_new.iloc[contract].nb_payements),TH_stressed))    

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
        TH_stressed=StressTest_table(TH,stress)[0]
        stresslist.append(stress)
        for contract in range(0,len(X)):
            TI_stressed.append(TermInsuranceAnnual(int(X.iloc[contract].age),int(X.iloc[contract].maturity),X.iloc[contract].interest_rate,X.iloc[contract].amount,int(X.iloc[contract].nb_payements),TH_stressed))
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
            TI_stressed.append(TermInsuranceAnnual(int(X.iloc[contract].age),int(X.iloc[contract].maturity),X.iloc[contract].interest_rate+stress,X.iloc[contract].amount,int(X.iloc[contract].nb_payements),TH))
        p_and_l.append(knn_sum-sum(TI_stressed))

    # plt.xlabel('Interest rate increase', fontsize=20)    
    # plt . ylabel ('Profit and loss', fontsize =20)
    # plt . title ('Profit and loss with stress on the interest rates',fontsize =16)
    # p1,=plt.plot(stresslist, p_and_l,label='Profit_and_loss',color='red')
    # plt . legend ( handles =[p1],fontsize =16)

    # plt.plot() 
    return stresslist, p_and_l

         
    

#def best_model_stress(stress_MT=0,stress_interest=0,X=X):
#    ##put the stress in %
#    y_stressed=list()
#    stress=stress_MT/100
#    stress_i=stress_interest/100
#    TH_stressed=StressTest_table(TH,stress)[0]
#    X_new=X.copy()
#    X_new.interest_rate=X.interest_rate+stress_i
#    for contract in range(0,len(X)):
#        y_stressed.append(TermInsuranceAnnual(int(X_new.iloc[contract].age),int(X_new.iloc[contract].maturity),X_new.iloc[contract].interest_rate,X_new.iloc[contract].amount,int(X_new.iloc[contract].nb_payements),TH_stressed))    
#
#    X_trainval, X_test, y_trainval, y_test = train_test_split(X_new, y_stressed, random_state=1)
#
#    X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)
#    valid_scores=list()
#    test_scores=list()
#    train_scores=list()
#    nn_list=list()
#    for nn in range (1,20):
#        model=KNeighborsRegressor(n_neighbors=nn,weights="distance")
#        model.fit(X_train, y_train)
#        valid_scores.append(model.score(X_valid,y_valid))
#        test_scores.append(model.score(X_test,y_test))
#        train_scores.append(model.score(X_train,y_train))
#        nn_list.append(nn)
#    m = max(test_scores)
#    max_nn=0
#    for nn in  range (1,20):
#        if test_scores[nn-1]==m:
#            max_nn=nn
#    model=KNeighborsRegressor(n_neighbors=max_nn,weights="distance")
#    model.fit(X_train, y_train)       
#    return model, nn_list, train_scores, valid_scores, test_scores, y_stressed    

# def plot_stress_model(stress=0,stress_interest=0):
#     ##put stres in %
#     rec=best_model_scale_knn(stress_MT=0,stress_interest=0,X=X)
#     listnn=rec[1]
#     p1,=plt.plot(listnn,rec[2],label='train')
#     p2,=plt.plot(listnn,rec[3],label='validation')
#     p3,=plt.plot(listnn,rec[4],label='test')
#     plt . xlabel ('number of neighbors', fontsize =20)
#     plt . ylabel ('R²', fontsize =20)
#     plt . title ('R² as a function of the number of neighbors with a stress on mortality table= {}'.format(stress)+'%'+" and on interest rate = {}".format(stress_interest)+ "%",fontsize =16)
#     plt . legend ( handles =[p1 , p2, p3],fontsize =16)
#     plt.show()


        
def plot_p_and_l_knn_sum_stress_new():
    p_and_l=list()
    stresslist=list()
    for stress in range(-10,10):
        p_and_l.append(new_p_and_l_sum(stress=stress,stress_interest=0))
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
        p_and_l.append(new_p_and_l_sum(stress=0,stress_interest=stress/10))
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
    X_new.interest_rate=X_new.interest_rate+stress_interest/100
    return sum((rec[0]).predict(X_new))-sum(rec[5])


def plot_p_and_l_sum():
    p_and_l=list()
    stresslist=list()
    for stress in range(-10,10):
        TI=list()
        TI_stressed=list()
        stress=stress/100
        TH_stressed=StressTest_table(TH,stress)[0]
        stresslist.append(stress)
        for contract in range(0,len(X)):
            TI.append(TermInsuranceAnnual(int(X.iloc[contract].age),int(X.iloc[contract].maturity),X.iloc[contract].interest_rate,X.iloc[contract].amount,int(X.iloc[contract].nb_payements),TH))
            TI_stressed.append(TermInsuranceAnnual(int(X.iloc[contract].age),int(X.iloc[contract].maturity),X.iloc[contract].interest_rate,X.iloc[contract].amount,int(X.iloc[contract].nb_payements),TH_stressed))
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
            TI.append(TermInsuranceAnnual(int(X.iloc[contract].age),int(X.iloc[contract].maturity),X.iloc[contract].interest_rate,X.iloc[contract].amount,int(X.iloc[contract].nb_payements),TH))
            TI_stressed.append(TermInsuranceAnnual(int(X.iloc[contract].age),int(X.iloc[contract].maturity),X.iloc[contract].interest_rate+stress,X.iloc[contract].amount,int(X.iloc[contract].nb_payements),TH))
        p_and_l.append(sum(TI)-sum(TI_stressed))

    # plt.xlabel('Interest rate increase', fontsize=20)    
    # plt . ylabel ('Profit and loss', fontsize =20)
    # plt . title ('Profit and loss with stress on the interest rates ',fontsize =16)
    # p1,=plt.plot(stresslist, p_and_l,label='Profit_and_loss',color='red')
    # plt . legend ( handles =[p1],fontsize =16)

    # plt.plot() 
    return stresslist, p_and_l





# def linear_error(m,i,x,stress):
#         TH_stressed=StressTest_table(TH,stress)[0]        
#         somme=0
#         somme_stress=0
#         for index in range (0,m):
#             somme=somme+NPX(x,m,TH)*1/(1+i)**index
#             somme_stress=somme_stress+NPX(x,m,TH_stressed)*1/(1+i)**index            
#         return(somme-somme_stress)     
        
        
# def plot_error(i=0.01,stress=10/100,m=1):
#     s=list()
#     g=list()
#     for age in range(1,100):
#         s.append(linear_error(m,i,age,stress))
#         g.append(age)
#     plt.xlabel('age', fontsize=20)    
#     plt . ylabel ('linear error', fontsize =20)
#     plt . title ('evolution of the linear error',fontsize =16)
#     p1,=plt.plot(g,s)
#     plt . legend ( handles =[p1],fontsize =16)
        
#     plt.plot()   
    