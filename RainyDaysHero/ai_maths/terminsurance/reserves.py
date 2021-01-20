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
        model=stresstest. best_model_stress(stress_MT,stress_interest_rates,X)[0]
        stress=stress_MT/100
        stress_i=stress_interest_rates/100
        newTH=stresstest.StressTest_table(TH,stress)[0]
        i=i+stress_i
        listcontract=np.zeros((1,n+1))
        annual_premium= model.predict([[x,m,n,i,a]])
        for term in range(1,n+1):
            qx=stresstest.Qx(x+term-1,newTH)
            down=1
            if (term<=m):
                left=listcontract[0][term-1]+annual_premium
            else:
                left=listcontract[0][term-1]
            right=a*qx
            listcontract[0][term]=(left-right)/down
        recurrence1=listcontract[0]       
    else:        
        model=stresstest. best_model_stress(0,0,X)[0]        
        stress=stress_MT/100
        stress_i=stress_interest_rates/100
        i=i+stress_i        
        newTH=stresstest.StressTest_table(TH,stress)[0]
        listcontract=np.zeros((1,n+1))
        annual_premium= model.predict([[x,m,n,i,a]])
        for term in range(1,n+1):
            qx=stresstest.Qx(x+term-1,newTH)
            down=1
            if (term<=m):
                left=listcontract[0][term-1]+annual_premium
            else:
                left=listcontract[0][term-1]
            right=a*qx
            listcontract[0][term]=(left-right)/down
        recurrence1=listcontract[0]    

    #put stress in %
    if adapt==True:
        stress=stress_MT/100
        stress_i=stress_interest_rates/100
        newTH=stresstest.StressTest_table(TH,stress)[0]
        listcontract=np.zeros((1,n+1))
        annual_premium=stresstest.TermInsuranceAnnual(x,n,i,a,m,newTH) 
        for term in range(1,n+1):
            qx=stresstest.Qx(x+term-1,newTH)
            down=1/(1+i)*(1-qx)
            if (term<=m):
                left=listcontract[0][term-1]+annual_premium
            else:
                left=listcontract[0][term-1]
            right=a*(1/(1+i))*qx
            listcontract[0][term]=(left-right)/down
        recurrence2=listcontract[0]
    else:
        stress=stress_MT/100
        stress_i=stress_interest_rates/100
        listcontract=np.zeros((1,n+1))
        annual_premium=stresstest.TermInsuranceAnnual(x,n,i-stress_i,a,m,TH) 
        for term in range(1,n+1):
            qx=stresstest.Qx(x+term-1,newTH)
            down=1/(1+i)*(1-qx)
            if (term<=m):
                left=listcontract[0][term-1]+annual_premium
            else:
                left=listcontract[0][term-1]
            right=a*(1/(1+i))*qx
            listcontract[0][term]=(left-right)/down
        recurrence2=listcontract[0]     
    p1,=plt.plot(np.arange(0,n+1,1),recurrence1,label='KNN model')
    p2,=plt.plot(np.arange(0,n+1,1),recurrence2,label='real reserves')
    plt . xlabel ('Years', fontsize =20)
    plt . ylabel ('Reserves', fontsize =20)
    plt . title ('Reserves with stress on mortality table={}'.format(stress*100)+"%"+" and stress on interest rates={}".format(stress_i*100)+"%",fontsize =16)
    plt . legend ( handles =[p1, p2],fontsize =16)
    plt.show()          




def reserves_sum(stress_MT=0,stress_interest_rates=0,adapt=True):
    #put stress in %
    if adapt==True:
        stress=stress_MT/100
        stress_i=stress_interest_rates/100
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
               down=1/(1+i)*(1-qx)
               if (term<=m):
                   left=listcontract[contract][term-1]+annual_premium
               else:
                   left=listcontract[contract][term-1]
               right=a*(1/(1+i))*qx
               listcontract[contract][term]=(left-right)/down
        recurrence2=list()       
        for term in range(0,41):       
           ## print(np.sum(listcontract[:,40]))
            recurrence2.append(np.sum(listcontract[:,term]))
    else:
        stress=stress_MT/100
        stress_i=stress_interest_rates/100
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
               if (term<=m):
                   left=listcontract[contract][term-1]+annual_premium
               else:
                   left=listcontract[contract][term-1]
               right=a*(1/(1+i+stress_i))*qx
               listcontract[contract][term]=(left-right)/down
        recurrence2=list()       
        for term in range(0,41):       
           ## print(np.sum(listcontract[:,40]))
            recurrence2.append(np.sum(listcontract[:,term]))
        
    p2,=plt.plot(np.arange(0,41,1),recurrence2,label='real reserves')
    plt . xlabel ('Years', fontsize =20)
    plt . ylabel ('Reserves', fontsize =20)
    plt . title ('Reserves with stress on mortality table={}'.format(stress*100)+"%"+" and stress on interest rates={}".format(stress_i*100)+"%",fontsize =16)
    plt.show()
    return(recurrence2)          



def reserves_sum_knn(stress_MT=0,stress_interest_rates=0,adapt=True):
    #put stress in %
    if adapt==True:
        ## First, We compute the best model
        model=stresstest. best_model_stress(stress_MT,stress_interest_rates,X)[0]
        stress=stress_MT/100
        stress_i=stress_interest_rates/100
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
               down=1
               if (term<=m):
                   left=listcontract[contract][term-1]+annual_premium
               else:
                   left=listcontract[contract][term-1]
               right=a*qx
               listcontract[contract][term]=(left-right)/down
        recurrence2=list()       
        for term in range(0,41):       
           ## print(np.sum(listcontract[:,40]))
            recurrence2.append(np.sum(listcontract[:,term]))
    else:
        
        model=stresstest. best_model_stress(0,0,X)[0]        
        stress=stress_MT/100
        stress_i=stress_interest_rates/100
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
               down=1
               if (term<=m):
                   left=listcontract[contract][term-1]+annual_premium
               else:
                   left=listcontract[contract][term-1]
               right=a*qx
               print(left-right)
               listcontract[contract][term]=(left-right)/down
        recurrence2=list()       
        for term in range(0,41):       
           ## print(np.sum(listcontract[:,40]))
            recurrence2.append(np.sum(listcontract[:,term]))
        
    p2,=plt.plot(np.arange(0,41,1),recurrence2,label='real reserves')
    plt . xlabel ('Years', fontsize =20)
    plt . ylabel ('Reserves', fontsize =20)
    plt . title ('KNN reserves with stress on mortality table={}'.format(stress*100)+"%"+" and stress on interest rates={}".format(stress_i*100)+"%",fontsize =16)
    plt.show()
    return(recurrence2)