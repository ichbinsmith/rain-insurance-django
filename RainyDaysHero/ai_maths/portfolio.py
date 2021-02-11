
from .terminsurance import reserves,stresstest,balance_sheet
from .pureendowment import Pure_endowment, PE_reserves, balancesheetPE, stresstest_PE


import matplotlib.pyplot as plt
import  reserves,balance_sheet, balancesheetPE, PE_reserves
import numpy as np
import os
import pandas as pd

TH = [100000,99511,99473,99446,99424,99406,99390,99376,99363,99350,99338,99325,99312,99296,99276,99250,99213,99163,99097,99015,98921,98820,98716,98612,98509,98406,98303,98198,98091,97982,97870,97756,97639,97517,97388,97249,97100,96939,96765,96576,96369,96141,95887,95606,95295,94952,94575,94164,93720,93244,92736,92196,91621,91009,90358,89665,88929,88151,87329,86460,85538,84558,83514,82399,81206,79926,78552,77078,75501,73816,72019,70105,68070,65914,63637,61239,58718,56072,53303,50411,47390,44234,40946,37546,34072,30575,27104,23707,20435,17338,14464,11852,9526,7498,5769,4331,3166,2249,1549,1032,663,410,244,139,75,39,19,9,4,2,1]
lx = TH
df = pd.read_csv(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/static/RainyDaysHero/data/LI/TI/dataset.csv")

X =df[['age','nb_payements','maturity','interest_rate','amount']]

y = df['target']
##We will consider that we are working with a population of 1000 people and we will use the mortality table to know how many people we have
##  And then take the percentage given by assurland.com
##ffa assurance gives us the number in 2019: 83% of insurances for retirement and 17% for health/death

age_20=int(TH[20]*21/100000)
age_30=int(TH[30]*31.4/100000)
age_40=int(TH[40]*35.1/100000)
age_50=int(TH[50]*36.5/100000)
age_60=int(TH[60]*41.8/100000)
age_70=int(TH[70]*43/100000)

all_age=[age_20,age_30,age_40,age_50,age_60,age_70]
v = locals()
j=0
for inc in range(20,70,10):
    v["var%d" % inc] = all_age[j]   
    print(all_age[j])
    j+=1
    
##We will now compute the balance sheet and multiply all the contratcs computed by the percentage of people of this age


def Portfolio_true(stress_MT=0,stress_interest_rates=0, adapt=True):
    listcontract=np.zeros((40,8))
    for contract in range(0,len(X)):
        x=int(X.iloc[contract].age)
        m=int(X.iloc[contract].nb_payements)
        n=int(X.iloc[contract].maturity)
        i=X.iloc[contract].interest_rate
        a=X.iloc[contract].amount
        bsTI=balance_sheet.balance_sheet_true(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)
        bsPE=balancesheetPE.balance_sheet_true(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)
        for term in range(0,n):
            for smash in range (0,7):
                listcontract[term][smash]=listcontract[term][smash]+bsTI[smash][term]*v["var%d" % x]*(17/100)+bsPE[smash][term]*v["var%d" % x] *(83/100)
            listcontract[term][7]=int(term+1)        
    return(listcontract)   
        
def plot_portfolio_true(stress_MT=0,stress_interest_rates=0, adapt=True):
    test= Portfolio_true(stress_MT,stress_interest_rates, adapt)
    Premiums=list([0])
    Claims=list([0])
    Reserves=list([0])
    for h in range(0,40):
        Reserves.append(test[h][4])
        Premiums.append(test[h][0])
        Claims.append(test[h][3])        
    # plt.close()    
    # p1,=plt.plot(np.arange(1,42,1),Reserves,label='Reserves')    
    # p2,=plt.plot(np.arange(1,42,1),Claims,label='Claims')
    # p3,=plt.plot(np.arange(1,42,1),Premiums,label='Level annual premiums')
    # plt . xlabel ('Years', fontsize =20)
    # plt . ylabel ('Reserves', fontsize =20)
    # plt . title ('Portfolio reserves with stress on mortality table={}'.format(stress_MT)+"%"+" and stress on interest rates={}".format(stress_interest_rates)+"%",fontsize =16)
    # plt . legend ( handles =[p1 , p2,p3],fontsize =16)
    # plt.show()   
    return([i for i in range(1,42)],Reserves,Claims,Premiums)

def Portfolio_predicted(stress_MT=0,stress_interest_rates=0, adapt=True):
    listcontract=np.zeros((40,8))
    if adapt==True:
        ## First, We compute the best model
        modelTI= reserves.best_model_scale_knn(stress_MT,stress_interest_rates,X=X)
        modelPE=PE_reserves.best_model_scale_knn(stress_MT,stress_interest_rates,X=X)
    else:        
        modelTI= reserves.best_model_scale_knn(stress_MT=0,stress_interest_rates=0,X=X)  
        modelPE=PE_reserves.best_model_scale_knn(stress_MT=0,stress_interest=0,X=X)

    for contract in range(0,len(X)):
        x=int(X.iloc[contract].age)
        m=int(X.iloc[contract].nb_payements)
        n=int(X.iloc[contract].maturity)
        i=X.iloc[contract].interest_rate
        a=X.iloc[contract].amount
        bsTI=balance_sheet.balance_sheet_predicted_model(x,n,i,a,m,modelTI,stress_MT,stress_interest_rates, adapt)
        bsPE=balancesheetPE.balance_sheet_predicted_model(x,n,i,a,m,modelPE,stress_MT,stress_interest_rates, adapt)
        for term in range(0,n):
            for smash in range (0,7):
                listcontract[term][smash]=listcontract[term][smash]+bsTI[smash][term]*v["var%d" % x]*(17/100)+bsPE[smash][term]*v["var%d" % x]*(83/100) 
            listcontract[term][7]=int(term+1)        
    return(listcontract)   
        
def plot_portfolio_predicted(stress_MT=0,stress_interest_rates=0, adapt=True):
    test= Portfolio_predicted(stress_MT,stress_interest_rates, adapt)
    Premiums=list([0])
    Claims=list([0])
    Reserves=list([0])
    for h in range(0,40):
        Reserves.append(test[h][4])
        Premiums.append(test[h][0])
        Claims.append(test[h][3])       
    # plt.close()
    # p1,=plt.plot(np.arange(1,42,1),Reserves,label='Reserves')    
    # p2,=plt.plot(np.arange(1,42,1),Claims,label='Claims')
    # p3,=plt.plot(np.arange(1,42,1),Premiums,label='Level annual premiums')
    # plt . xlabel ('Years', fontsize =20)
    # plt . ylabel ('Reserves', fontsize =20)
    # plt . title ('Portfolio reserves with stress on mortality table={}'.format(stress_MT)+"%"+" and stress on interest rates={}".format(stress_interest_rates)+"%",fontsize =16)
    # plt . legend ( handles =[p1 , p2,p3],fontsize =16)
    # plt.show()   
    return([i for i in range(1,42)],Reserves,Claims,Premiums)

