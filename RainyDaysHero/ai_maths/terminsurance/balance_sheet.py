# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 01:27:52 2021

@author: Mon PC
"""

from RainyDaysHero.ai_maths.terminsurance import  reserves,stresstest
import numpy as np
import os
import pandas as pd

#def true_balance_sheet_sum(stress_MT=0,stress_interest_rates=0,adapt=True):
#    total=reserves.reserves_sum(stress_MT,stress_interest_rates,adapt)
#    Premiums=total[3]
#    Financial_income=total
#         
#         (np.arange(0,41,1),reserve_total,natural_premium_total,level_annual_premium_total)  
#
TH = [100000,99511,99473,99446,99424,99406,99390,99376,99363,99350,99338,99325,99312,99296,99276,99250,99213,99163,99097,99015,98921,98820,98716,98612,98509,98406,98303,98198,98091,97982,97870,97756,97639,97517,97388,97249,97100,96939,96765,96576,96369,96141,95887,95606,95295,94952,94575,94164,93720,93244,92736,92196,91621,91009,90358,89665,88929,88151,87329,86460,85538,84558,83514,82399,81206,79926,78552,77078,75501,73816,72019,70105,68070,65914,63637,61239,58718,56072,53303,50411,47390,44234,40946,37546,34072,30575,27104,23707,20435,17338,14464,11852,9526,7498,5769,4331,3166,2249,1549,1032,663,410,244,139,75,39,19,9,4,2,1]

lx = TH

df = pd.read_csv(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/static/RainyDaysHero/data/LI/TI/dataset.csv")

X =df[['age','nb_payements','maturity','interest_rate','amount']]

y = df['target']



def balance_sheet_true(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True): 
    if adapt==True:
        bs=reserves.reserves_true(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)
        Premiums=bs[1+1]
        Financial_income=(bs[0+1]+Premiums)*i
        Last_premium_reserves=bs[0+1]
        Claims=bs[2+1]*(1+i)
        Premium_reserves=bs[0+1][1:len(bs[0+1])]
        Premium_reserves=Premium_reserves+[0]    
        Total_Asset=list()
        Total_liability=list()
        for age in range(0,len(Premium_reserves)):
             Premium_reserves[age]= Premium_reserves[age]*stresstest.NPX(x+age,1,stresstest.StressTest_table(TH,stress_MT)[0])
             Total_Asset.append(Financial_income[age]+Premiums[age]+ Last_premium_reserves[age])
             Total_liability.append(Claims[age]+Premium_reserves[age])
    else:
        bs=reserves.reserves_true(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)
        Premiums=bs[1+1]
        Financial_income=(bs[0+1]+Premiums)*i
        Last_premium_reserves=bs[0+1]
        Claims=bs[2+1]*(1+i)
        Premium_reserves=bs[0+1][1:len(bs[1])]
        Premium_reserves=Premium_reserves+[0]    
        Total_Asset=list()
        Total_liability=list()
        for age in range(0,len(Premium_reserves)):
             Premium_reserves[age]= Premium_reserves[age]*stresstest.NPX(x+age,1,stresstest.StressTest_table(TH,stress_MT)[0])
             Total_Asset.append(Financial_income[age]+Premiums[age]+ Last_premium_reserves[age])
             Total_liability.append(Claims[age]+Premium_reserves[age])
            
    return Premiums[0:-1],Financial_income[0:-1],Last_premium_reserves[0:-1],Claims[0:-1],Premium_reserves,Total_Asset, Total_liability


def balance_sheet_knn(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True): 
    if adapt==True:
        bs=reserves.reserves_predicted_scale_knn(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)
        Premiums=bs[1+1]
        Financial_income=(bs[0+1]+Premiums)*i
        Last_premium_reserves=bs[0+1]
        Claims=bs[2+1]*(1+i)
        Premium_reserves=bs[0+1][1:len(bs[0+1])]
        Premium_reserves=Premium_reserves+[0]    
        Total_Asset=list()
        Total_liability=list()
        for age in range(0,len(Premium_reserves)):
             Total_Asset.append(Financial_income[age]+Premiums[age]+ Last_premium_reserves[age])
             
             Premium_reserves[age]=Total_Asset[age]-Claims[age]
             Total_liability.append(Claims[age]+Premium_reserves[age])
    else:
        bs=reserves.reserves_predicted_scale_knn(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)
        Premiums=bs[1+1]
        Financial_income=(bs[0+1]+Premiums)*i
        Last_premium_reserves=bs[0+1]
        Claims=bs[2+1]*(1+i)
        Premium_reserves=bs[0+1][1:len(bs[0])]
        Premium_reserves=Premium_reserves+[0]    
        Total_Asset=list()
        Total_liability=list()
        for age in range(0,len(Premium_reserves)):
             Total_Asset.append(Financial_income[age]+Premiums[age]+ Last_premium_reserves[age])
            
             Premium_reserves[age]=Total_Asset[age]-Claims[age]
             Total_liability.append(Claims[age]+Premium_reserves[age])  
    return Premiums[0:-1],Financial_income[0:-1],Last_premium_reserves[0:-1],Claims[0:-1],Premium_reserves,Total_Asset, Total_liability





def reserves_predicted_model(x,n,i,a,m,model,stress_MT=0,stress_interest_rates=0, adapt=True):
    if adapt==True:
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
    return(recurrence1,list_annual_premium1[0],list_natural_premium1[0])    
        

def balance_sheet_predicted_model(x,n,i,a,m,model,stress_MT=0,stress_interest_rates=0, adapt=True): 
    if adapt==True:
        bs=reserves_predicted_model(x,n,i,a,m,model,stress_MT,stress_interest_rates, adapt)
        Premiums=bs[1]
        Financial_income=(bs[0]+Premiums)*i
        Last_premium_reserves=bs[0]
        Claims=bs[2]*(1+i)
        Premium_reserves=bs[0][1:len(bs[0])]
        Premium_reserves=Premium_reserves+[0]    
        Total_Asset=list()
        Total_liability=list()
        for age in range(0,len(Premium_reserves)):
             Premium_reserves[age]= Premium_reserves[age]*stresstest.NPX(x+age,1,TH)
             Total_Asset.append(Financial_income[age]+Premiums[age]+ Last_premium_reserves[age])
             Total_liability.append(Claims[age]+Premium_reserves[age])
    else:
        bs=reserves_predicted_model(x,n,i,a,m,model,stress_MT,stress_interest_rates, adapt)
        Premiums=bs[1]
        Financial_income=(bs[0]+Premiums)*i
        Last_premium_reserves=bs[0]
        Claims=bs[2]*(1+i)
        Premium_reserves=bs[0][1:len(bs[0])]
        Premium_reserves=Premium_reserves+[0]    
        Total_Asset=list()
        Total_liability=list()
        for age in range(0,len(Premium_reserves)):
             Premium_reserves[age]= Premium_reserves[age]*stresstest.NPX(x+age,1,TH)
             Total_Asset.append(Financial_income[age]+Premiums[age]+ Last_premium_reserves[age])
             Total_liability.append(Claims[age]+Premium_reserves[age])            
    return Premiums[0:-1],Financial_income[0:-1],Last_premium_reserves[0:-1],Claims[0:-1],Premium_reserves,Total_Asset, Total_liability




##  We now compute the total balance sheet

def total_balance_sheet_true(stress_MT=0,stress_interest_rates=0, adapt=True):
    listcontract=np.zeros((40,8))
    for contract in range(0,len(X)):
        x=int(X.iloc[contract].age)
        m=int(X.iloc[contract].nb_payements)
        n=int(X.iloc[contract].maturity)
        i=X.iloc[contract].interest_rate
        a=X.iloc[contract].amount
        bs=balance_sheet_true(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)
        for term in range(0,n):
            for smash in range (0,7):
                listcontract[term][smash]=listcontract[term][smash]+bs[smash][term]
            listcontract[term][7]=int(term+1)      
    return(listcontract)   
        

def total_balance_sheet_predicted(stress_MT=0,stress_interest_rates=0, adapt=True):
    if adapt==True:
        ## First, We compute the best model
        model= reserves.best_model_scale_knn(stress_MT,stress_interest_rates,X=X)
    else:        
        model= reserves.best_model_scale_knn(stress_MT=0,stress_interest=0,X=X)  
    listcontract=np.zeros((40,8))
    for contract in range(0,len(X)):
        x=int(X.iloc[contract].age)
        m=int(X.iloc[contract].nb_payements)
        n=int(X.iloc[contract].maturity)
        i=X.iloc[contract].interest_rate
        a=X.iloc[contract].amount
        bs= balance_sheet_predicted_model(x,n,i,a,m,model,stress_MT,stress_interest_rates, adapt)
        for term in range(0,n):
            for smash in range (0,7):
                listcontract[term][smash]=listcontract[term][smash]+bs[smash][term]
            listcontract[term][7]=int(term+1)      
    return(listcontract)   
    
    
    












