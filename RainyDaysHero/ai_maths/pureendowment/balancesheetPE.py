
from RainyDaysHero.ai_maths.pureendowment import PE_reserves
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
TF = [100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,99987.943454803000000,99976.891621705800000,99966.844500708300000,99956.797379710800000,99946.750258713400000,99935.698425616100000,99924.646592518900000,99912.590047321900000,99899.528790025200000,99883.453396429200000,99863.359154434300000,99839.246064040300000,99810.109413147700000,99776.953913856000000,99741.788990364800000,99706.624066873600000,99672.463855482200000,99637.262739119000000,99602.061622755800000,99562.808587152700000,99520.536087272300000,99474.237635022500000,99422.906742310700000,99365.536921044500000,99302.128171224000000,99232.680492849200000,99157.193885920100000,99073.655374251900000,98981.058469752200000,98879.403172420900000,98766.676506073400000,98642.878470709600000,98507.002578237200000,98358.042340563700000,98195.997757689200000,98004.419419985400000,97798.724783713900000,97579.922155915400000,97348.011536589700000,97104.001232777500000,96846.882937438200000,96573.631729450100000,96282.230994732200000,95970.664119203400000,95638.931102863600000,95264.475758390500000,94869.779584486300000,94452.818498208100000,94010.556375141100000,93537.933007927700000,93029.888189210000000,92482.373753101900000,91837.624008223600000,91136.853353442800000,90372.931854771900000,89538.729578223200000,88625.079465812800000,87620.777335560000000,86517.674691479100000,85305.585913587900000,83972.288257907500000,82501.484732466400000,80663.431201173300000,78624.843847641900000,76364.941964904500000,73862.944845993400000,71093.915642547400000,68033.956541553500000,64662.286836044200000,60970.594243232300000,56968.230081253200000,52685.326375210200000,48171.756821825700000,43502.331966183500000,38775.760166380100000,33634.170688786900000,28741.863237795800000,24153.183658068700000,19508.781438646500000,15400.161585067600000,11857.408133916700000,8887.683951047370000,6469.500438899080000,4562.745548692110000,3110.116353598300000,2042.849341423850000,1289.315853633560000,779.319804862954000,449.827975601043000,246.402585361081000,127.499012192652000,63.033219510974200,28.651463414079200,12.893158536335600,5.730292682815840,1.432573170703960]

lx = TF

df = pd.read_csv(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/static/RainyDaysHero/data/LI/PE/dataset.csv")

X =df[['age','nb_payements','maturity','interest_rate','amount']]

y = df['target']

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=5)

X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)
def NPX(x,n,lx):
#    print(n)
#    print(lx[int(x+n)]/lx[x])
    return lx[int(x+n)]/lx[x]



def balance_sheet_true(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True): 
    if adapt==True:
        bs=PE_reserves.reserves_true(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)
        Premiums=bs[1+1]
        Financial_income=(bs[0+1]+Premiums)*i
        Last_premium_reserves=bs[0+1]
        Claims=bs[2+1]*(1+i)
        Premium_reserves=bs[0+1][1:len(bs[0+1])]
        Premium_reserves=Premium_reserves+[0]    
        Total_Asset=list()
        Total_liability=list()
        for age in range(0,len(Premium_reserves)):
             Premium_reserves[age]= Premium_reserves[age]*NPX(x+age,1,PE_reserves.StressTest_table(TF,stress_MT)[0])
             Total_Asset.append(Financial_income[age]+Premiums[age]+ Last_premium_reserves[age])
             Total_liability.append(Claims[age]+Premium_reserves[age])
    else:
        bs=PE_reserves.reserves_true(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)
        Premiums=bs[1+1]
        Financial_income=(bs[0+1]+Premiums)*i
        Last_premium_reserves=bs[0+1]
        Claims=bs[2+1]*(1+i)
        Premium_reserves=bs[1+0][1:len(bs[1])]
        Premium_reserves=Premium_reserves+[0]    
        Total_Asset=list()
        Total_liability=list()
        for age in range(0,len(Premium_reserves)):
             Premium_reserves[age]= Premium_reserves[age]*NPX(x+age,1,PE_reserves.StressTest_table(TF,stress_MT)[0])
             Total_Asset.append(Financial_income[age]+Premiums[age]+ Last_premium_reserves[age])
             Total_liability.append(Claims[age]+Premium_reserves[age])
            
    return Premiums[0:-1],Financial_income[0:-1],Last_premium_reserves[0:-1],Claims[0:-1],Premium_reserves,Total_Asset, Total_liability

def balance_sheet_knn(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True): 
    if adapt==True:
        bs=PE_reserves.reserves_predicted_scale_knn(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)
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
            # Premium_reserves[age]= Premium_reserves[age]*NPX(x+age,1,PE_reserves.StressTest_table(TF,stress_MT)[0])
    else:
        bs=PE_reserves.reserves_predicted_scale_knn(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)
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
            # Premium_reserves[age]= Premium_reserves[age]*NPX(x+age,1,PE_reserves.StressTest_table(TF,stress_MT)[0])
    return Premiums[0:-1],Financial_income[0:-1],Last_premium_reserves[0:-1],Claims[0:-1],Premium_reserves,Total_Asset, Total_liability

#We now compute the total balance sheet

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
            listcontract[term][7]=term+1      
    return(listcontract)   
        

def total_balance_sheet_predicted(stress_MT=0,stress_interest_rates=0, adapt=True):
    if adapt==True:
        ## First, We compute the best model
        model= PE_reserves.best_model_scale_knn(stress_MT,stress_interest_rates,X=X)
    else:        
        model= PE_reserves.best_model_scale_knn(stress_MT=0,stress_interest=0,X=X)  
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
            listcontract[term][7]=term+1      
    return(listcontract)   
    
   

def reserves_predicted_model(x,n,i,a,m,model,stress_MT=0,stress_interest_rates=0, adapt=True):
    if adapt==True:
        ## First, We compute the best model
        stress=stress_MT
        stress_i=stress_interest_rates
        newTF=PE_reserves.StressTest_table(TF,stress)[0]
        i=i+stress_i
        listcontract=np.zeros((1,n+1))
        list_annual_premium1=np.zeros((1,n+1))
        list_natural_premium1=np.zeros((1,n+1))        
        annual_premium= model.predict([[x,m,n,i,a]])
        for term in range(1,n+1):
            qx=PE_reserves.Qx(x+term-1,newTF)
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
        list_annual_premium1=np.zeros((1,n+1))
        list_natural_premium1=np.zeros((1,n+1))        
        stress=stress_MT
        stress_i=stress_interest_rates
        newTF=PE_reserves.StressTest_table(TF,stress)[0]
        listcontract=np.zeros((1,n+1))
        annual_premium= model.predict([[x,m,n,i,a]])
        for term in range(1,n+1):
            qx=PE_reserves.Qx(x+term-1,newTF)
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
             Total_Asset.append(Financial_income[age]+Premiums[age]+ Last_premium_reserves[age])
             Premium_reserves[age]=Total_Asset[age]-Claims[age]
             Total_liability.append(Claims[age]+Premium_reserves[age])            
            # Premium_reserves[age]= Premium_reserves[age]*NPX(x+age,1,PE_reserves.StressTest_table(TF,stress_MT)[0])
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
             Total_Asset.append(Financial_income[age]+Premiums[age]+ Last_premium_reserves[age])
             Premium_reserves[age]=Total_Asset[age]-Claims[age]
             Total_liability.append(Claims[age]+Premium_reserves[age])            
            # Premium_reserves[age]= Premium_reserves[age]*NPX(x+age,1,PE_reserves.StressTest_table(TF,stress_MT)[0])          
    return Premiums[0:-1],Financial_income[0:-1],Last_premium_reserves[0:-1],Claims[0:-1],Premium_reserves,Total_Asset, Total_liability









