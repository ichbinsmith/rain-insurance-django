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
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import csv
import os

import statsmodels.api as sm
import statsmodels.formula.api as smf

## Premium computation module
os.chdir(os.path.abspath(os.path.dirname(__file__)))
from .premiumComputation import TermInsuranceAnnual

df = pd.read_csv(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/static/RainyDaysHero/data/LI/TI/dataset.csv")

X =df[['age','nb_payements','maturity','interest_rate','amount']]

y = df['target']

def get_r2_statsmodels(x, y, k=1):
    xpoly = np.column_stack([x**i for i in range(k+1)])    
    return sm.OLS(y, xpoly).fit().rsquared


##faire les plots en 3d

##analyse des données
def visualisation_données():
    df.quantile(q=0.5)
    df.quantile(q=0.25)
    df.quantile(q=0.75)
    print(np.mean(y))
    print(np.min(y))
    print(np.max(y))
    plt.plot(X.amount,y)
    plt.show()
    pd.plotting.scatter_matrix(df, figsize=(15,15),
                               marker='o', hist_kwds={'bins': 20}, s=60,
                               alpha=.8, cmap=mg.cm3)
    plt.show()


##séparaion des données

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=5)

X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)

## D'abord on entraine sur X train puis on test sur X_test. On modifie alors en fonction. Puis on valide avecX_valid


# Evaluation du training set
##lasso001 = Lasso(alpha=0.001, max_iter=100000,positive=True).fit(X_train, y_train)

def Polynomiale(degree):
    #entrainement du modèle
    polynomial_features= PolynomialFeatures(degree=degree)
    X_train_poly = polynomial_features.fit_transform(X_train)
    X_valid_poly = polynomial_features.fit_transform(X_valid)
    X_test_poly=polynomial_features.fit_transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    # Evaluation du training set
    
    y_train_predict = model.predict(X_train_poly)
    rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
    r2train = get_r2_statsmodels(y_train, y_train_predict)
 
    print('La performance du modèle sur la base dapprentissage')
    print('--------------------------------------')
    print("L'erreur quadratique moyenne est {}".format(rmse))
    print('le score R2 est {}'.format(r2train))
    print('\n')

    # valuation sur le test set
    y_test_predict = model.predict(X_test_poly)
    rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
    r2test = get_r2_statsmodels(y_test, y_test_predict)
    
    print('La performance du modèle sur la base de test')
    print('--------------------------------------')
    print("L'erreur quadratique moyenne est {}".format(rmse))
    print('le score R2 est {}'.format(r2test))
    print('\n')
        
    # Evaluation sur le validation set
    y_valid_predict = model.predict(X_valid_poly)
    rmse = (np.sqrt(mean_squared_error(y_valid, y_valid_predict)))
    r2valid = get_r2_statsmodels(y_valid, y_valid_predict)
 
    print('La performance du modèle sur la base de validation')
    print('--------------------------------------')
    print("L'erreur quadratique moyenne est {}".format(rmse))
    print('le score R2 est {}'.format(r2valid))
    print('\n')
    
    return r2train,r2valid,r2test, rmse, y_test_predict
##Nous dépassons enfin les 90% pour le test et la validation

def affichage_polynomiale(degremax):
##On va afficher le R**2 des données test, données de validation et données train en fonction du degré    
    liste_erreurs=np.zeros((4,degremax))
    for i in range(1,degremax):
        liste_erreurs[0,i]=(Polynomiale(i)[0])
        liste_erreurs[1,i]=(Polynomiale(i)[1])
        liste_erreurs[2,i]=(Polynomiale(i)[2])
        liste_erreurs[3,i]=i

    p1,=plt.plot(liste_erreurs[3,],liste_erreurs[0,],label='train')
    p2,=plt.plot(liste_erreurs[3,],liste_erreurs[1,],label='validation')
    p3,=plt.plot(liste_erreurs[3,],liste_erreurs[2,],label='test')
    plt . xlabel ('degree', fontsize =20)
    plt . ylabel ('R²', fontsize =20)
    plt . title ('R² as a function of the degree',fontsize =16)
    plt . legend ( handles =[p1 , p2, p3],fontsize =16)
    plt.show()


#y_test_predict=Polynomiale(6)[4]
#
#y_test_predict.to_csv (r'C:\Users\Ron\Desktop\export_dataframe.csv', index = False, header=True)
#with open('tests predits.csv', 'w') as csvfile:
#    fieldnames = ['first_name', 'last_name']
#    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#with open("names.csv", "w") as f:
#    w = csv.writer(f)
#    w.writerows(zip(y_test_predict, y_test))


def learning_curve(degree=1):
     polynomial_features= PolynomialFeatures(degree)
     r2train=np.zeros(len(X_train)-50)
     r2test=np.zeros(len(X_train)-50)
     r2valid=np.zeros(len(X_train)-50)
     listei=np.zeros(len(X_train)-50)
     for i in range(50,len(X_train)):
         print(i)
         X_train_new=X_train[0:i]
         y_train_new=y_train[0:i]
         X_train_poly=polynomial_features.fit_transform(X_train_new)
         X_test_poly=polynomial_features.fit_transform(X_test)
         X_valid_poly=polynomial_features.fit_transform(X_valid)
             # Evaluation du training set
         model = LinearRegression()
         model.fit(X_train_poly, y_train_new)         
         y_train_predict = model.predict(X_train_poly)

         r2train[i-50] = get_r2_statsmodels(y_train_new, y_train_predict)
         
         y_test_predict = model.predict(X_test_poly)
         r2test[i-50] = get_r2_statsmodels(y_test, y_test_predict)


         y_valid_predict = model.predict(X_valid_poly)
         r2valid[i-50] = get_r2_statsmodels(y_valid, y_valid_predict)
         
         listei[i-50]=i
#         if ((np.abs(r2test[i-50]-r2train[i-50]))<10**-2):
#             print(r2test[i-50]-r2train[i-50])
#             return r2train[i-50],r2test[i-50], r2valid[i-50], listei[i-50]       
     p1,=plt.plot(listei,r2train,label='train')
     p2,=plt.plot(listei,r2test,label='test')
     p3,=plt.plot(listei,r2valid,label='valid')     
     plt . xlabel ('training set size', fontsize =20)
     plt . ylabel ('R²', fontsize =20)
     plt . title ('learning curve',fontsize =16)
     plt . legend ( handles =[p1 , p2, p3],fontsize =16)
     plt.show()    
     return r2train,r2test, r2valid, listei   

def Polynomiale_lasso(degree,alpha):
    #entrainement du modèle
    polynomial_features= PolynomialFeatures(degree=degree)
    X_train_poly = polynomial_features.fit_transform(X_train)
    X_valid_poly = polynomial_features.fit_transform(X_valid)    
    X_test_poly=polynomial_features.fit_transform(X_test)

    # Evaluation du training set
    model = Lasso(alpha=alpha, max_iter=1000,positive=True).fit(X_train_poly, y_train)

    y_train_predict = model.predict(X_train_poly)
    rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
    r2train = get_r2_statsmodels(y_train, y_train_predict)
 
    print('La performance du modèle sur la base dapprentissage')
    print('--------------------------------------')
    print("L'erreur quadratique moyenne est {}".format(rmse))
    print('le score R2 est {}'.format(r2train))
    print('\n')
 
    
    # évaluation sur le test set
    y_test_predict = model.predict(X_test_poly)
    rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
    r2test = get_r2_statsmodels(y_test, y_test_predict)
    
    print('La performance du modèle sur la base de test')
    print('--------------------------------------')
    print("L'erreur quadratique moyenne est {}".format(rmse))
    print('le score R2 est {}'.format(r2test))
    print('\n')
    
        # Evaluation sur le validation set
    y_valid_predict = model.predict(X_valid_poly)
    rmse = (np.sqrt(mean_squared_error(y_valid, y_valid_predict)))
    r2valid = get_r2_statsmodels(y_valid, y_valid_predict)
 
    print('La performance du modèle sur la base de validation')
    print('--------------------------------------')
    print("L'erreur quadratique moyenne est {}".format(rmse))
    print('le score R2 est {}'.format(r2valid))
    print('\n')   
     
    return model,r2train, r2test, r2valid, rmse
##Nous dépassons enfin les 90% pour le test et la validation
### EN diminuant alpha on colle plus aux données et on risque le sur apprentissage (on part vers la droite dans les données)

def sweet_spot_poly_lasso():
    print("Lasso sweet spot")
    z = [10**(i) for i in range(-10,1)]
    r = []
    t = []
    u = []
    for v in z:
        lasso = Polynomiale_lasso(1,v)
        r.append(lasso[1])
        t.append(lasso[2])
        u.append(lasso[3])        
    p1,=plt.plot(z[::-1],r[::-1],label = "train")
    p2,=plt.plot(z[::-1],t[::-1],label = "test")
    p3,=plt.plot(z[::-1],u[::-1],label = "validation")    
    plt . legend ( handles =[p1 , p2,p3],fontsize =16)
    plt . title ('R² as a function of alpha',fontsize =16)
    plt.show()



def learning_curve_polyfinal(degree=6):
     X_train_new=np.zeros(len(X_train)-50)
     r2train=np.zeros(len(X_train)-50)
     r2test=np.zeros(len(X_train)-50)
     r2valid=np.zeros(len(X_train)-50)     
     listei=np.zeros(len(X_train)-50)
     for i in range(50,len(X_train)):
         print(i)
         X_train_new=X_train[0:i]
         y_train_new=y_train[0:i]
         predictions = polyfinal_learning(degree,X_train_new,y_train_new)         
         r2train[i-50] = predictions[2]         
         r2test[i-50] = predictions[1]   
         r2valid[i-50] = predictions[3]                  
         listei[i-50]=i
   
     p1,=plt.plot(listei,r2train,label='train')
     p2,=plt.plot(listei,r2test,label='test')
     p3,=plt.plot(listei,r2valid,label='valid')
     plt . xlabel ('training set size', fontsize =20)
     plt . ylabel ('R²', fontsize =20)
     plt . title ('learning curve',fontsize =16)
     plt . legend ( handles =[p1 , p2,p3],fontsize =16)
     plt.show()    
     return r2train,r2test, r2valid,listei      


def polyfinal_learning(degree=6,X_train=X_train,y_train=y_train):
    small= model_small_premium_learning(degree,X_train,y_train)
    ##big=model_big_premium(degree)
    big=model_Polynomiale_learning(degree,X_train,y_train)
    
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
        if (y_test_predict[i]<=1000):
            a=small.predict(X_test_poly[[i]])[0]
            y_predictionfinale.append(a)
        if (y_test_predict[i]>1000):
            y_predictionfinale.append(big.predict(X_test_poly[[i]])[0])

    for i in range(0,len(y_train_predict)):
        if (y_train_predict[i]<=1000):
            a=small.predict(X_train_poly[[i]])[0]
            y_predictiontrain.append(a)
        if (y_train_predict[i]>1000):
            y_predictiontrain.append(big.predict(X_train_poly[[i]])[0])
            
    for i in range(0,len(y_valid_predict)):
        if (y_valid_predict[i]<=1000):
            a=small.predict(X_valid_poly[[i]])[0]
            y_predictionvalid.append(a)
        if (y_valid_predict[i]>1000):
            y_predictionvalid.append(big.predict(X_valid_poly[[i]])[0])

    ##calculons les r2
    r2test = get_r2_statsmodels(y_test, y_predictionfinale)
    r2train = get_r2_statsmodels(y_train, y_predictiontrain)          
    r2valid = get_r2_statsmodels(y_valid, y_predictionvalid)              
    
    return(y_predictionfinale,r2test,r2train,r2valid)  
 


def affichage_polyfinal(degremax):
##On va afficher le R**2 des données test, données de validation et données train en fonction du degré    
    liste_erreurs=np.zeros((4,degremax))
    for i in range(1,degremax):
        liste_erreurs[0,i]=(polyfinal_learning(i)[2])
        liste_erreurs[1,i]=(polyfinal_learning(i)[3])
        liste_erreurs[2,i]=(polyfinal_learning(i)[1])
        liste_erreurs[3,i]=i

    p1,=plt.plot(liste_erreurs[3,],liste_erreurs[0,],label='train')
    p2,=plt.plot(liste_erreurs[3,],liste_erreurs[1,],label='validation')
    p3,=plt.plot(liste_erreurs[3,],liste_erreurs[2,],label='test')
    plt . xlabel ('degree', fontsize =20)
    plt . ylabel ('R²', fontsize =20)
    plt . title ('R² as a function of the degree',fontsize =16)
    plt . legend ( handles =[p1 , p2, p3],fontsize =16)
    plt.show()


    
    

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

    
def model_Polynomiale_learning(degree,X_train,y_train):
    #entrainement du modèle
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
    y_train_predict = model_Polynomiale(degree).predict(X_train_poly)
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

def model_Polynomiale(degree):
    #entrainement du modèle
    polynomial_features= PolynomialFeatures(degree=degree)
    X_train_poly=polynomial_features.fit_transform(X_train)    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    return model

   
    


def polyfinal_learning_under0(degree=6,X_train=X_train,y_train=y_train):
    small= model_small_premium_learning(degree,X_train,y_train)
    ##big=model_big_premium(degree)
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
        ##On ramène finalement les valeurs à 0 à 10
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
        ##On ramène finalement les valeurs à 0 à 10
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
        ##On ramène finalement les valeurs à 0 à 10
        if (y_predictionvalid[i]<0):    
            y_predictionvalid[i]=10

    ##calculons les r2
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
#         if ((np.abs(r2test[i-100]-r2train[i-100]))<10**-2) and r2test[i-100]>0.95:
#             print(r2test[i-100]-r2train[i-100])
#             return r2train[i-100],r2test[i-100], r2valid[i-100], listei[i-100]       

     p1,=plt.plot(listei,r2train,label='train')
     p2,=plt.plot(listei,r2test,label='test')
     p3,=plt.plot(listei,r2valid,label='valid')
     plt . xlabel ('training set size', fontsize =20)
     plt . ylabel ('R²', fontsize =20)
     plt . title ('learning curve',fontsize =16)
     plt . legend ( handles =[p1 , p2,p3],fontsize =16)
     plt.show()    
     return r2train,r2test, r2valid,listei      

def affichage_polyfinal_under0(degremax):
##On va afficher le R**2 des données test, données de validation et données train en fonction du degré    
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
    plt.show()

#affichage_polyfinal_under0(20)
#
#y_test_predict=polyfinal_learning_under0(6)[0]
#y_valid_predict=polyfinal_learning_under0(6)[4]
#
#
#with open("under0predict.csv", "w") as f:
#    w = csv.writer(f)
#    w.writerows(zip(y_test,y_test_predict,y_valid,y_valid_predict))
#
#polyfinal_learning_under0(6)[1]
#polyfinal_learning_under0(6)[2]
#polyfinal_learning_under0(6)[3]
#rmse = (np.sqrt(mean_squared_error(y_valid_predict, y_valid)))
#rmse = (np.sqrt(mean_squared_error(y_test_predict, y_test)))
#


# use automatically configured elastic net algorithm

#from sklearn.linear_model import ElasticNetCV
#from sklearn.model_selection import RepeatedKFold
#
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#ratios = np.arange(0, 1, 0.01)
#alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
#model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)
## fit model
#model.fit(X_test, y_test)
## summarize chosen configuration
#print('alpha: %f' % model.alpha_)
#print('l1_ratio_: %f' % model.l1_ratio_)
#

x = 50
i = 1/100
n = 20
m = 10
a = 1000
degree=6


def term_insurance_predicted(x,m,n,i,a,degree):
    small= model_small_premium_learning(degree,X_train,y_train)
    big=model_Polynomiale_learning(degree,X_train,y_train)
    under0=model_under0_premium_learning(degree,X_train,y_train)
    if (m>=n):
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
        final_premium=10    
    return f'{final_premium[0]:.2f}'


def term_insurance_predicted_polynomiale_no_constraint(x,m,n,i,a,degree):
    big=model_Polynomiale_learning(degree,X_train,y_train)
    if (m>=n):
        return('error')
    data=[[x,m,n,i,a]] 
    premium_to_predict=pd.DataFrame(data=data,columns=['age','nb_payements','maturity','interest_rate','amount'])
    polynomial_features=PolynomialFeatures(degree=degree)
    premium_to_predict = polynomial_features.fit_transform(premium_to_predict)
    final_premium = big.predict(premium_to_predict)  
    return f'{final_premium[0]:.2f}'   



def term_insurance_predicted_polynomiale_lasso(x,m,n,i,a,degree,alpha):
    if (m>=n):
        return('error')
    model=Polynomiale_lasso(degree,alpha)[0]    
    data=[[x,m,n,i,a]] 
    premium_to_predict=pd.DataFrame(data=data,columns=['age','nb_payements','maturity','interest_rate','amount'])
    polynomial_features=PolynomialFeatures(degree=degree)
    premium_to_predict = polynomial_features.fit_transform(premium_to_predict)
    final_premium = model.predict(premium_to_predict)  
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
    plt.show()    
    
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
