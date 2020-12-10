import os
import sys
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
from joblib import dump, load


df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"\\dataset.csv")

X = df[['age','nb_payements','maturity','interest_rate','amount']]

y = df['target']


##analyse des données

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


##Regression lineaire

lr = LinearRegression()
lr.fit(X_train, y_train)
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_valid, y_valid)))

dump(lr, os.path.dirname(os.path.realpath(__file__))+'\\Model\\regression_model.joblib')


##La régression linéaire n'étant pas très efficace, on passe à la regression de ridge

##On baisse la valeur d'alpha car on était en sous apprentissage
ridge= Ridge(alpha=10**-7).fit(X_train, y_train)
print("Ridge :")
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_valid, y_valid)))

dump(ridge, os.path.dirname(os.path.realpath(__file__))+'\\Model\\ridge_model.joblib')

##On est toujours trop faible
plt.plot(ridge.coef_, 'v', label = "Ridge alpha =10**-7")

##Testons Lasso

lasso = Lasso().fit(X_train, y_train)
print("Lasso")
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_valid, y_valid)))
print("Number of features used: {}".format(np.sum(lasso.coef_ !=0)))

dump(lasso, os.path.dirname(os.path.realpath(__file__))+'\\Model\\lasso_model.joblib')



def sweet_spot_lasso():
    print("Lasso sweet spot")
    z = [10**(i) for i in range(-10,1)]
    r = []
    t = []
    for v in z:
        lasso = Lasso(alpha=v).fit(X_train, y_train)
        r.append(lasso.score(X_train, y_train))
        t.append(lasso.score(X_valid, y_valid))
    plt.plot(z[::-1],r[::-1])
    plt.plot(z[::-1],t[::-1])
    plt.show()

#sweet_spot_lasso()

##On ne semble pas capable de dépasser les 30%     
    
##Testons les Support Vector Machine

regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1))
regr.fit(X_train, y_train)
predictions=regr.predict(X_valid)
np.linalg.norm(predictions-y_valid,ord=1)
regr.score(X_valid, y_valid, sample_weight=None)
regr.score(X_train, y_train, sample_weight=None)
##Le R**2 est très très bas



##Testons la regression polynomiale puisque la linéaire était trop faible

 
def Polynomiale(degree):
    #entrainement du modèle
    polynomial_features= PolynomialFeatures(degree=degree)
    X_train_poly = polynomial_features.fit_transform(X_train)
    X_valid_poly = polynomial_features.fit_transform(X_valid)
    X_trainval_poly=polynomial_features.fit_transform(X_trainval)
    X_test_poly=polynomial_features.fit_transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    # Evaluation du training set
 
    y_train_predict = model.predict(X_train_poly)
    rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
    r2train = r2_score(y_train, y_train_predict)
 
    print('La performance du modèle sur la base dapprentissage')
    print('--------------------------------------')
    print("L'erreur quadratique moyenne est {}".format(rmse))
    print('le score R2 est {}'.format(r2train))
    print('\n')
 
    # Evaluation sur le validation set
    y_valid_predict = model.predict(X_valid_poly)
    rmse = (np.sqrt(mean_squared_error(y_valid, y_valid_predict)))
    r2valid = r2_score(y_valid, y_valid_predict)
 
    print('La performance du modèle sur la base de validation')
    print('--------------------------------------')
    print("L'erreur quadratique moyenne est {}".format(rmse))
    print('le score R2 est {}'.format(r2valid))
    print('\n')
    
    # valuation sur le test set
    model = LinearRegression()
    model.fit(X_trainval_poly, y_trainval)
    y_test_predict = model.predict(X_test_poly)
    rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
    r2test = r2_score(y_test, y_test_predict)
 
    print('La performance du modèle sur la base de test')
    print('--------------------------------------')
    print("L'erreur quadratique moyenne est {}".format(rmse))
    print('le score R2 est {}'.format(r2test))
    print('\n')
    return r2train,r2valid,r2test

##Nous dépassons enfin les 90% pour le test et la validation

##On va afficher le R**2 des données test, données de validation et données train en fonction du degré    
liste_erreurs=np.zeros((4,10))
for i in range(1,10):
    liste_erreurs[0,i]=(Polynomiale(i)[0])
    liste_erreurs[1,i]=(Polynomiale(i)[1])
    liste_erreurs[2,i]=(Polynomiale(i)[2])
    liste_erreurs[3,i]=i

p1,=plt.plot(liste_erreurs[3,],liste_erreurs[0,],label='train')
p2,=plt.plot(liste_erreurs[3,],liste_erreurs[1,],label='validation')
p3,=plt.plot(liste_erreurs[3,],liste_erreurs[2,],label='test')
plt . xlabel ('degre', fontsize =20)
plt . ylabel ('R²', fontsize =20)
plt . title ('R² en fonction du degre',fontsize =16)
plt . legend ( handles =[p1 , p2, p3],fontsize =16)
plt.show()


##Valeur négative de R² en degre=8??? A vérifier
##Le sweet spot est atteint en degre=6




















