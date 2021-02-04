from .terminsurance import lr,pr,lasso,polynomial_scaled,KNN,reserves,stresstest


def term_insurance_predicted(x,m,n,i,a,degree):
    return pr.term_insurance_predicted(x,m,n,i,a,degree)

def term_insurance_predicted_polynomiale_no_constraint(x,m,n,i,a,degree):
    return lr.term_insurance_predicted_polynomiale_no_constraint(x,m,n,i,a,degree)


def term_insurance_predicted_polynomiale_lasso(x,m,n,i,a,degree,alpha=0.1):
    return lasso.term_insurance_predicted_polynomiale_lasso(x,m,n,i,a,degree)  

def term_insurance_predicted_polynomiale_scaled(x,m,n,i,a,degree=8):
    return polynomial_scaled.term_insurance_predicted_polynomiale_scaled(x,m,n,i,a,degree)

def term_insurance_predicted_knn(x,m,n,i,a,nn=8):
	return str("%.2f" % KNN.term_insurance_predicted_knn(x,m,n,i,a))

def reserves_sum_knn(stress_MT=0,stress_interest_rates=0,adapt=True):
	return reserves.reserves_sum_knn(stress_MT,stress_interest_rates,adapt)

def reserves_sum(stress_MT=0,stress_interest_rates=0,adapt=True):
	return reserves.reserves_sum(stress_MT,stress_interest_rates,adapt)


## à changer

def reserves_true(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):
	return reserves.reserves_true(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)

def reserves_predicted_scale_knn(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):
	return reserves.reserves_predicted_scale_knn(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)



#Stress test

def lx_evolution(stress):
	return stresstest.lx_evolution(stress)
def qx_evolution(stress):
	return stresstest.qx_evolution(stress)



##plot le calcul actuariel stressé sur la mortalité
def plot_p_and_l_point(TH,x,i,n,m,a):
	return stresstest.plot_p_and_l_point(TH,x,i,n,m,a)

##plot le calcul actuariel stressé sur les taux d'intérêts
def plot_p_and_l_point_interest(TH,x,i,n,m,a):
	return stresstest.plot_p_and_l_point_interest(TH,x,i,n,m,a)

##plot la prédiction avec le stress sur la mortalité
def plot_p_and_l_point_knn(TH,x,i,n,m,a,nn=10):
	return stresstest.plot_p_and_l_point_knn(TH,x,i,n,m,a,nn)
##	plot la prédiction avec le stress sur les intérêtd
def plot_p_and_l_point_interest_knn(TH,x,i,n,m,a,nn=10):
	return stresstest.plot_p_and_l_point_interest_knn(TH,x,i,n,m,a,nn)

##Ici si tu mets True ça r$te prédit le modèle adapté sur la table de mortalité et False c'est sur les intérêts
def plot_p_and_l_point_new(x,m,n,i,a,stress_MT=True):
	return stresstest.plot_p_and_l_point_new(x,m,n,i,a,stress_MT)


## Balance sheet#

##Cette fonction renvoie de quoi faire le bilan pour le total en fonction des stress avec le calcul actuariel
def total_balance_sheet_true(stress_MT=0,stress_interest_rates=0, adapt=True):
	return balance_sheet.total_balance_sheet_true(stress_MT,stress_interest_rates, adapt)

##Cette fonction renvoie de quoi faire le bilan pour le total en fonction des stress avec l'IA'
def total_balance_sheet_predicted(stress_MT=0,stress_interest_rates=0, adapt=True):
	return balance_sheet.total_balance_sheet_predicted(stress_MT,stress_interest_rates, adapt)


##Cette fonction renvoie de quoi faire le bilan pour un contrat avec l'IA
def balance_sheet_knn(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):
	return balance_sheet.balance_sheet_knn(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)

##Cette fonction renvoie de quoi faire le bilan pour un contrat le calcul actuariel
def balance_sheet_true(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):
	return balance_sheet.balance_sheet_true(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)