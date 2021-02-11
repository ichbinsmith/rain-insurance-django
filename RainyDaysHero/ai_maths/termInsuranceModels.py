import numpy as np
from .terminsurance import lr,pr,lasso,polynomial_scaled,KNN,reserves,stresstest,balance_sheet
TH = [100000,99511,99473,99446,99424,99406,99390,99376,99363,99350,99338,99325,99312,99296,99276,99250,99213,99163,99097,99015,98921,98820,98716,98612,98509,98406,98303,98198,98091,97982,97870,97756,97639,97517,97388,97249,97100,96939,96765,96576,96369,96141,95887,95606,95295,94952,94575,94164,93720,93244,92736,92196,91621,91009,90358,89665,88929,88151,87329,86460,85538,84558,83514,82399,81206,79926,78552,77078,75501,73816,72019,70105,68070,65914,63637,61239,58718,56072,53303,50411,47390,44234,40946,37546,34072,30575,27104,23707,20435,17338,14464,11852,9526,7498,5769,4331,3166,2249,1549,1032,663,410,244,139,75,39,19,9,4,2,1]


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



def reserves_true(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):
	return reserves.reserves_true(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)

def reserves_predicted_scale_knn(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):
	return reserves.reserves_predicted_scale_knn(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)


def lx_evolution(stress):
	return stresstest.lx_evolution(stress)
def qx_evolution(stress):
	return stresstest.qx_evolution(stress)


def profit_and_loss(x,m,n,i,a):
	return stresstest.profit_and_loss(x,m,n,i,a)

def plot_p_and_l_point(TH,x,i,n,m,a):
	return stresstest.plot_p_and_l_point(TH,x,i,n,m,a)

def plot_p_and_l_point_interest(TH,x,i,n,m,a):
	return stresstest.plot_p_and_l_point_interest(TH,x,i,n,m,a)

def plot_p_and_l_point_knn(TH,x,i,n,m,a,degree=8):
	return stresstest.plot_p_and_l_point_knn(TH,x,i,n,m,a,degree)

def plot_p_and_l_point_interest_knn(TH,x,i,n,m,a,degree=8):
	return stresstest.plot_p_and_l_point_interest_knn(TH,x,i,n,m,a,degree)

##stress_MT =true is for the stress on mortality table and if it's false it's on the interest rates
def plot_p_and_l_point_new(x,m,n,i,a,stress_MT=True):
	return stresstest.plot_p_and_l_point_new(x,m,n,i,a,stress_MT)


## Balance sheet#

def total_balance_sheet_true(stress_MT=0,stress_interest_rates=0, adapt=True):
	return balance_sheet.total_balance_sheet_true(stress_MT,stress_interest_rates, adapt)

def total_balance_sheet_predicted(stress_MT=0,stress_interest_rates=0, adapt=True):
	return balance_sheet.total_balance_sheet_predicted(stress_MT,stress_interest_rates, adapt)


def balance_sheet_knn(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):
	test= balance_sheet.balance_sheet_knn(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)
	listcontract=listcontract=np.zeros((m,8))
	for term in range(0,n):
		for smash in range (0,7):
			listcontract[term][smash]=test[smash][term]
		listcontract[term][7]=int(term+1)
	return(listcontract)

def balance_sheet_true(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):
	test= balance_sheet.balance_sheet_true(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)

	listcontract=listcontract=np.zeros((m,8))
	for term in range(0,n):
		for smash in range (0,7):
			listcontract[term][smash]=test[smash][term]
		listcontract[term][7]=int(term+1)
	return(listcontract)
