from .terminsurance import lr,pr,lasso,polynomial_scaled,KNN,reserves,stresstest


def term_insurance_predicted(x,m,n,i,a,degree):
    return pr.term_insurance_predicted(x,m,n,i,a,degree)

def term_insurance_predicted_polynomiale_no_constraint(x,m,n,i,a,degree):
    return lr.term_insurance_predicted_polynomiale_no_constraint(x,m,n,i,a,degree)


def term_insurance_predicted_polynomiale_lasso(x,m,n,i,a,degree,alpha=0.1):
    return lasso.term_insurance_predicted_polynomiale_lasso(x,m,n,i,a,degree)  

def term_insurance_predicted_polynomiale_scaled(x,m,n,i,a,degree=4):
    return polynomial_scaled.term_insurance_predicted_polynomiale_scaled(x,m,n,i,a,degree)

def term_insurance_predicted_knn(x,m,n,i,a,nn=8):
	return str("%.2f" % KNN.term_insurance_predicted_knn(x,m,n,i,a))

def reserves_sum_knn(stress_MT=0,stress_interest_rates=0,adapt=True):
	return reserves.reserves_sum_knn(stress_MT,stress_interest_rates,adapt)

def reserves_sum(stress_MT=0,stress_interest_rates=0,adapt=True):
	return reserves.reserves_sum(stress_MT,stress_interest_rates,adapt)

def reserves_predicted(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):
	return reserves.reserves_predicted(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True)	

def lx_evolution(stress):
	return stresstest.lx_evolution(stress)
def qx_evolution(stress):
	return stresstest.qx_evolution(stress)


def plot_p_and_l_sum():
	return stresstest.plot_p_and_l_sum()
def plot_p_and_l_sum_interest():
	return stresstest.plot_p_and_l_sum_interest()

def profit_and_loss_knn():
	return strestest.profit_and_loss_knn()

def plot_p_and_l_knn_sum_stress(nn=10):
	return stresstest. plot_p_and_l_knn_sum_stress(nn)
def plot_p_and_l_sum_interest_knn(nn=10):
	return stresstest.plot_p_and_l_sum_interest_knn(nn)

def plot_p_and_l_point(TH,x,i,n,m,a):
	return stresstest.plot_p_and_l_point(TH,x,i,n,m,a)
def plot_p_and_l_point_interest(TH,x,i,n,m,a):
	return stresstest.plot_p_and_l_point_interest(TH,x,i,n,m,a)

def plot_p_and_l_point_knn(TH,x,i,n,m,a,nn=10):
	return stresstest.plot_p_and_l_point_knn(TH,x,i,n,m,a,nn)
def plot_p_and_l_point_interest_knn(TH,x,i,n,m,a,nn=10):
	return stresstest.plot_p_and_l_point_interest_knn(TH,x,i,n,m,a,nn)

def new_p_and_l_sum(stress=0,stress_interest=0):
	return stresstest.new_p_and_l_sum(stress,stress_interest)
def new_p_and_l_point(x,m,n,i,a,stress=0,stress_interest=0):
	return stresstest.new_p_and_l_point(x,m,n,i,a,stress,stress_interest)

def plot_p_and_l_knn_sum_stress_new():
	return stresstest.plot_p_and_l_knn_sum_stress_new()
def plot_p_and_l_knn_sum_stress_interest_new():
	return stresstest.plot_p_and_l_knn_sum_stress_interest_new()

def plot_p_and_l_point_new(x,m,n,i,a,stress_MT=True):
	return stresstest.plot_p_and_l_point_new(x,m,n,i,a,stress_MT)







