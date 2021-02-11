from .pureendowment import Pure_endowment, PE_reserves, balancesheetPE, stresstest_PE


def Pure_endowment_predicted_polynomiale_scaled(x,m,n,i,a,degree=8):
	return Pure_endowment.Pure_endowment_predicted_polynomiale_scaled(x,m,n,i,a,degree)



##Reserves
def reserves_sum(stress_MT=0,stress_interest_rates=0,adapt=True):
	PE_reserves.reserves_sum(stress_MT,stress_interest_rates,adapt)

def reserves_sum_knn(stress_MT=0,stress_interest_rates=0,adapt=True): 
	PE_reserves.reserves_sum_knn(stress_MT,stress_interest_rates,adapt)

def reserves_true(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):
	PE_reserves.reserves_true(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)

def reserves_predicted_scale_knn(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):	
	PE_reserves.reserves_predicted_scale_knn(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)



def profit_and_loss(x,m,n,i,a):
	return stresstest_PE.profit_and_loss(x,m,n,i,a)


def plot_p_and_l_point(TF,x,i,n,m,a):
	return stresstest_PE.plot_p_and_l_point(TF,x,i,n,m,a)

def plot_p_and_l_point_interest(TF,x,i,n,m,a):
	return stresstest_PE.plot_p_and_l_point_interest(TF,x,i,n,m,a)

def plot_p_and_l_point_knn(TH,x,i,n,m,a,degree=8):
	return stresstest_PE.plot_p_and_l_point_knn(TH,x,i,n,m,a,degree)

def plot_p_and_l_point_interest_knn(TH,x,i,n,m,a,degree=8):
	return stresstest_PE.plot_p_and_l_point_interest_knn(TH,x,i,n,m,a,degree)

##stress_MT =true is for the stress on mortality table and if it's false it's on the interest rates
def plot_p_and_l_point_new(x,m,n,i,a,stress_MT=True):
	return stresstest_PE.plot_p_and_l_point_new(x,m,n,i,a,stress_MT)


#balance sheet

def total_balance_sheet_true(stress_MT=0,stress_interest_rates=0, adapt=True):
	return PE_reserves.total_balance_sheet_true(stress_MT,stress_interest_rates, adapt)

def total_balance_sheet_predicted(stress_MT=0,stress_interest_rates=0, adapt=True):
	return  PE_reserves.total_balance_sheet_predicted(stress_MT,stress_interest_rates, adapt)

def balance_sheet_knn(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):
	return  PE_reserves.balance_sheet_knn(x,n,i,a,m,stress_MT,stress_interest_rates, adapt)

def balance_sheet_true(x,n,i,a,m,stress_MT=0,stress_interest_rates=0, adapt=True):
	return  PE_reserves.balance_sheet_true(x,n,i,a,m,stress_MT,stress_interest_rates, adapt) 





