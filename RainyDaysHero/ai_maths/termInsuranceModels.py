
from .terminsurance import lr,pr,lasso,polynomial_scaled


def term_insurance_predicted(x,m,n,i,a,degree):
    return pr.term_insurance_predicted(x,m,n,i,a,degree)



def term_insurance_predicted_polynomiale_no_constraint(x,m,n,i,a,degree):
    return lr.term_insurance_predicted_polynomiale_no_constraint(x,m,n,i,a,degree)



def term_insurance_predicted_polynomiale_lasso(x,m,n,i,a,degree,alpha=0.1):
    return lasso.term_insurance_predicted_polynomiale_lasso(x,m,n,i,a,degree)  

def term_insurance_predicted_polynomiale_scaled(x,m,n,i,a,degree=4):
    return polynomial_scaled.term_insurance_predicted_polynomiale_scaled(x,m,n,i,a,degree)
