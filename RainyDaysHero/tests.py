from django.test import TestCase


#termInsurance premium calculation function : TermInsuranceAnnual(x,m,n,i,a)
from .ai_maths.premiumComputation import TermInsuranceAnnual



#Term insurance premium
class TIPremiumTestCase(TestCase):
    def test_premium_values(self):
        """Premium values checked bellow"""
        self.assertEqual(TermInsuranceAnnual(50,5,5,1,100), '0.31') #0.66
