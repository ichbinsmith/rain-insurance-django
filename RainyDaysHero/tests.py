from django.test import TestCase


#termInsurance premium calculation function : TermInsuranceAnnual(x,m,n,i,a)
from .ai_maths.premiumComputation import TermInsuranceAnnual


#Term insurance premium
class TIPremiumTestCase(TestCase):
    def test_premium_values(self):
        """Premium values checked bellow"""
        self.assertEqual(TermInsuranceAnnual(50,5,5,0.01,100), '0.66')
        self.assertEqual(TermInsuranceAnnual(50,5,5,0.01,100), '0.66')
        self.assertEqual(TermInsuranceAnnual(50,5,10,1/100,4000), '60.44')
        self.assertEqual(TermInsuranceAnnual(30,8,14,3/100,1000), '2.84')
        self.assertEqual(TermInsuranceAnnual(20,20,21,1.5/100,3000), '4.07')
        self.assertEqual(TermInsuranceAnnual(40,2,5,2/100,500), '3.49')
        self.assertEqual(TermInsuranceAnnual(60,35,40,1/100,2000),'86.59')
        self.assertEqual(TermInsuranceAnnual(45,7,20,2.6/100,8000),'144.91')
        
