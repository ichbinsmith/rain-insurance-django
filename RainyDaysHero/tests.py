from django.test import TestCase


#termInsurance premium calculation function : TermInsuranceAnnual(x,m,n,i,a)
from .ai_maths.premiumComputation import TermInsuranceAnnual, TermInsuranceAnnualToTest
from .ai_maths.terminsurance.reserves import reserves_true


#Term insurance premium
class TIPremiumTestCase(TestCase):
    def test_premium_values(self):
        """Premium values checked bellow"""
        self.assertEqual(TermInsuranceAnnual(50,5,5,0.01,100), '0.66')
        self.assertEqual(TermInsuranceAnnualToTest(50,5,5,0.01,100), '0.663093902')
        self.assertEqual(TermInsuranceAnnual(50,5,5,0.01,100), '0.66')
        self.assertEqual(TermInsuranceAnnual(50,5,10,0.01,4000), '60.44')
        self.assertEqual(TermInsuranceAnnual(30,8,14,0.03,1000), '2.84')
        self.assertEqual(TermInsuranceAnnual(20,20,21,0.015,3000), '4.07')
        self.assertEqual(TermInsuranceAnnual(40,2,5,0.02,500), '3.49')
        self.assertEqual(TermInsuranceAnnual(60,35,40,0.01,2000),'86.59')
        self.assertEqual(TermInsuranceAnnual(45,7,20,0.026,8000),'144.91')
        
class TIReserveTestCase(TestCase):
    def test_reserve_values(self):
        """Reserve values checked bellow""" 
        result = [0.000000000,  0.087938770,    0.135718124,    0.139764597,    0.096261788,    0.000000000]
        computed = reserves_true(50,5,0.01,100,5)[1]
        for i in range(5):
            self.assertEqual(f'{computed[i]:.9f}', f'{result[i]:.9f}')
