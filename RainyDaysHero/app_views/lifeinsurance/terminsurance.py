from django.shortcuts import render
from django.http import Http404
from django.http import HttpResponse
from django.template import loader
from django.http import HttpResponseRedirect
from django.http import FileResponse
from django.shortcuts import render

import io
import os
from random import choice
from string import ascii_uppercase

import after_response
from django.templatetags.static import static

## forms
from RainyDaysHero.app_forms.termInsuranceForm import TermInsuranceForm

## libs for IA - load trained models
from joblib import dump, load #load - save models
from sklearn.preprocessing import PolynomialFeatures


## Our IA module & actuarial formules
from RainyDaysHero.ai_maths import termInsuranceModels
from RainyDaysHero.ai_maths import premiumComputation



def terminsurance(request):
    context = {}
    if request.method == 'GET':
        template = loader.get_template('RainyDaysHero/terminsurance.html')
        form = TermInsuranceForm(request.POST)
        context = dict(form= form)
        return HttpResponse(template.render(context, request))
    else:
        template = loader.get_template('RainyDaysHero/terminsuranceAnswer.html')
        form = TermInsuranceForm(request.POST)
        if form.is_valid():
            context['form'] = form
            #compute price
            x,m,n,i,a,mdl = form['clientAge'].value(),form['numberOfPayements'].value(),form['maturity'].value(),form['interestRate'].value(),form['amount'].value(),form['model'].value()
            premium = predictTIPremiumLive(x,n,m,i,a,mdl)
            context['price'] = premium
            context['actuarial_price'] = premiumComputation.TermInsuranceAnnual(int(x),int(m),int(n),float(i)/100,float(a))
            return HttpResponse(template.render(context, request))


def terminsuranceAnalysis(request):
    context = {}
    if request.method == 'GET':
        template = loader.get_template('RainyDaysHero/terminsuranceAnalysis.html')
        form = TermInsuranceForm(request.POST)
        context = dict(form= form)
        return HttpResponse(template.render(context, request))
    else:
        pass

'''
Prediction using saved models
'''
def predictTIPremium(x,n,m,i,a,mdl):
    x,n,m,i,a,mdl = int(x),int(n),int(m),float(i)/100,float(a),mdl
    if mdl=='lr':
        polynomial_features = PolynomialFeatures(degree=1)
        var = polynomial_features.fit_transform([(x,m,n,i,a)])
        model = load(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'static/RainyDaysHero/data/LI/TI/models/'+mdl+'.joblib'))
        return model.predict(var)[0]
    elif mdl=='plr':
        polynomial_features = PolynomialFeatures(degree=6)
        var = polynomial_features.fit_transform([(x,m,n,i,a)])
        model = load(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'static/RainyDaysHero/data/LI/TI/models/'+mdl+'.joblib'))
        return model.predict(var)[0]

'''
'Live' Prediction
'''

def predictTIPremiumLive(x,n,m,i,a,mdl):
    x,n,m,i,a,mdl = int(x),int(n),int(m),float(i)/100,float(a),mdl
    if mdl=='lr':
        return termInsuranceModels.term_insurance_predicted_polynomiale_no_constraint(x,m,n,i,a,1)
    elif mdl=='plr':
        return termInsuranceModels.term_insurance_predicted(x,m,n,i,a,6)
    elif mdl=='lasso':
        return termInsuranceModels.term_insurance_predicted_polynomiale_lasso(x,m,n,i,a,6)
    elif mdl=='ps':
        return termInsuranceModels.term_insurance_predicted_polynomiale_scaled(x,m,n,i,a,4)

