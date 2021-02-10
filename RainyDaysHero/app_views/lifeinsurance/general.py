from django.shortcuts import render
from django.http import Http404
from django.http import HttpResponse
from django.template import loader
from django.http import HttpResponseRedirect
from django.http import FileResponse
from django.shortcuts import render
import json

import io
import os
from random import choice
from string import ascii_uppercase

import after_response
from django.templatetags.static import static

## forms
from RainyDaysHero.app_forms.lifeIsuranceGeneralForms import PortfolioForm, LxQxStressForm

## libs for IA - load trained models
from joblib import dump, load #load - save models
from sklearn.preprocessing import PolynomialFeatures


## Our IA module & actuarial formules
from RainyDaysHero.ai_maths import termInsuranceModels
from RainyDaysHero.ai_maths import premiumComputation



def portfolio(request):
    context = {}
    if request.method == 'GET':
        template = loader.get_template('RainyDaysHero/life-insurance/TI/portfolio.html')
        form = PortfolioForm(request.POST)
        context = dict(form= form)
        return HttpResponse(template.render(context, request))
    else:
        template = loader.get_template('RainyDaysHero/life-insurance/TI/portfolioResult.html')
        form = PortfolioForm(request.POST)
        context = dict(form= form)
        return HttpResponse(template.render(context, request))


def lxQxStress(request):
    context = {}
    if request.method == 'GET':
        template = loader.get_template('RainyDaysHero/life-insurance/TI/lxQxStress.html')
        form = LxQxStressForm(request.POST)
        context = dict(form= form)
        context['requestType']='GET'
        context['a'] = json.dumps(list([0,1,2,3]))
        context['b'] = json.dumps(list([4,3,2,1]))
        context['c'] = json.dumps(list([1,2,3,4]))
        context['labelOne'] = 'None'
        context['labelTwo'] = 'None'
        return HttpResponse(template.render(context, request))
    else:
        template = loader.get_template('RainyDaysHero/life-insurance/TI/lxQxStress.html')
        form = LxQxStressForm(request.POST)
        if form.is_valid():
            context['form'] = form
            #compute price
            stressOn, stressOnTable= form['stressOn'].value(),form['stressOnTable'].value()

            context['a'] = json.dumps(list([0,1,2,3]))
            context['b'] = json.dumps(list([4,3,2,1]))
            context['c'] = json.dumps(list([1,2,3,4]))
            context['labelOne'] = stressOn
            context['labelTwo'] = stressOn+' Stressed'
            context['requestType'] = 'POST'
            return HttpResponse(template.render(context, request))

