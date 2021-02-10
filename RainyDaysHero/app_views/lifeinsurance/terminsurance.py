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
from RainyDaysHero.app_forms.termInsuranceForm import TermInsuranceForm,TermInsuranceReserveForm,TermInsuranceStressForm, TermInsuranceBSForm

## libs for IA - load trained models
from joblib import dump, load #load - save models
from sklearn.preprocessing import PolynomialFeatures


## Our IA module & actuarial formules
from RainyDaysHero.ai_maths import termInsuranceModels
from RainyDaysHero.ai_maths import premiumComputation



def terminsurance(request):
    context = {}
    if request.method == 'GET':
        template = loader.get_template('RainyDaysHero/life-insurance/TI/terminsurance.html')
        form = TermInsuranceForm(request.POST)
        context = dict(form= form)
        return HttpResponse(template.render(context, request))
    else:
        template = loader.get_template('RainyDaysHero/life-insurance/TI/terminsuranceAnswer.html')
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
        template = loader.get_template('RainyDaysHero/life-insurance/TI/terminsuranceAnalysis.html')
        form = TermInsuranceForm(request.POST)
        context = dict(form= form)
        return HttpResponse(template.render(context, request))
    else:
        pass


def terminsuranceReserve(request):
    context = {}
    if request.method == 'GET':
        template = loader.get_template('RainyDaysHero/life-insurance/TI/terminsuranceReserve.html')
        form = TermInsuranceReserveForm(request.POST)
        context = dict(form= form)
        context['requestType']='GET'
        context['a']=list()
        context['b']=list()
        context['c']=list()
        return HttpResponse(template.render(context, request))
    else:
        form = TermInsuranceReserveForm(request.POST)
        context = dict(form= form)
        context['requestType']='POST'

        reserveType = form['contractOrTotal'].value()
        IAorActuarial = form['IAorActuarial'].value()
        context['IAorActuarial']=IAorActuarial

        if reserveType=='Contract':
            x,m,n,i,a = int(form['clientAge'].value()),int(form['numberOfPayements'].value()),int(form['maturity'].value()),float(form['interestRate'].value())/100,float(form['amount'].value())
            mortalityStress=float(form['mortalityStress'].value())/100
            interestRateStress=float(form['interestRateStress'].value())/100
            adaptedModel=form['adaptedModel'].value()=='Yes'
            print(x,m,n,i,a,mortalityStress,interestRateStress,adaptedModel)
            if IAorActuarial=='IA':
                reserveResponse=termInsuranceModels. reserves_predicted_scale_knn(x,n,i,a,m,mortalityStress,interestRateStress,adaptedModel)
                context['a']=json.dumps(list(reserveResponse[0]))
                context['b']=json.dumps(list(reserveResponse[1]))
                context['c']=json.dumps(list(reserveResponse[1]))
            elif IAorActuarial=='Actuarial':
                reserveResponse=termInsuranceModels.reserves_true(x,n,i,a,m,mortalityStress,interestRateStress,adaptedModel)
                context['a']=json.dumps(list(reserveResponse[0]))
                context['b']=json.dumps(list(reserveResponse[1]))
                context['c']=json.dumps(list(reserveResponse[1]))
            else:
                reserveResponseIA=termInsuranceModels.reserves_predicted_scale_knn(x,n,i,a,m,mortalityStress,interestRateStress,adaptedModel)
                reserveResponseActuarial=termInsuranceModels.reserves_true(x,n,i,a,m,mortalityStress,interestRateStress,adaptedModel)               
                context['a']=json.dumps(list(reserveResponseIA[0]))
                context['b']=json.dumps(list(reserveResponseIA[1]))
                context['c']=json.dumps(list(reserveResponseActuarial[1]))            
        else:
            mortalityStress=float(form['mortalityStress'].value())/100
            interestRateStress=float(form['interestRateStress'].value())/100
            adaptedModel=form['adaptedModel'].value()=='Yes'
            print(mortalityStress,interestRateStress,adaptedModel)

            if IAorActuarial=='IA':
                reserveResponseIA=termInsuranceModels.reserves_sum_knn(mortalityStress,interestRateStress,adaptedModel)
                context['a']=json.dumps(list(reserveResponseIA[0]))
                context['b']=json.dumps(list(reserveResponseIA[1]))
                context['c']=json.dumps(list(reserveResponseIA[1]))
            elif IAorActuarial=='Actuarial':
                reserveResponseActuarial=termInsuranceModels.reserves_sum(mortalityStress,interestRateStress,adaptedModel)
                context['a']=json.dumps(list(reserveResponseActuarial[0]))
                context['b']=json.dumps(list(reserveResponseActuarial[1]))
                context['c']=json.dumps(list(reserveResponseActuarial[1]))
            else:
                reserveResponseIA=termInsuranceModels.reserves_sum_knn(mortalityStress,interestRateStress,adaptedModel)
                reserveResponseActuarial=termInsuranceModels.reserves_sum(mortalityStress,interestRateStress,adaptedModel)
                context['a']=json.dumps(list(reserveResponseIA[0]))
                context['b']=json.dumps(list(reserveResponseIA[1]))
                context['c']=json.dumps(list(reserveResponseActuarial[1]))
        template = loader.get_template('RainyDaysHero/life-insurance/TI/terminsuranceReserve.html')
        return HttpResponse(template.render(context, request))

def terminsuranceAccounting(request):
    context = {}
    if request.method == 'GET':
        template = loader.get_template('RainyDaysHero/life-insurance/TI/terminsuranceAccounting.html')
        form = TermInsuranceBSForm(request.POST)
        context = dict(form= form)
        return HttpResponse(template.render(context, request))
    else:
        template = loader.get_template('RainyDaysHero/life-insurance/TI/terminsuranceAccountingResult.html')
        form = TermInsuranceBSForm(request.POST)
        context = dict(form= form)
        return HttpResponse(template.render(context, request))


def terminsuranceStress(request):
    context = {}
    if request.method == 'GET':
        template = loader.get_template('RainyDaysHero/life-insurance/TI/terminsuranceStresstest.html')
        form = TermInsuranceStressForm(request.POST)
        context = dict(form= form)


        context['requestType']='GET'
        context['plotNumber']='1'
        context['a'] = json.dumps(list([0,1,2,3]))
        context['b'] = json.dumps(list([4,3,2,1]))
        context['c'] = json.dumps(list([1,2,3,4]))
        context['d'] = json.dumps(list([1,2,2,1]))
        context['labelOne'] = 'Nein'
        context['labelTwo'] = 'Nein'
        context['labelThree'] = 'Nein'
        return HttpResponse(template.render(context, request))
    else:
        template = loader.get_template('RainyDaysHero/life-insurance/TI/terminsuranceStresstest.html')
        form = TermInsuranceStressForm(request.POST)
        context = dict(form= form)
        context['requestType']='POST'
        context['plotNumber']='1'
        context['a'] = json.dumps(list([0,1,2,3]))
        context['b'] = json.dumps(list([4,3,2,1]))
        context['c'] = json.dumps(list([1,2,3,4]))
        context['d'] = json.dumps(list([1,2,2,1]))
        context['labelOne'] = 'Nein'
        context['labelTwo'] = 'Nein'
        context['labelThree'] = 'Nein'

        x,m,n,i,a = int(form['clientAge'].value()),int(form['numberOfPayements'].value()),int(form['maturity'].value()),float(form['interestRate'].value())/100,float(form['amount'].value())
        stressOn=form['stressOn'].value()
        stressType=form['stressType'].value()

        if(form['stressType'].value()=='All'):
            context['requestType']='POST'
            context['a'] = json.dumps(list([0,1,2,3]))
            context['b'] = json.dumps(list([4,3,2,1]))
            context['c'] = json.dumps(list([1,2,3,4]))
            context['d'] = json.dumps(list([1,2,2,1]))
            if stressType=='All':
                context['plotNumber']='3'
                context['labelOne'] = 'IA non adapted'
                context['labelTwo'] = 'IA adapted'
                context['labelThree'] = 'Actuarial'
            else:
                context['plotNumber']='1'
                context['labelOne'] = stressType
                context['labelTwo'] = 'Nein'
                context['labelThree'] = 'Nein'

        return HttpResponse(template.render(context, request))

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
        return termInsuranceModels.term_insurance_predicted_polynomiale_scaled(x,m,n,i,a,8)
    elif mdl=='knn':
        return termInsuranceModels.term_insurance_predicted_knn(x,m,n,i,a)
