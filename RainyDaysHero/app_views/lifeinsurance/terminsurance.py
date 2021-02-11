
from django.http import HttpResponse
from django.template import loader
import json

import os

## forms
from RainyDaysHero.app_forms.termInsuranceForm import TermInsuranceForm,TermInsuranceReserveForm,TermInsuranceStressForm, TermInsuranceBSForm

## libs for IA - load trained models
from joblib import dump, load #load - save models
from sklearn.preprocessing import PolynomialFeatures


## Our IA module & actuarial formules
from RainyDaysHero.ai_maths import termInsuranceModels
from RainyDaysHero.ai_maths import premiumComputation

TH = [100000,99511,99473,99446,99424,99406,99390,99376,99363,99350,99338,99325,99312,99296,99276,99250,99213,99163,99097,99015,98921,98820,98716,98612,98509,98406,98303,98198,98091,97982,97870,97756,97639,97517,97388,97249,97100,96939,96765,96576,96369,96141,95887,95606,95295,94952,94575,94164,93720,93244,92736,92196,91621,91009,90358,89665,88929,88151,87329,86460,85538,84558,83514,82399,81206,79926,78552,77078,75501,73816,72019,70105,68070,65914,63637,61239,58718,56072,53303,50411,47390,44234,40946,37546,34072,30575,27104,23707,20435,17338,14464,11852,9526,7498,5769,4331,3166,2249,1549,1032,663,410,244,139,75,39,19,9,4,2,1]

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
        accountingType = form['contractOrTotal'].value()
        IAOrActuarial = form['IAorActuarial'].value()
        adaptedModel = form['adaptedModel'].value() == 'Yes'
        mortalityStress = float(form['mortalityStress'].value()) / 100
        interestRateStress = float(form['interestRateStress'].value()) / 100

        if accountingType == 'Contract':
            x,m,n,i,a = int(form['clientAge'].value()),int(form['numberOfPayements'].value()),int(form['maturity'].value()),float(form['interestRate'].value())/100,float(form['amount'].value())
            if IAOrActuarial=='IA':
                result = termInsuranceModels.balance_sheet_knn(x,n,i,a,m,mortalityStress,interestRateStress,adaptedModel)
            else:
                result = termInsuranceModels.balance_sheet_true(x,n,i,a,m,mortalityStress,interestRateStress,adaptedModel)
        else:
            if IAOrActuarial=='IA':
                result = termInsuranceModels.total_balance_sheet_predicted(mortalityStress,interestRateStress,adaptedModel)
            else:
                result = termInsuranceModels.total_balance_sheet_true(mortalityStress,interestRateStress,adaptedModel)
        res = []
        for x in result:
            res.append(list(x))
            res[-1][-1] = int(res[-1][-1])
        context['years'] = res
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
        if stressOn=='Mortality Table':
            if stressType=='All':
                context['requestType']='POST'
                context['plotNumber']='3'
                context['labelOne'] = 'IA non adapted'
                context['labelTwo'] = 'IA adapted'
                context['labelThree'] = 'Actuarial'

                res = termInsuranceModels.plot_p_and_l_point_knn(TH, x, i, n, m, a)
                context['a'] = json.dumps(list(res[0]))
                context['b'] = json.dumps(list(map(float,list(res[1]))))
                res = termInsuranceModels.plot_p_and_l_point_new(x, m, n, i, a)
                context['c'] = json.dumps(list(map(float,list(res[1]))))
                res = termInsuranceModels.plot_p_and_l_point(TH, x, i, n, m, a)
                context['d'] = json.dumps(list(map(float,list(res[1]))))
            else:
                context['plotNumber'] = '1'
                context['labelOne'] = stressType
                if stressType=='Non Adapted IA':
                    res = termInsuranceModels.plot_p_and_l_point_knn(TH,x,i,n,m,a)
                    context['a'] = json.dumps(list(res[0]))
                    context['b'] = json.dumps(list(map(float,list(res[1]))))
                if stressType=='Adapted IA':
                    res = termInsuranceModels.plot_p_and_l_point_new(x,m,n,i,a)
                    context['a'] = json.dumps(list(res[0]))
                    context['b'] = json.dumps(list(map(float,list(res[1]))))
                if stressType=='Actuarial':
                    res = termInsuranceModels.plot_p_and_l_point(TH,x,i,n,m,a)
                    context['a'] = json.dumps(list(res[0]))
                    context['b'] = json.dumps(list(map(float,list(res[1]))))

        else:
            if stressType == 'All':
                context['requestType']='POST'
                context['plotNumber']='3'
                context['labelOne'] = 'IA non adapted'
                context['labelTwo'] = 'IA adapted'
                context['labelThree'] = 'Actuarial'

                res = termInsuranceModels.plot_p_and_l_point_interest_knn(TH,x,i,n,m,a)
                context['a'] = json.dumps(list(res[0]))
                context['b'] = json.dumps(list(map(float,list(res[1]))))
                res = termInsuranceModels.plot_p_and_l_point_new(x, m, n, i, a, False)
                context['c'] = json.dumps(list(map(float,list(res[1]))))
                res = termInsuranceModels.plot_p_and_l_point_interest(TH,x,i,n,m,a)
                context['d'] = json.dumps(list(map(float,list(res[1]))))
            else:
                context['plotNumber'] = '1'
                context['labelOne'] = stressType
                if stressType == 'Non Adapted IA':
                    res = termInsuranceModels.plot_p_and_l_point_interest_knn(TH,x,i,n,m,a)
                    context['a'] = json.dumps(list(res[0]))
                    context['b'] = json.dumps(list(map(float,list(res[1]))))
                if stressType == 'Adapted IA':
                    res = termInsuranceModels.plot_p_and_l_point_new(x, m, n, i, a, False)
                    context['a'] = json.dumps(list(res[0]))
                    context['b'] = json.dumps(list(map(float,list(res[1]))))
                if stressType == 'Actuarial':
                    res = termInsuranceModels.plot_p_and_l_point_interest(TH,x,i,n,m,a)
                    context['a'] = json.dumps(list(res[0]))
                    context['b'] = json.dumps(list(map(float,list(res[1]))))

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
