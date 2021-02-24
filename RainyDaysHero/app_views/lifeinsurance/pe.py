from django.http import HttpResponse
from django.http import FileResponse
from django.template import loader
import json

import os
from random import choice
from string import ascii_uppercase

## forms
from RainyDaysHero.app_forms.pureEndowmentForm import PureEndowmentForm,PureEndowmentReserveForm,PureEndowmentStressForm,PureEndowmentBSForm

## libs for IA - load trained models
from joblib import dump, load #load - save models
from sklearn.preprocessing import PolynomialFeatures

## Our IA module & actuarial formules
from RainyDaysHero.ai_maths import pureEndowmentModels
from RainyDaysHero.ai_maths import premiumComputation

TH = [100000,99511,99473,99446,99424,99406,99390,99376,99363,99350,99338,99325,99312,99296,99276,99250,99213,99163,99097,99015,98921,98820,98716,98612,98509,98406,98303,98198,98091,97982,97870,97756,97639,97517,97388,97249,97100,96939,96765,96576,96369,96141,95887,95606,95295,94952,94575,94164,93720,93244,92736,92196,91621,91009,90358,89665,88929,88151,87329,86460,85538,84558,83514,82399,81206,79926,78552,77078,75501,73816,72019,70105,68070,65914,63637,61239,58718,56072,53303,50411,47390,44234,40946,37546,34072,30575,27104,23707,20435,17338,14464,11852,9526,7498,5769,4331,3166,2249,1549,1032,663,410,244,139,75,39,19,9,4,2,1]

import after_response
from fpdf import FPDF
'''Remove a temp file after request'''
@after_response.enable
def removequotationfile(filename):
    os.remove(filename)


def PureEndowment(request):
    context = {}
    if request.method == 'GET':
        template = loader.get_template('RainyDaysHero/life-insurance/PE/pe.html')
        form = PureEndowmentForm(request.POST)
        context = dict(form= form)
        return HttpResponse(template.render(context, request))
    else:
        template = loader.get_template('RainyDaysHero/life-insurance/PE/peAnswer.html')
        form = PureEndowmentForm(request.POST)
        if form.is_valid():
            context['form'] = form
            #compute price
            x,m,n,i,a,mdl = form['clientAge'].value(),form['numberOfPayements'].value(),form['maturity'].value(),form['interestRate'].value(),form['amount'].value(),form['model'].value()
            premium = predictPEPremiumLive(x,n,m,i,a,mdl)
            context['price'] = premium
            context['actuarial_price'] = premiumComputation.PEAnnual(int(x),int(m),int(n),float(i)/100,float(a))

            if form['printPDF'].value()=='Yes':
                pdf = FPDF(orientation='P', unit='mm', format='A4')
                pdf.add_page()
                pdf.rect(5.0, 5.0, 200.0, 287.0)
                pdf.rect(8.0, 8.0, 194.0, 282.0)

                # App pic
                # pdf.image(static('RainyDaysHero/images/rdh.png'), 10, 8, 33)
                dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                pdf.image(os.path.join(dirname, 'static/RainyDaysHero/images/rdh.png'), 10, 8, 33)
                pdf.set_font('Arial', 'B', 15)

                # Client Name
                pdf.cell(140)
                pdf.cell(0, 5, str(form['clientName'].value()), ln=1)

                # Company name
                pdf.ln(25)
                pdf.cell(0, 5, 'Rainy Days Hero', ln=1)

                # Informations
                pdf.set_text_color(238, 58, 20)
                pdf.ln(6)
                pdf.cell(60)
                pdf.cell(65, 10, 'Life Insurance quotation', 'B', ln=2)
                pdf.set_text_color(0, 0, 0)
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(65, 10, "Product: Pure Endowment", ln=2)
                pdf.cell(65, 10, "Age: " + str(form['clientAge'].value()), ln=2)
                pdf.cell(65, 10, "Maturity: " + str(form['maturity'].value()), ln=2)
                pdf.cell(65, 10, "Number of payements: " + str(form['numberOfPayements'].value()), ln=2)
                pdf.cell(65, 10, "Interest rate: " + form['interestRate'].value()+"%", ln=2)
                pdf.cell(65, 10, "Amount: " + form['amount'].value(), ln=2)

                pdf.set_text_color(39, 174, 96)
                pdf.ln(25)
                pdf.cell(60)
                pdf.set_font('Arial', 'B', 15)
                pdf.cell(65, 10, "Premium: " + premium + " " + chr(128), ln=2)

                # save file and del after response

                quotationPDFFile = ''.join(choice(ascii_uppercase) for i in range(100)) + '.pdf'
                pdf.output(quotationPDFFile, 'F')
                response = HttpResponse(pdf, content_type='application/pdf')
                response['Content-Disposition'] = "attachment; filename=quotationPDFFile"
                removequotationfile.after_response(quotationPDFFile)

                return FileResponse(open(quotationPDFFile, "rb"), as_attachment=True, filename='quotation.pdf')
            else:
                return HttpResponse(template.render(context, request))



def PureEndowmentAnalysis(request):
    context = {}
    if request.method == 'GET':
        template = loader.get_template('RainyDaysHero/life-insurance/PE/peAnalysis.html')
        form = PureEndowmentForm(request.POST)
        context = dict(form= form)
        return HttpResponse(template.render(context, request))
    else:
        pass


def PureEndowmentReserve(request):
    context = {}
    if request.method == 'GET':
        template = loader.get_template('RainyDaysHero/life-insurance/PE/peReserve.html')
        form = PureEndowmentReserveForm(request.POST)
        context = dict(form= form)
        context['requestType']='GET'
        context['a']=list()
        context['b']=list()
        context['c']=list()
        return HttpResponse(template.render(context, request))
    else:
        form = PureEndowmentReserveForm(request.POST)
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
                reserveResponse=pureEndowmentModels.reserves_predicted_scale_knn(x,n,i,a,m,mortalityStress,interestRateStress, adaptedModel)
                context['a']=json.dumps(list(reserveResponse[0]))
                context['b']=json.dumps(list(reserveResponse[1]))
                context['c']=json.dumps(list(reserveResponse[1]))
            elif IAorActuarial=='Actuarial':
                reserveResponse=pureEndowmentModels.reserves_true(x,n,i,a,m,mortalityStress,interestRateStress,adaptedModel)
                context['a']=json.dumps(list(reserveResponse[0]))
                context['b']=json.dumps(list(reserveResponse[1]))
                context['c']=json.dumps(list(reserveResponse[1]))
            else:
                reserveResponseIA=pureEndowmentModels.reserves_predicted_scale_knn(x,n,i,a,m,mortalityStress,interestRateStress,adaptedModel)
                reserveResponseActuarial=pureEndowmentModels.reserves_true(x,n,i,a,m,mortalityStress,interestRateStress,adaptedModel)
                context['a']=json.dumps(list(reserveResponseIA[0]))
                context['b']=json.dumps(list(reserveResponseIA[1]))
                context['c']=json.dumps(list(reserveResponseActuarial[1]))
        else:
            mortalityStress=float(form['mortalityStress'].value())/100
            interestRateStress=float(form['interestRateStress'].value())/100
            adaptedModel=form['adaptedModel'].value()=='Yes'

            if IAorActuarial=='IA':
                reserveResponseIA=pureEndowmentModels.reserves_sum_knn(mortalityStress,interestRateStress,adaptedModel)
                context['a']=json.dumps(list(reserveResponseIA[0]))
                context['b']=json.dumps(list(reserveResponseIA[1]))
                context['c']=json.dumps(list(reserveResponseIA[1]))
            elif IAorActuarial=='Actuarial':
                reserveResponseActuarial=pureEndowmentModels.reserves_sum(mortalityStress,interestRateStress,adaptedModel)
                context['a']=json.dumps(list(reserveResponseActuarial[0]))
                context['b']=json.dumps(list(reserveResponseActuarial[1]))
                context['c']=json.dumps(list(reserveResponseActuarial[1]))
            else:
                reserveResponseIA=pureEndowmentModels.reserves_sum_knn(mortalityStress,interestRateStress,adaptedModel)
                reserveResponseActuarial=pureEndowmentModels.reserves_sum(mortalityStress,interestRateStress,adaptedModel)
                context['a']=json.dumps(list(reserveResponseIA[0]))
                context['b']=json.dumps(list(reserveResponseIA[1]))
                context['c']=json.dumps(list(reserveResponseActuarial[1]))
        template = loader.get_template('RainyDaysHero/life-insurance/PE/peReserve.html')
        return HttpResponse(template.render(context, request))

def PureEndowmentAccounting(request):
    context = {}
    if request.method == 'GET':
        template = loader.get_template('RainyDaysHero/life-insurance/PE/peAccounting.html')
        form = PureEndowmentBSForm(request.POST)
        context = dict(form= form)
        return HttpResponse(template.render(context, request))
    else:
        template = loader.get_template('RainyDaysHero/life-insurance/PE/peAccountingResult.html')
        form = PureEndowmentBSForm(request.POST)
        context = dict(form= form)
        accountingType = form['contractOrTotal'].value()
        IAOrActuarial = form['IAorActuarial'].value()
        adaptedModel = form['adaptedModel'].value() == 'Yes'
        mortalityStress = float(form['mortalityStress'].value()) / 100
        interestRateStress = float(form['interestRateStress'].value()) / 100

        if accountingType == 'Contract':
            x,m,n,i,a = int(form['clientAge'].value()),int(form['numberOfPayements'].value()),int(form['maturity'].value()),float(form['interestRate'].value())/100,float(form['amount'].value())
            if IAOrActuarial=='IA':
                result = pureEndowmentModels.balance_sheet_knn(x,n,i,a,m,mortalityStress,interestRateStress,adaptedModel)
            else:
                result = pureEndowmentModels.balance_sheet_true(x,n,i,a,m,mortalityStress,interestRateStress,adaptedModel)
        else:
            if IAOrActuarial=='IA':
                result = pureEndowmentModels.total_balance_sheet_predicted(mortalityStress,interestRateStress,adaptedModel)
            else:
                result = pureEndowmentModels.total_balance_sheet_true(mortalityStress,interestRateStress,adaptedModel)
        res = []
        for x in result:
            res.append(list(x))
            res[-1][-1] = int(res[-1][-1])
            for i in range(len(res[-1])-1):
                res[-1][i]=f'{res[-1][i]:.2f}'
        context['years'] = res
        return HttpResponse(template.render(context, request))

def PureEndowmentStress(request):
    context = {}
    if request.method == 'GET':
        template = loader.get_template('RainyDaysHero/life-insurance/PE/peStresstest.html')
        form = PureEndowmentStressForm(request.POST)
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
        template = loader.get_template('RainyDaysHero/life-insurance/PE/peStresstest.html')
        form = PureEndowmentStressForm(request.POST)
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

                res = pureEndowmentModels.plot_p_and_l_point_knn(TH, x, i, n, m, a)
                context['a'] = json.dumps(list(res[0]))
                context['b'] = json.dumps(list(map(float,list(res[1]))))
                res = pureEndowmentModels.plot_p_and_l_point_new(x, m, n, i, a)
                context['c'] = json.dumps(list(map(float,list(res[1]))))
                res = pureEndowmentModels.plot_p_and_l_point(TH, x, i, n, m, a)
                context['d'] = json.dumps(list(map(float,list(res[1]))))
            else:
                context['plotNumber'] = '1'
                context['labelOne'] = stressType
                if stressType=='Non Adapted IA':
                    res = pureEndowmentModels.plot_p_and_l_point_knn(TH,x,i,n,m,a)
                    context['a'] = json.dumps(list(res[0]))
                    context['b'] = json.dumps(list(map(float,list(res[1]))))
                if stressType=='Adapted IA':
                    res = pureEndowmentModels.plot_p_and_l_point_new(x,m,n,i,a)
                    context['a'] = json.dumps(list(res[0]))
                    context['b'] = json.dumps(list(map(float,list(res[1]))))
                if stressType=='Actuarial':
                    res = pureEndowmentModels.plot_p_and_l_point(TH,x,i,n,m,a)
                    context['a'] = json.dumps(list(res[0]))
                    context['b'] = json.dumps(list(map(float,list(res[1]))))

        else:
            if stressType == 'All':
                context['requestType']='POST'
                context['plotNumber']='3'
                context['labelOne'] = 'IA non adapted'
                context['labelTwo'] = 'IA adapted'
                context['labelThree'] = 'Actuarial'

                res = pureEndowmentModels.plot_p_and_l_point_interest_knn(TH,x,i,n,m,a)
                context['a'] = json.dumps(list(res[0]))
                context['b'] = json.dumps(list(map(float,list(res[1]))))
                res = pureEndowmentModels.plot_p_and_l_point_new(x, m, n, i, a, False)
                context['c'] = json.dumps(list(map(float,list(res[1]))))
                res = pureEndowmentModels.plot_p_and_l_point_interest(TH,x,i,n,m,a)
                context['d'] = json.dumps(list(map(float,list(res[1]))))
            else:
                context['plotNumber'] = '1'
                context['labelOne'] = stressType
                if stressType == 'Non Adapted IA':
                    res = pureEndowmentModels.plot_p_and_l_point_interest_knn(TH,x,i,n,m,a)
                    context['a'] = json.dumps(list(res[0]))
                    context['b'] = json.dumps(list(map(float,list(res[1]))))
                if stressType == 'Adapted IA':
                    res = pureEndowmentModels.plot_p_and_l_point_new(x, m, n, i, a, False)
                    context['a'] = json.dumps(list(res[0]))
                    context['b'] = json.dumps(list(map(float,list(res[1]))))
                if stressType == 'Actuarial':
                    res = pureEndowmentModels.plot_p_and_l_point_interest(TH,x,i,n,m,a)
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
def predictPEPremiumLive(x,n,m,i,a,mdl):
    x,n,m,i,a,mdl = int(x),int(n),int(m),float(i)/100,float(a),mdl
    if mdl=='ps':
        return pureEndowmentModels.Pure_endowment_predicted_polynomial_scaled(x,m,n,i,a)
