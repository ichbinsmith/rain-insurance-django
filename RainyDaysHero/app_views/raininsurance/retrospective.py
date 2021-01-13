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

import matplotlib
matplotlib.use('Agg')

import datetime as DtT
from fpdf import FPDF
import pandas as pd
import matplotlib.pyplot as plt
import requests as REQ

## forms
from RainyDaysHero.app_forms.quotationForm import QuotationForm
from RainyDaysHero.app_forms.retroForm  import RetroForm

## libs for IA - load trained models
from joblib import dump, load #load - save models
from sklearn.preprocessing import PolynomialFeatures


## Global vars
interestRate = 0.1
cityId = {'Nice' :'181' , 'Paris':'188' , 'Nantes':'221', 'Strasbourg':'153', 'Brest' : '175', 'Ajaccio' : '179', 'Laon' : '1896', 'Calais' : '214', 'Aubusson' : '1788'}
baseUrl = 'https://www.historique-meteo.net/site/export.php?ville_id='

'''Download data'''

def dataUpdateCity(city):
    try:
        r = REQ.get(baseUrl+cityId[city], allow_redirects=True)
        if r.status_code == 200:
            print ('DATA DOWNLOADED!')
            print( r.headers.get('Content-disposition').split(';')[1].split('=')[1] )
            dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            #server-name : r.headers.get('Content-disposition').split(';')[1].split('=')[1]
            open(os.path.join(dirname, 'static/RainyDaysHero/data/'+'export-'+city), 'wb').write(r.content)
        else:
            print ('Error with request.')
    except Exception as e: #requests.exceptions.ConnectionError
        print(e)

def dataUpdateAllCity():
    for city in cityId.keys():
        dataUpdateCity(city)

'''Remove a temp file after request'''
@after_response.enable
def removequotationfile(filename):
    os.remove(filename)


def retrospective(request):
    context = {}
    if request.method == 'POST':
        template = loader.get_template('RainyDaysHero/retrospectiveAnswer-onSide.html')
        form = RetroForm(request.POST)
        if form.is_valid():
            context['form'] = form
            #compute retrospective
            premium, covered, notcovered,c,nc, cm, ncm = computeRetro(form['location'].value(),form['subscriptionDate'].value(),form['rainfall'].value(),form['dailyMaxTurnover'].value(),form['fixedCosts'].value())
            context['price'] = str(premium)
            context['c'] = str(covered)
            context['nc'] = str(notcovered)
            context['cm'] = str(list(cm.values()))
            context['ncm'] = str(list(ncm.values()))
            print(str(list(cm.values())))
            print(str(list(ncm.values())))
            pdf = FPDF(orientation='P', unit='mm', format='A4')
            pdf.add_page()
            pdf.rect(5.0, 5.0, 200.0,287.0)
            pdf.rect(8.0, 8.0, 194.0,282.0)

            #App pic
            #pdf.image(static('RainyDaysHero/images/rdh.png'), 10, 8, 33)
            dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            pdf.image( os.path.join(dirname, 'static/RainyDaysHero/images/rdh.png'), 10, 8, 33)
            pdf.set_font('Arial', 'B', 15)

            # Client Name
            pdf.cell(140)
            pdf.cell(0, 5, str(form['clientName'].value()), ln=1)

            #Company name
            pdf.ln(25)
            pdf.cell(0, 5, 'Rainy Days Hero', ln=1)

            
            #Informatios
            pdf.set_text_color(238, 58, 20)
            pdf.ln(6)
            pdf.cell(60)
            pdf.cell(65, 10, 'Rain Insurance quotation','B', ln=2)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(65, 10, "Max daily turover: "+str(form['dailyMaxTurnover'].value()), ln=2)
            pdf.cell(65, 10, "Fixed costs: "+str(form['fixedCosts'].value()), ln=2)
            pdf.cell(65, 10, "Crucial rainfall: "+str(form['rainfall'].value()), ln=2)
            pdf.cell(65, 10, "Subsciption date: "+form['subscriptionDate'].value(), ln=2)
            #pdf.cell(65, 10, "Subsciption date: "+form['subscriptionDate'].value().strftime("%Y-%m-%d"), ln=2)
            pdf.cell(65, 10, "Duration: "+"365 days", ln=2)
            pdf.cell(65, 10, "Location: "+form['location'].value(), ln=2)

            #premium
            pdf.set_text_color(39, 174, 96)
            pdf.ln(10)
            pdf.cell(60)
            pdf.set_font('Arial', 'B', 15)
            pdf.cell(65, 10, "Premium Price: "+premium+" "+chr(128), ln=2)
            pdf.cell(65, 10, "Covered Result: "+covered+" "+chr(128), ln=2)
            pdf.cell(65, 10, "Uncovered Result: "+notcovered+" "+chr(128), ln=2)

            #graph days

            '''
            days = [i for i in range(365)]
            plt.plot(days,c,label="Covered")
            plt.plot(days,nc,label="Uncovered")
            '''

            #graph months
            months = [i for i in range(1,13)]
            months_c = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Months') 
            plt.ylabel('Results')
            plt.plot(months_c,cm.values(),label="Covered")
            plt.plot(months_c,ncm.values(),label="Uncovered")


            #plot graph
            plt.title("Result Evolution Graph", fontweight="bold", fontsize=16, color="blue")
            plt.tight_layout()
            #plt.legend(loc='best')
            #lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
            lgd = plt.legend(title='Caption', bbox_to_anchor=(1.05, 1), loc='upper left')
            retroFIG = ''.join(choice(ascii_uppercase) for i in range(100))+'.png'

            plt.savefig(retroFIG, bbox_extra_artists=(lgd,), bbox_inches='tight')
            pdf.image(retroFIG,45, 170 ,120)
            #delete the fig
            os.remove(retroFIG)
            plt.close()

            #save file and del after response
            quotationPDFFile = ''.join(choice(ascii_uppercase) for i in range(100))+'.pdf'
            pdf.output(quotationPDFFile,'F')
            response = HttpResponse(pdf, content_type='application/pdf')
            response['Content-Disposition'] = "attachment; filename=quotationPDFFile"
            removequotationfile.after_response(quotationPDFFile)
            #response.write(open("tuto.pdf"))
            if form['printPDF'].value()=='Yes':
                return FileResponse(open(quotationPDFFile, "rb"), as_attachment=True, filename='retrospective.pdf')
            else:
                return HttpResponse(template.render(context, request))
    else:
        template = loader.get_template('RainyDaysHero/retrospective.html')
        form = RetroForm(request.POST)
        context = dict(form= form)
        return HttpResponse(template.render(context, request)) 


'''Retrospective computation for Rain Insurrance: Without IA'''

def computeRetro(location,date,rainfall,turnover,fixedCosts):
    rainfall = float(rainfall)
    turnover = float(turnover)
    fixedCosts = float(fixedCosts)
    #data to use according to city : init dataFrame //TODO : update data before using
    location = location.lower()
    #dataUpdateCity(location)
    dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    df = pd.read_csv(os.path.join(dirname, 'static/RainyDaysHero/data/'+'export-'+location+'.csv'),skiprows=3)


    days = [i for i in range(365)]
    nc = [0 for _ in range(365)]
    c = [0 for _ in range(365)]
    cm = {i:0 for i in range(1,13)}
    ncm = {i:0 for i in range(1,13)}

    #compute retro
    tempDate = date+'-01-01'
    subscriptionDateEntry = DtT.datetime.strptime(tempDate, "%Y-%m-%d")
    tempDate = DtT.datetime.strptime(tempDate, "%Y-%m-%d")
    countSinister = 0
    premium = 0
    for i in range(365):
        tempDate = tempDate + DtT.timedelta(days=1)
        mm = '%02d' %  tempDate.month
        dd = '%02d' %  tempDate.day
        m,d,y = str(tempDate.month),str(tempDate.day),str(tempDate.year)
        pltDf = df[df['DATE'].str.contains(mm+'-'+dd)]
        #pltDf = df[df['DATE'].str.contains(mm+'-'+dd) and df['DATE'] < retroYearEntry ]

        sinister=0
        for plt in pltDf['PRECIP_TOTAL_DAY_MM']:
            s=0
            if plt > rainfall:
                s = fixedCosts
            else:
                s = -min(0, turnover*((rainfall - plt) / rainfall) - fixedCosts)
            sinister+=(s/len(pltDf.index))
        premium+=sinister / ( 1 + interestRate*i/360 )

        if df[df['DATE'].str.contains(date+'-'+mm+'-'+dd)].iloc[0]['PRECIP_TOTAL_DAY_MM'] > rainfall:
            nc[i] = - fixedCosts
            countSinister+=1
        else:
            nc[i] = turnover*( (rainfall - df[df['DATE'].str.contains(date+'-'+mm+'-'+dd)].iloc[0]['PRECIP_TOTAL_DAY_MM']) / rainfall ) - fixedCosts
            c[i]  = max(0,turnover*( (rainfall - df[df['DATE'].str.contains(date+'-'+mm+'-'+dd)].iloc[0]['PRECIP_TOTAL_DAY_MM']) / rainfall ) - fixedCosts)

        ncm[int(m)]+=nc[i]
        cm[int(m)]+=c[i]
    #return str("%.2f" % premium),str("%.2f" % (sum(c)-premium) ),str("%.2f" % sum(nc)), c, nc
    return str("%.2f" % premium),str("%.2f" % (sum(c)-premium) ),str("%.2f" % sum(nc)), c, nc, cm, ncm

