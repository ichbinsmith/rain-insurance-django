from django.shortcuts import render
from django.http import Http404
from django.http import HttpResponse
from django.template import loader
from django.http import HttpResponseRedirect
from django.http import FileResponse
from django.shortcuts import render
from .forms import QuotationForm
import io
import os
from random import choice
from string import ascii_uppercase

import after_response
from django.templatetags.static import static


from fpdf import FPDF

@after_response.enable
def removequotationfile(filename):
    os.remove(filename)

def index(request):
    template = loader.get_template('RainyDaysHero/index.html')
    context = {}
    return HttpResponse(template.render(context, request))

def quotation(request):
    context = {}
    if request.method == 'POST':
        template = loader.get_template('RainyDaysHero/quotationAnswer.html')
        form = QuotationForm(request.POST)
        if form.is_valid():
            context['form'] = form
            context['price'] = form.price
            #compute quotation
            ##return HttpResponse(template.render(context, request))
            pdf = FPDF(orientation='P', unit='mm', format='A4')
            pdf.add_page()
            pdf.rect(5.0, 5.0, 200.0,287.0)
            pdf.rect(8.0, 8.0, 194.0,282.0)

            #App pic
            #pdf.image(static('RainyDaysHero/images/rdh.png'), 10, 8, 33)
            dirname = os.path.dirname(__file__)
            pdf.image( os.path.join(dirname, 'static/RainyDaysHero/images/rdh.png'), 10, 8, 33)
            pdf.set_font('Arial', 'B', 15)

            # Client Name
            pdf.cell(140)
            pdf.cell(0, 5, "clientName", ln=1)

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
            premium = 0
            pdf.set_text_color(39, 174, 96)
            pdf.ln(25)
            pdf.cell(60)
            pdf.set_font('Arial', 'B', 15)
            pdf.cell(65, 10, "Premium: "+str("%.2f" % premium)+" "+chr(128), ln=2)

            #save file and del after response
        
            quotationPDFFile = ''.join(choice(ascii_uppercase) for i in range(100))+'.pdf'
            pdf.output(quotationPDFFile,'F')
            response = HttpResponse(pdf, content_type='application/pdf')
            response['Content-Disposition'] = "attachment; filename=quotationPDFFile"
            removequotationfile.after_response(quotationPDFFile)
            #response.write(open("tuto.pdf"))
            return FileResponse(open(quotationPDFFile, "rb"), as_attachment=True, filename='quotation.pdf')


            if os.path.exists(os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')+'\\quotation.pdf'):
                os.remove(os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')+'\\quotation.pdf')
            pdf.output(os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')+'\\quotation.pdf','F')
    else:
        template = loader.get_template('RainyDaysHero/quotation.html')
        form = QuotationForm(request.POST)
        context = dict(form= form)
        return HttpResponse(template.render(context, request))

def retrospective(request):
    template = loader.get_template('RainyDaysHero/retrospective.html')
    context = {}
    return HttpResponse(template.render(context, request))

def contact(request):
    template = loader.get_template('RainyDaysHero/contact.html')
    context = {}
    return HttpResponse(template.render(context, request))

def about(request):
    template = loader.get_template('RainyDaysHero/about.html')
    context = {}
    return HttpResponse(template.render(context, request))
