from django import forms
import datetime as DtT
#from bootstrap_datepicker_plus import DatePickerInput


cities = ( 
    ("paris", "Paris"), 
    ("nantes", "Nantes"), 
    ("nice", "Nice"),
    ("strasbourg", "Strasbourg")
)
class QuotationForm(forms.Form):
    clientName = forms.CharField(label='Company Name',max_length=100)
    clientName.widget.attrs.update({'class': 'form-control', 'value':'Carrefour Antibes'})

    dailyMaxTurnover = forms.FloatField(label='Daily Max Turnover')
    dailyMaxTurnover.widget.attrs.update({'class': 'form-control','value':1000})

    fixedCosts = forms.FloatField(label='Daily Fixed Costs')
    fixedCosts.widget.attrs.update({'class': 'form-control','value':450	})

    rainfall = forms.FloatField(label='Critic Rainfall (mm)')
    rainfall.widget.attrs.update({'class': 'form-control','value':2})

    #subscriptionDate = forms.DateField(widget=DatePickerInput(format='%m/%d/%Y'))
    subscriptionDate = forms.DateField(widget=forms.DateInput(attrs={'type': 'date', 'class':'form-control','pattern':'\\d{4}-\\d{2}-\\d{2}' , 'value': DtT.datetime.today().strftime('%Y-%m-%d')}), label='Subscription Date')

    location = forms.ChoiceField(label = 'Company Location',choices = cities)
    location.widget.attrs.update({'class': 'form-control'})

    printPDF = forms.ChoiceField(label = 'Export As Pdf',choices = ( ("No", "No"),("Yes", "Yes")))
    printPDF.widget.attrs.update({'class': 'form-control'})


class RetroForm(forms.Form):
    clientName = forms.CharField(label='Company Name',max_length=100)
    clientName.widget.attrs.update({'class': 'form-control', 'value':'Burger King'})

    dailyMaxTurnover = forms.FloatField(label='Daily Max Turnover')
    dailyMaxTurnover.widget.attrs.update({'class': 'form-control','value':1000})

    fixedCosts = forms.FloatField(label='Daily Fixed Costs')
    fixedCosts.widget.attrs.update({'class': 'form-control','value':450 })

    rainfall = forms.FloatField(label='Critic Rainfall (mm)')
    rainfall.widget.attrs.update({'class': 'form-control','value':2})

    #subscriptionDate = forms.DateField(widget=DatePickerInput(format='%m/%d/%Y'))
    subscriptionDate = forms.CharField(label='Retrospective Year',max_length=4)
    subscriptionDate.widget.attrs.update({'class': 'form-control', 'value':'2019'})
    
    location = forms.ChoiceField(label = 'Company Location',choices = cities)
    location.widget.attrs.update({'class': 'form-control'})

    printPDF = forms.ChoiceField(label = 'Export As Pdf',choices = ( ("No", "No"), ("Yes", "Yes") ) )
    printPDF.widget.attrs.update({'class': 'form-control'})




