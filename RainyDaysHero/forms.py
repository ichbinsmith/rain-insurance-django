from django import forms

class QuotationForm(forms.Form):
    clientName = forms.CharField(label='Company Name', max_length=100)
    dailyMaxTurnover = forms.CharField(label='Company Name', max_length=100)
    fixedCosts = forms.CharField(label='Company Name', max_length=100)
    rainfall = forms.CharField(label='Company Name', max_length=100)
    subcriptionDate = forms.CharField(label='Company Name', max_length=100)
    location = forms.CharField(label='Company Name', max_length=100)