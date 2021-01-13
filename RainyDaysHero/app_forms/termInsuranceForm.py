from django import forms
import datetime as DtT
#from bootstrap_datepicker_plus import DatePickerInput

models = (
    ("lr", "Linear Regression"),
    ("plr", "Polynomial Regression"),
    ("ps", "Polynomial Scaled"),
    ("lasso", "Lasso")
)

class TermInsuranceForm(forms.Form):
    #clientAge = forms.IntegerField(label='Age',max_length=3) --> No max length
    clientAge = forms.IntegerField(label='Age')
    clientAge.widget.attrs.update({'class': 'form-control', 'value':50})

    numberOfPayements = forms.IntegerField(label='Number of annual payements')
    numberOfPayements.widget.attrs.update({'class': 'form-control', 'value':5})

    maturity = forms.IntegerField(label='Maturity')
    maturity.widget.attrs.update({'class': 'form-control', 'value':5})

    interestRate = forms.FloatField(label='Interest Rate (%)')
    interestRate.widget.attrs.update({'class': 'form-control','value':1})

    amount = forms.FloatField(label='Amount')
    amount.widget.attrs.update({'class': 'form-control','value':1000})

    model = forms.ChoiceField(label = 'Model',choices = models)
    model.widget.attrs.update({'class': 'form-control'})




