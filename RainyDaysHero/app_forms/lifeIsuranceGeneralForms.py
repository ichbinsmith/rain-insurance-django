from django import forms

class LxQxStressForm(forms.Form):

    stressOn = forms.ChoiceField(label = 'Stress On',choices = ( ("Lx", "Lx"), ("Qx", "Qx")))
    stressOn.widget.attrs.update({'class': 'form-control'})

    stressOnTable = forms.FloatField(label='Stress on table (%)')
    stressOnTable.widget.attrs.update({'class': 'form-control','value':0})

class PortfolioForm(forms.Form):
    mortalityStress = forms.FloatField(label='Stress on mortality table (%)')
    mortalityStress.widget.attrs.update({'class': 'form-control','value':0})

    interestRateStress = forms.FloatField(label='Stress on interest rate (%)')
    interestRateStress.widget.attrs.update({'class': 'form-control','value':0})

    adaptedModel = forms.ChoiceField(label = 'Adapted model',choices = ( ("No", "No"), ("Yes", "Yes") ))
    adaptedModel.widget.attrs.update({'class': 'form-control'})

    IAorActuarial = forms.ChoiceField(label = 'IA, Actuarial',choices = ( ("IA", "IA"), ("Actuarial", "Actuarial") ))
    IAorActuarial.widget.attrs.update({'class': 'form-control'})


