from django import forms

# from bootstrap_datepicker_plus import DatePickerInput

models = (
    ("knn", "KNN"),
)


class PureEndowmentForm(forms.Form):
    # clientAge = forms.IntegerField(label='Age',max_length=3) --> No max length
    clientAge = forms.IntegerField(label='Age')
    clientAge.widget.attrs.update({'class': 'form-control', 'value': 50})

    numberOfPayements = forms.IntegerField(label='Number of annual payements')
    numberOfPayements.widget.attrs.update({'class': 'form-control', 'value': 5})

    maturity = forms.IntegerField(label='Maturity')
    maturity.widget.attrs.update({'class': 'form-control', 'value': 5})

    interestRate = forms.FloatField(label='Interest Rate (%)')
    interestRate.widget.attrs.update({'class': 'form-control', 'value': 1})

    amount = forms.FloatField(label='Amount')
    amount.widget.attrs.update({'class': 'form-control', 'value': 1000})

    model = forms.ChoiceField(label='Model', choices=models)
    model.widget.attrs.update({'class': 'form-control'})


class PureEndowmentReserveForm(forms.Form):
    contractOrTotal = forms.ChoiceField(label='Contract/Total Reserve',
                                        choices=(("Total", "Total"), ("Contract", "Contract")))
    contractOrTotal.widget.attrs.update({'class': 'form-control', 'onchange': 'reserveTypeSwitch()'})

    mortalityStress = forms.FloatField(label='Stress on mortality table (%)')
    mortalityStress.widget.attrs.update({'class': 'form-control', 'value': 0})

    interestRateStress = forms.FloatField(label='Stress on interest rate (%)')
    interestRateStress.widget.attrs.update({'class': 'form-control', 'value': 0})

    adaptedModel = forms.ChoiceField(label='Adapted model', choices=(("No", "No"), ("Yes", "Yes")))
    adaptedModel.widget.attrs.update({'class': 'form-control'})

    # clientAge = forms.IntegerField(label='Age',max_length=3) --> No max length
    clientAge = forms.IntegerField(label='Age')
    clientAge.widget.attrs.update({'class': 'form-control', 'value': 50})

    numberOfPayements = forms.IntegerField(label='Number of annual payements')
    numberOfPayements.widget.attrs.update({'class': 'form-control', 'value': 5})

    maturity = forms.IntegerField(label='Maturity')
    maturity.widget.attrs.update({'class': 'form-control', 'value': 5})

    interestRate = forms.FloatField(label='Interest Rate (%)')
    interestRate.widget.attrs.update({'class': 'form-control', 'value': 1})

    amount = forms.FloatField(label='Amount')
    amount.widget.attrs.update({'class': 'form-control', 'value': 1000})

    IAorActuarial = forms.ChoiceField(label='IA, Actuarial or Both',
                                      choices=(("IA", "IA"), ("Actuarial", "Actuarial"), ("Both", "Both")))
    IAorActuarial.widget.attrs.update({'class': 'form-control'})


class PureEndowmentStressForm(forms.Form):
    contractOrTotal = forms.ChoiceField(label='Contract/Total Reserve',
                                        choices=(("Total", "Total"), ("Contract", "Contract")))
    contractOrTotal.widget.attrs.update({'class': 'form-control', 'onchange': 'stressTypeSwitch()'})

    stressOn = forms.ChoiceField(label='Stress On',
                                 choices=(("Mortality Table", "Mortality Table"), ("Interest Rate", "Interest Rate")))
    stressOn.widget.attrs.update({'class': 'form-control'})

    stressType = forms.ChoiceField(label='Stress Type', choices=(
    ("Non Adapted IA", "Non Adapted IA"), ("Adapted IA", "Adapted IA"), ("Actuarial", "Actuarial"), ("All", "All")))
    stressType.widget.attrs.update({'class': 'form-control'})

    # clientAge = forms.IntegerField(label='Age',max_length=3) --> No max length
    clientAge = forms.IntegerField(label='Age')
    clientAge.widget.attrs.update({'class': 'form-control', 'value': 50})

    numberOfPayements = forms.IntegerField(label='Number of annual payements')
    numberOfPayements.widget.attrs.update({'class': 'form-control', 'value': 5})

    maturity = forms.IntegerField(label='Maturity')
    maturity.widget.attrs.update({'class': 'form-control', 'value': 5})

    interestRate = forms.FloatField(label='Interest Rate (%)')
    interestRate.widget.attrs.update({'class': 'form-control', 'value': 1})

    amount = forms.FloatField(label='Amount')
    amount.widget.attrs.update({'class': 'form-control', 'value': 1000})


class PureEndowmentBSForm(forms.Form):
    contractOrTotal = forms.ChoiceField(label='Contract/Total Reserve',
                                        choices=(("Total", "Total"), ("Contract", "Contract")))
    contractOrTotal.widget.attrs.update({'class': 'form-control', 'onchange': 'balanceTypeSwitch()'})

    mortalityStress = forms.FloatField(label='Stress on mortality table (%)')
    mortalityStress.widget.attrs.update({'class': 'form-control', 'value': 0})

    interestRateStress = forms.FloatField(label='Stress on interest rate (%)')
    interestRateStress.widget.attrs.update({'class': 'form-control', 'value': 0})

    adaptedModel = forms.ChoiceField(label='Adapted model', choices=(("No", "No"), ("Yes", "Yes")))
    adaptedModel.widget.attrs.update({'class': 'form-control'})

    # clientAge = forms.IntegerField(label='Age',max_length=3) --> No max length
    clientAge = forms.IntegerField(label='Age')
    clientAge.widget.attrs.update({'class': 'form-control', 'value': 50})

    numberOfPayements = forms.IntegerField(label='Number of annual payements')
    numberOfPayements.widget.attrs.update({'class': 'form-control', 'value': 5})

    maturity = forms.IntegerField(label='Maturity')
    maturity.widget.attrs.update({'class': 'form-control', 'value': 5})

    interestRate = forms.FloatField(label='Interest Rate (%)')
    interestRate.widget.attrs.update({'class': 'form-control', 'value': 1})

    amount = forms.FloatField(label='Amount')
    amount.widget.attrs.update({'class': 'form-control', 'value': 1000})

    IAorActuarial = forms.ChoiceField(label='IA, Actuarial', choices=(("IA", "IA"), ("Actuarial", "Actuarial")))
    IAorActuarial.widget.attrs.update({'class': 'form-control'})


