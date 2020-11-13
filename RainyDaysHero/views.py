from django.shortcuts import render
from django.http import Http404
from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template('RainyDaysHero/index.html')
    context = {}
    return HttpResponse(template.render(context, request))

def quotation(request):
    template = loader.get_template('RainyDaysHero/quotation.html')
    context = {}
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
