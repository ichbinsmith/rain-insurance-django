from django.shortcuts import render
from django.http import Http404
from django.http import HttpResponse
from django.template import loader
from django.http import HttpResponseRedirect
from django.http import FileResponse
from django.shortcuts import render
import os



def index(request):
    template = loader.get_template('RainyDaysHero/index.html')
    context = {}
    return HttpResponse(template.render(context, request))

