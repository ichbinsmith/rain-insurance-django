from django.shortcuts import render
from django.http import Http404
from django.http import HttpResponse
from django.template import loader
from django.http import HttpResponseRedirect
from django.http import FileResponse
from django.shortcuts import render
import os


def subject(request):
    dirname = os.path.dirname(__file__)
    return FileResponse(open(os.path.join(dirname, 'static/RainyDaysHero/docs/subject.pdf'), "rb"), as_attachment=True, filename='subject.pdf')
