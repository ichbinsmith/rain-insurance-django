from django.urls import path

from . import views


app_name = 'rdh'
urlpatterns = [
    # ex: /polls/
    path('', views.index, name='index'),
    path('quotation/', views.quotation, name='quotation'),
    path('retrospective/', views.retrospective, name='retrospective'),
    path('terminsurance/', views.terminsurance, name='terminsurance'),
    path('terminsurance-analysis/', views.terminsuranceAnalysis, name='terminsuranceAnalysis'),
    path('contact/', views.contact, name='contact'),
    path('about/', views.about, name='about'),
    path('userguide/', views.about, name='userguide'),
    path('subject/', views.about, name='subject'),
]