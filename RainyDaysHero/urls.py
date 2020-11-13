from django.urls import path

from . import views


app_name = 'rdh'
urlpatterns = [
    # ex: /polls/
    path('', views.index, name='index'),
    path('quotation/', views.quotation, name='quotation'),
    path('retrospective/', views.retrospective, name='retrospective'),
    path('contact/', views.contact, name='contact'),
    path('about/', views.about, name='about'),
]