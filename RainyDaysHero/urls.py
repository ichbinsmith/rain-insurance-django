from django.urls import path

#home page and utils
from .app_views.index import index
from .app_views.general.contact import contact
from .app_views.general.about import about
from .app_views.general.subject import subject
from .app_views.general.userguide import userguide

#raininsurance views
from .app_views.raininsurance.quotation import quotation
from .app_views.raininsurance.retrospective import retrospective

#lifeinsurance views
from .app_views.lifeinsurance.general import portfolio
from .app_views.lifeinsurance.general import lxQxStress

##Term Insurance
from .app_views.lifeinsurance.terminsurance import terminsurance
from .app_views.lifeinsurance.terminsurance import terminsuranceAnalysis
from .app_views.lifeinsurance.terminsurance import terminsuranceReserve
from .app_views.lifeinsurance.terminsurance import terminsuranceAccounting
from .app_views.lifeinsurance.terminsurance import terminsuranceStress

##Pure Endowment
from .app_views.lifeinsurance.pe import PureEndowment
from .app_views.lifeinsurance.pe import PureEndowmentAnalysis
from .app_views.lifeinsurance.pe import PureEndowmentReserve
from .app_views.lifeinsurance.pe import PureEndowmentAccounting
from .app_views.lifeinsurance.pe import PureEndowmentStress


app_name = 'rdh'
urlpatterns = [
    # ex: /polls/
    path('', index, name='index'),
    path('quotation/', quotation, name='quotation'),
    path('retrospective/', retrospective, name='retrospective'),
    path('terminsurance/', terminsurance, name='terminsurance'),
    path('terminsurance-analysis/', terminsuranceAnalysis, name='terminsuranceAnalysis'),
    path('terminsurance-reserve/', terminsuranceReserve, name='terminsuranceReserve'),
    path('terminsurance-stress/', terminsuranceStress, name='terminsuranceStress'),
    path('terminsurance-accounting/', terminsuranceAccounting, name='terminsuranceAccounting'),
    path('pureendowment/', PureEndowment, name='PureEndowment'),
    path('pureendowment-analysis/', PureEndowmentAnalysis, name='PureEndowmentAnalysis'),
    path('pureendowment-reserve/', PureEndowmentReserve, name='PureEndowmentReserve'),
    path('pureendowment-stress/', PureEndowmentStress, name='PureEndowmentStress'),
    path('pureendowment-accounting/', PureEndowmentAccounting, name='PureEndowmentAccounting'),
    path('lx-qx-stress/', lxQxStress, name='lxQxStress'),
    path('portfolio/', portfolio, name='portfolio'),
    path('contact/', contact, name='contact'),
    path('about/', about, name='about'),
    path('userguide/', userguide, name='userguide'),
    path('subject/', subject, name='subject'),
]