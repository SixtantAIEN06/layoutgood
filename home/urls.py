from django.urls import path
from . import views
urlpatterns = [
    path('',views.index),
    path('gallery/',views.gallery),
    path('styletransfer/',views.styletransfer),
    path('styletransfer2/',views.styletransfer2),
    path('delmypic/',views.delmypic,name='uplaodfile'),
    # path('styletransfer/{imgpath:imgpath}',views.styletransfer),
    path('json/',views.json),
    path('objectdetection/',views.objectdetection),
    path('facerecognition/',views.facerecognition),
    path('styletransfer/json/',views.json),
    path('styletransfer/httpget/',views.httpget),

]
