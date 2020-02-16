from django.shortcuts import render,redirect
from django.http import HttpResponse,JsonResponse
from django.core.files.storage import FileSystemStorage
from .dlib import forImport_recognize_faces_image
from .BeautyGAN import main2 as BeautyGAN
from .BeautyGAN import split as beautysplit
import cv2
import glob
import os
# import subprocess
import tensorflow as tf
import datetime 
from .models import Classified
# from .modelsclassified import classified
from django.core import serializers
from django.core.serializers import serialize



# from .flask.peeweetest import Classified

from .object_detection.evaluate import YoloTest
YoloTest = YoloTest()

# Create your views here.
def index(request):

    #return HttpResponse("<p>Hello world!</p>")
    return render(request,'home/index.html')

def selected(request):
    # if request.method=='POST':
    #     request
    # condition = request.POST['searchbox']
    typessname="person"
    typenamewithnober=typessname+"__gt"
    print(typenamewithnober)
    numbers=1
    # # keypair={type:number}
    # select_pic=classified()
    # datas=select_pic.selected(types,numbers)
    
    # select_pic.objects.filter(types=numbers)
    datas=Classified.objects.values_list("image_path").filter(bear__gte=numbers)
    # datas=cat.objects.filter(types=numbers)

    # print(datas)
    return render(request,'select.html',locals())

def gallery(request):
    # public_pic=classified()
    # datas=public_pic.all()
    # datas=Classified.objects.filter(person__gt=0)


    # print(Classified.objects.values_list("image_path"))
    datas=Classified.objects.values_list("image_path")
    # print(datas[89])
    # return JsonResponse(datas,safe=False)
    return render(request,'gallery.html',locals())

def facerecognition(request):
    if request.method =='POST' and request.FILES['photoupload']:
        myfile=request.FILES['photoupload']
        fs = FileSystemStorage(location='home/static/images/')
        fs.save(myfile.name,myfile)
        forImport_recognize_faces_image.readPara("home/dlib/encoding3.pickle",f'home/static/images/{myfile.name}','cnn') #f'home/static/images/{myfile.name}
        photopath="images/upload.jpg"

    title = "FACE RECOGNITION"
    now = datetime.datetime.now()
    return render(request,'layout.html',locals())


def styletransfer(request):  
    if request.method =='POST' and request.FILES['photoupload']:
        myfile=request.FILES['photoupload']
        fs = FileSystemStorage(location='home/static/images/')
        fs.save(myfile.name,myfile)
        # forImport_recognize_faces_image.readPara("home/dlib/encoding3.pickle",f'home/static/images/{myfile.name}','cnn')
        BeautyGAN.beauty(f'home/static/images/{myfile.name}')
        makeups = glob.glob(os.path.join('home','static','makeupstyle','*'))
        photopaths=[]
        for i in range(len(makeups)):
            photopaths.append(f"makeupstyle/{i+1}.jpg")
        print(photopaths)

        return redirect("/styletransfer2")
    title = "STYLE TRANSFER"
    now = datetime.datetime.now()
    return render(request,'layout.html',locals())
    
def styletransfer2(request):
    if request.method =='POST' and request.POST["style"]:

        beautysplit.split(request.POST["style"])
        makeups = glob.glob(os.path.join('home','static','makeupstyle','*'))
        photopath="./home/static/temp/split.jpg"
        # stylephoto=os.listdir("./home/static/makeupstyle")
        # print(stylephoto)
        title = "SELECT THE STYLE YOU LIKE!"
    # elif os.path('./home/static/temp/split.jpg'):
    #     os.remove('./home/static/temp/split.jpg')  
    
    now=datetime.datetime.now()
    return render(request,'styletransfer2.html',locals())  
        

def objectdetection(request):
    if request.method =='POST' and request.FILES['photoupload']:
        myfile=request.FILES['photoupload']
        fs = FileSystemStorage(location='home/static/images/')
        fs.save(myfile.name,myfile)
        # print(f'home/static/images/{myfile.name}.jpg')
        
        # photopath="images/upload.jpg"
        path_head='home/static/'
        img_path=f'images/{myfile.name}'
        # YoloTest.evaluate(f'home/static/images/{myfile.name}')
        YoloTest.evaluate(path_head,img_path)
        # print(YoloTest.dlist)
        for item in YoloTest.dlist:
            print("========================================")

            print(item)
            print("========================================")
            sort =Classified.objects.create(**item)
            sort .save()
            # sort = Classified(**item) 
            # sort.save()
    title="OBJECT DETECTION"
    now=datetime.datetime.now()
    return render(request,'layout.html',locals())


def delmypic(request):
    now=datetime.datetime.now()
    return render(request,'delmypic.html',locals())

def json(request):
    # if request.method =='POST':
    datas={"name":"ford","age":"31"}
    # now=datetime.datetime.now()
    return JsonResponse(datas,safe=False)


def httpget(request):
    # if request.method =='POST':
    name=request.GET["name"]
    age=request.GET["age"]

    # now=datetime.datetime.now()
    return HttpResponse(f"HELLO {name},{age}")