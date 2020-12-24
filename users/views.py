from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.mail import BadHeaderError, send_mail,EmailMultiAlternatives
from django.template.loader import render_to_string, get_template
from django.template import Context
from django.conf import settings
from datetime import datetime,timedelta
from django.http import JsonResponse,HttpResponse
from rest_framework.authtoken.models import Token
from django.db.models import F,Count
from django.views.decorators.http import require_http_methods
from collections import defaultdict
from rest_framework import serializers
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage 
from .serializers import *
from .models import *
from django import forms 
from django.shortcuts import render

# from django.db import connection


#####Python Libraries.
import pandas as pd
import  time
from datetime import date,datetime
import json
import random, string
import bcrypt
import uuid
import hashlib
import requests
import requests, zipfile, io
import gpxpy
import pickle 
################
import numpy as np
import os
import cv2
import imageio

# %matplotlib inline
import matplotlib.pyplot as plt
import pickle
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
# from keras.preprocessing import image
# import keras
from pathlib import Path

from skimage import io
# %matplotlib notebook
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

from skimage.io import imread
from skimage.transform import resize


#### Code start's below

def hash_password(password):
    hashPass = hashlib.sha256(password.encode()).hexdigest()
    return hashPass

def get_random_string(length):
    # put your letters in the following string
    sample_letters = 'abcdef123j4k5l6m7n8o9p0qgrhsitxyzuvw'
    return (''.join((random.choice(sample_letters) for i in range(length))))


def api_list(request,pwd):
    # put your letters in the following string
    return render(request, 'api.html')

subject = "xxxxxxx"
message = "xxxxxxxxxx"
register_email_from = settings.EMAIL_HOST_USER
@csrf_exempt
def register(request):
    if request.method == 'POST':
        try:
            js = json.loads(request.body)
            dateField = datetime.now()
            js["registered_date"] = dateField.date()
            #print(js["registered_date"],"===============================")
            password = js["password"]
            hashPass = hash_password(password)
            js["password"] = str(hashPass)
            emailCheck = registerInfo.objects.filter(Q(email__icontains=js["email"]) | Q(mobileNumber__icontains=js["mobileNumber"]))
            if emailCheck.exists():
                return JsonResponse({"status":"failure","message":"User already exists"})
            else:
                registerSerializer = registerInfoSerializer(data=js)
                if registerSerializer.is_valid() :
                    registerId = registerSerializer.save()
                    registerId.save()
                    # msg_plain = render_to_string('email.txt', {'username': "some_params"})
                    # msg_html = render_to_string('test.html', {'username': "jjjjj"})
                    # recipient_list = [js["email"]]
                    # msg = EmailMultiAlternatives(subject, msg_plain, register_email_from, recipient_list)
                    # msg.attach_alternative(msg_html, "text/html")
                    # msg.send()
                    return JsonResponse({"status":"success","message":"Registered successfully"}) 
                else:
                    return JsonResponse({"status":"failure","message":"Request not valid"})        
        except Exception as error:
            return JsonResponse({"status":"failue","message":str(error)})          

@csrf_exempt
def login(request):
    if request.method == 'POST':
        try:
            js = json.loads(request.body)
            email = js["email"]
            password = js["password"]
            hashPass = hash_password(password)
            js["registered_date"] = datetime.now()
            pwdCheck = registerInfo.objects.filter(email=email,password=str(hashPass))
            if pwdCheck.exists():
                referenceToken = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase +string.digits, k = 26))
                accessToken = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase +string.digits, k = 31))
                registerInfo.objects.filter(email=email).update(reference_token=referenceToken,access_token=accessToken,onlineStatus="online")
                userDetail = registerInfo.objects.get(email=email)
                jsonData = registerInfoSerializer(userDetail,many=False)
                del jsonData.data["password"]
                return JsonResponse({"status":"success","message":"Logged in successfully","userId":jsonData.data["id"],"userDetail":jsonData.data})
            else:
                return JsonResponse({"status":"failure","message":"Email and Password mismatching"})   
        except Exception as error:
            return JsonResponse({"status":"failue","message":str(error)})          

@csrf_exempt
def logout(request):
    if request.method == 'POST':
        try:
            js = json.loads(request.body)
            userId = js["userId"]
            userIdCheck = registerInfo.objects.filter(id=userId)
            if userIdCheck.exists():
                logout = registerInfo.objects.filter(id=userId).update(onlineStatus="offline")
                data = {"status":"success","message":"logged out successfully"}
            else:
                data = {"status":"failure","message":"user not exists"}
            return JsonResponse(data)
        except Exception as error:
            return JsonResponse({"status":"failue","message":str(error)})          

@csrf_exempt
#@validate
@require_http_methods(["POST"])
def Noncrack_photo(request):
    image_type = request.POST["ImageType"]   
    timeStamp = datetime.now().timestamp()
    filePath = FileSystemStorage(location='/var/www/html/noncrack')
    timeStamp = str(timeStamp).replace('.','_')
    # r = requests.get("https://omexon.co/images1.zip")
    # z = zipfile.ZipFile(io.BytesIO(r.content))
    print('Extracting all the files now...') 
    # z.extractall("D:/Road crack Colan/images/zip")
    print('Done!') 
    fileUrl = []
    # print(request.FILES,"sdsddddddd")
    for  i in request.FILES:
        file = request.FILES[i] 
        print(file,"ddddddddddddddd")
        # print(file.name,"dddddddddddddddddddddddddddd")
        extensions=['.png','.jpg']
        if file.name.endswith(tuple(extensions)):
            fileName = get_random_string(5)+str(timeStamp)+".png"
            #print(fileName,"333333333")
            path = filePath.save(fileName, ContentFile(file.read()))
            fileUrl.append(fileName)
            img = photoGallery(image_type=image_type,profile_picture=fileUrl)
            img.save()
        
        if file.name.endswith('.zip'):
            print ('File is a ZIP')
            zip_file = zipfile.ZipFile(file, 'r')
            for file in zip_file.namelist():    
                if file.endswith(tuple(extensions)):
                    # print("dddddddd")
                    zip_file.extract(file,'/var/www/html/noncrack')  
            zip_file.close()
    data = {"status":"success","message":"Image uploaded successfully"}
    return JsonResponse(data)


@csrf_exempt
#@validate
@require_http_methods(["POST"])
def Crack_photo(request):
    image_type = request.POST["ImageType"]   
    timeStamp = datetime.now().timestamp()
    filePath = FileSystemStorage(location='/var/www/html/crack')
    timeStamp = str(timeStamp).replace('.','_')
    fileUrl = []
    # print(request.FILES,"sdsddddddd")
    for  i in request.FILES:
        file = request.FILES[i] 
        print(file,"ddddddddddddddd")
        # print(file.name,"dddddddddddddddddddddddddddd")
        extensions=['.png','.jpg']
        if file.name.endswith(tuple(extensions)):
            fileName = get_random_string(5)+str(timeStamp)+".png"
            #print(fileName,"333333333")
            path = filePath.save(fileName, ContentFile(file.read()))
            fileUrl.append(fileName)
            img = photoGallery(image_type=image_type,profile_picture=fileUrl)
            img.save()
        
        if file.name.endswith('.zip'):
            print ('File is a ZIP')
            zip_file = zipfile.ZipFile(file, 'r')
            for file in zip_file.namelist():    
                if file.endswith(tuple(extensions)):
                    # print("dddddddd")
                    zip_file.extract(file,'/var/www/html/crack')  
            zip_file.close()
    data = {"status":"success","message":"Image uploaded successfully"}
    return JsonResponse(data)


def load_image_files(container_path, dimension=(64, 64)):
    """
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset
    
    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to
        
    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)



@csrf_exempt
#@validate
@require_http_methods(["GET"])
def train_images(request):
    data = []
    DIRECTORY = '/var/www/html'#'D:/Road crack Colan/xnzhj3x8v4-2/448/train'#'
    print("okkkkkkkkkkkkkkkkkkkkkkkkk")
    CATEGORIES = ['crack', 'noncrack']
    image_dataset = load_image_files(DIRECTORY)
    X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)
    param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid)
    print("okkkkkkkkkkkkkkkkkkkkkkkkk")
    clf.fit(X_train, y_train)
    print("okkkkkkkkkkkkkkkkkkkkkkkkk")
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)
    
    print("okkkkkkkkkkkkkkkkkkkkkkkkk")
    data = {"status":"success","message":"Images trained successfully"}
    return JsonResponse(data)
    # for category in CATEGORIES:
    #     path = os.path.join(DIRECTORY, category)
    #     for img in os.listdir(path):
    #         img_path = os.path.join(path, img)
    #         label = CATEGORIES.index(category)
    #         arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #         new_arr = cv2.resize(arr, (60, 60))
    #         data.append([new_arr, label])


    # random.shuffle(data)

    # X = []
    # y = []

    # for features, label in data:
    #     X.append(features)
    #     y.append(label)

    # X = np.array(X)
    # y = np.array(y)

    # import pickle

    # pickle.dump(X, open('X.pkl', 'wb'))
    # pickle.dump(y, open('y.pkl', 'wb'))

    # X = pickle.load(open('X.pkl', 'rb'))
    # y = pickle.load(open('y.pkl', 'rb'))

    # X = X/255

    # X

    # X = X.reshape(-1, 60, 60, 1)

    # from tensorflow.keras.models import Sequential
    # from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

    # model = Sequential()

    # model.add(Conv2D(64, (3,3), activation = 'relu'))
    # model.add(MaxPooling2D((2,2)))

    # model.add(Conv2D(64, (3,3), activation = 'relu'))
    # model.add(MaxPooling2D((2,2)))

    # model.add(Flatten())

    # model.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))

    # model.add(Dense(2, activation = 'softmax'))

    # model.compile(optimizer='adam',
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])

    # model.fit(X, y, epochs=5, validation_split=0.1)

    # import pickle

    # X = pickle.load(open('X.pkl', 'rb'))
    # y = pickle.load(open('y.pkl', 'rb'))

    # X = X/255

    # X = X.reshape(-1, 60, 60, 1)

    # from tensorflow.keras.models import Sequential
    # from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

    # from tensorflow.keras.callbacks import TensorBoard
    # import time

    # dense_layers = [3]
    # conv_layers = [3]
    # neurons = [64]


    # for dense_layer in dense_layers:
    #     for conv_layer in conv_layers:
    #         for neuron in neurons:

    #             NAME = '{}-denselayer-{}-convlayer-{}-neuron-{}'.format(dense_layer, conv_layer, neuron, int(time.time()))
    #             tensorboard = TensorBoard(log_dir = 'logs2\\{}'.format(NAME))


    #             model = Sequential()

    #             for l in range(conv_layer):
    #                 model.add(Conv2D(neuron, (3,3), activation = 'relu'))
    #                 model.add(MaxPooling2D((2,2)))

    #             model.add(Flatten())

    #             model.add(Dense(neuron, input_shape = X.shape[1:], activation = 'relu'))

    #             for l in range(dense_layer - 1):
    #                 model.add(Dense(neuron, activation = 'relu'))

    #             model.add(Dense(2, activation = 'softmax'))

    #             model.compile(optimizer='adam',
    #                         loss='sparse_categorical_crossentropy',
    #                         metrics=['accuracy'])

    #             model.fit(X, y, epochs=8, batch_size = 32, validation_split=0.1, callbacks = [tensorboard])

    #             model.save('3x3x64-catvsdog.model')
    # data = {"status":"success","message":"Images trained successfully"}
    # return JsonResponse(data)


@csrf_exempt
#@validate
@require_http_methods(["GET"])
def list_noncrack_images(request):
    a=[]
    for i in  os.listdir('/var/www/html/noncrack'):
        # print(i)
        url="http://44.233.138.4/noncrack/"+i
        print(url)
        a.append(url)
    
    data = {"status":"success","message":"File Name","data":a}
    return JsonResponse(data)



@csrf_exempt
#@validate
@require_http_methods(["GET"])
def crack_images(request):
    a=[]
    for i in  os.listdir('/var/www/html/crack'):
        # print(i)
        url="http://44.233.138.4/crack/"+i
        print(url)
        a.append(url)
    
    data = {"status":"success","message":"File Name","data":a}
    return JsonResponse(data)



@csrf_exempt
#@validate
@require_http_methods(["POST"])
def crack_images_delete(request):
    js = json.loads(request.body)
    for i in js["fileName"] :
        filepath="/var/www/html/crack/"+i
        print(filepath)
        os.remove(filepath)
    data = {"status":"success","message":"Files Deleted successfully"}
    return JsonResponse(data)






@csrf_exempt
#@validate
@require_http_methods(["POST"])
def noncrack_images_delete(request):
    js = json.loads(request.body)
    for i in js["fileName"] :
        filepath="/var/www/html/noncrack/"+i
        print(filepath)
        os.remove(filepath)
    data = {"status":"success","message":"Files Deleted successfully"}
    return JsonResponse(data)




@csrf_exempt
#@validate
@require_http_methods(["POST"])
def videos_delete(request):
    js = json.loads(request.body)
    for i in js["fileName"] :
        filepath="/var/www/html/videos/"+i
        print(filepath)
        os.remove(filepath)
    data = {"status":"success","message":"Deleted successfully"}
    return JsonResponse(data)


@csrf_exempt
#@validate
@require_http_methods(["POST"])
def upload_video(request):   
    project = request.POST["projectId"] 
    timeStamp = datetime.now().timestamp()
    filePath = FileSystemStorage(location="D:/var/")#'/var/www/html/videos/')#"D:/")
    timeStamp = str(timeStamp).replace('.','_')
    fileUrl = []
    # print(request.FILES,"sdsddddddd")
    tmst = str(timeStamp)
    for  i in request.FILES:
        print(i,"kkkkkkkkkkkkkkkk")
        file = request.FILES[i] 
        print(file,"dddddddddddddddkkkkkkkk")
        print(file.name,"dddddddddddddddddddddddddddd")
        extensions=['.mp4','.avi','.mkv','.wmv','.MP4']
        exten=['.gpx']
        if file.name.endswith(tuple(extensions)):
            videofileName = tmst+file.name
            fileName=videofileName
            print(videofileName,"333333333")
        if file.name.endswith(tuple(exten)):
            gpxfileName = tmst+file.name
            fileName=gpxfileName
            print(gpxfileName,"333333333")
        path = filePath.save(fileName, ContentFile(file.read()))
    print("llllllllllllll")
    pro = projectdetails(projectId=project,videoName=videofileName,status=0,gpxFile=gpxfileName)
    pro.save()
    data = {"status":"success","message":"Video uploaded successfully"}
    return JsonResponse(data)





@csrf_exempt
#@validate
@require_http_methods(["GET"])
def list_video(request,id):
    project = list(projectdetails.objects.filter(projectId=int(id)).values())
    print("ddddddd",project)
    # for i in js["fileName"] :
    #     filepath="/var/www/html/noncrack/"+i
    #     print(filepath)
    #     os.remove(filepath)
    path=[{"pathUrl":"http://44.233.138.4/videos/","videosList":project}]
    data = {"status":"success","data":path}
    return JsonResponse(data)


def image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img, (60, 60))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 60, 60, 1)
    return new_arr

@csrf_exempt
#@validate
@require_http_methods(["POST"])
def predict(request):  
    CATEGORIES = ['crack', 'noncrack']
    timeStamp = datetime.now().timestamp()
    model = keras.models.load_model('3x3x64-catvsdog.model')
    timeStamp = datetime.now().timestamp()
    
    timeStamp = str(timeStamp).replace('.','_')
    for  i in request.FILES:
        file = request.FILES[i] 
        prediction = model.predict([ContentFile(file.read())])
        print(CATEGORIES[prediction.argmax()])
    data = {"status":"success","predicted result":CATEGORIES[prediction.argmax()]}
    return JsonResponse(data)



         
@csrf_exempt
#@validate
@require_http_methods(["POST"])
def project_add(request):  
    js = json.loads(request.body)
    project = projectname.objects.filter(projectName=js["projectName"])
    if project.exists():
        return JsonResponse({"status":"failure","message":"Project already exists"})
    else:
        pro = projectname(projectName=js["projectName"],projectDesc=js["projectDesc"])
        pro.save()
        return JsonResponse({"status":"Success","message":"Project Created Successfully!"})


@csrf_exempt
#@validate
@require_http_methods(["GET"])
def project_list(request):  
    project = list(projectname.objects.all().values())
    return JsonResponse({"status":"Success","message":"project name list","data":project})


@csrf_exempt
#@validate
@require_http_methods(["POST"])
def project_update(request):  
    js = json.loads(request.body)
    update = projectname.objects.filter(id=js["projectId"]).update(projectName=js["projectName"],projectDesc=js["projectDesc"])
    return JsonResponse({"status":"Success","message":"Updated Successfully"})


@csrf_exempt
#@validate
@require_http_methods(["POST"])
def video_detect(request): 
    js = json.loads(request.body)
    
    try:
        #img = io.imread("D:/Road crack Colan/xnzhj3x8v4-2/448/test/Cracks/IMG_20180513_170430826_HDR resized_448.jpg")    
        timeStamp = datetime.now().timestamp()
        timeStamp = str(timeStamp).replace('.','_')
        video_folder="/var/www/html/videos/" #
        image_folder="/var/www/html/images/" #"D:/var/"#"D:/var/"
        # vidcap = cv2.VideoCapture(video_folder+js["videoName"])
        reader = imageio.get_reader(video_folder+js["videoName"])
        metadata = reader.get_meta_data()
        gpx = gpxpy.parse(open(video_folder+js["gpxFileName"]))
        track = gpx.tracks[0]
        segment = track.segments[0]
        data=[]
        datalon = []
        datalat = []
        segment_length = segment.length_3d()
        for point_idx, point in enumerate(segment.points):
            tim=str(point.time.hour)+str(point.time.minute)+str(point.time.second)
        #     print(tim)
            if tim in data:
                pass
            else:
                datalon.append(point.longitude)
                datalat.append(point.latitude)
                data.append(tim)


      
        if metadata["fps"]<25 and metadata["fps"]>23:
            fs=24
        if metadata["fps"]<31 and metadata["fps"]>28:
            fs=30
        poin=0
        for frame_number, im in enumerate(reader):
            # im is numpy array
            if frame_number % fs == 0:
                imname=image_folder+str(frame_number)+"_"+timeStamp+".jpg"
                namesave=str(frame_number)+"_"+timeStamp+".jpg"
                imageio.imwrite(imname,im)
                img_resized = resize(im, (64,64), anti_aliasing=True, mode='reflect')
                p=img_resized.flatten()
                pkl_filename = "pickle_model.pkl"
                with open(pkl_filename, 'rb') as file:
                    pickle_model = pickle.load(file)
                print("okkkkkkkkkkkkkkkkkk")
                y_pred = pickle_model.predict([p])
                print(y_pred,"dddddddddddddddddddd")
                if y_pred ==0:
                    detectStatus=1
                if y_pred ==1:
                    detectStatus=0   
                else:
                    detectStatus=0
                print(y_pred)
            
                # print(datalat[poin])
                if poin>=len(datalat):
                    print(poin)
                    return JsonResponse({"status":"Success","message":"Frame detected Successfully"})
                else:
                    detect = detectiondetails(imageName=namesave,detectStatus=detectStatus,videoId=js["videoId"],imlat=datalat[poin],imlong=datalon[poin])
                    detect.save()
                    print(poin)
                    poin+=1
                    
                
        # CATEGORIES = ['crack', 'noncrack']
        # def getFrame(sec):
        #     vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        #     hasFrames,image = vidcap.read()
        #     if hasFrames:
        #         imname=image_folder+str(count)+"_"+timeStamp+".jpg"
        #         namesave=str(count)+"_"+timeStamp+".jpg"
        #         cv2.imwrite(imname, image)     # save frame as JPG file
        #         # model = keras.models.load_model('3x3x64-catvsdog.model')
        #         img = cv2.imread(imname, cv2.IMREAD_GRAYSCALE)
        #         new_arr = cv2.resize(img, (60, 60))
        #         new_arr = np.array(new_arr)
        #         new_arr = new_arr.reshape(-1, 60, 60, 1)
                # prediction = model.predict([new_arr])
                # if CATEGORIES[prediction.argmax()]== "crack":
                #     detectStatus=1
                # else:
                #     detectStatus=0
                # print(detectStatus)
                # detect = detectiondetails(imageName=namesave,detectStatus=0,videoId=js["videoId"])
                # detect.save()
            # return hasFrames
        # sec = 0
        # frameRate = js["FramesPerSecond"] #//it will capture image in each 0.5 second
        # count=1
        # success = getFrame(sec)
        # while success:
        #     count = count + 1
        #     sec = sec + frameRate
        #     sec = round(sec, 2)
        #     success = getFrame(sec)
        # update = projectdetails.objects.filter(id=js["videoId"]).update(status=1,totalFrames=count,detectedFrames=0)
        return JsonResponse({"status":"Success","message":"Frame detected Successfully"})
    except Exception as e:
        return JsonResponse({"status":"Failure","message":str(e)})

    
@csrf_exempt
#@validate
@require_http_methods(["GET"])
def list_detected_images(request,id):
    project = list(detectiondetails.objects.filter(videoId=int(id)).values())
    path=[{"pathUrl":"http://44.233.138.4/images/","videoImagesList":project}]
    data = {"status":"success","data":path}
    return JsonResponse(data)



