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

################
import numpy as np
import os
import cv2
# %matplotlib inline
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

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
    filePath = FileSystemStorage(location='/var/www/domain2.com/public_html/448/train/NonCracks')
    timeStamp = str(timeStamp).replace('.','_')
   
    fileUrl = []
    for i in request.FILES:
        file = request.FILES[i]
        # fileName = str(userId)+"_"+str(timeStamp)+str(file)
        fileName = get_random_string(5)+str(timeStamp)+".png"
        #print(fileName,"333333333")
        path = filePath.save(fileName, ContentFile(file.read()))
        fileUrl.append(fileName)
    img = photoGallery(image_type=image_type,profile_picture=fileUrl)
    img.save()
    data = {"status":"success","message":"Image uploaded successfully"}
    return JsonResponse(data)


@csrf_exempt
#@validate
@require_http_methods(["POST"])
def Crack_photo(request):
    image_type = request.POST["ImageType"]   
    timeStamp = datetime.now().timestamp()
    filePath = FileSystemStorage(location='/var/www/domain2.com/public_html/448/train/Cracks')
    timeStamp = str(timeStamp).replace('.','_')
    fileUrl = []
    for i in request.FILES:
        file = request.FILES[i]
        # fileName = str(userId)+"_"+str(timeStamp)+str(file)
        fileName = get_random_string(5)+str(timeStamp)+".png"
        #print(fileName,"333333333")
        path = filePath.save(fileName, ContentFile(file.read()))
        fileUrl.append(fileName)
    img = photoGallery(image_type=image_type,profile_picture=fileUrl)
    img.save()
    data = {"status":"success","message":"Image uploaded successfully"}
    return JsonResponse(data)


@csrf_exempt
#@validate
@require_http_methods(["GET"])
def train_images(request):
    DIRECTORY = '/var/www/domain2.com/public_html/448/train'
    CATEGORIES = ['Cracks', 'NonCracks']
    data = []
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            label = CATEGORIES.index(category)
            arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            new_arr = cv2.resize(arr, (60, 60))
            data.append([new_arr, label])
    random.shuffle(data)
    X = []
    y = []
    for features, label in data:
        X.append(features)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    pickle.dump(X, open('X.pkl', 'wb'))
    pickle.dump(y, open('y.pkl', 'wb'))
    X = pickle.load(open('X.pkl', 'rb'))
    y = pickle.load(open('y.pkl', 'rb'))
    X = X/255
    X = X.reshape(-1, 60, 60, 1)
    model = Sequential()

    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())

    model.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))

    model.add(Dense(2, activation = 'softmax'))

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(X, y, epochs=5, validation_split=0.1)
    data = {"status":"success","message":"Images trained successfully"}
    return JsonResponse(data)


