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


#### Code start's below

def hash_password(password):
    hashPass = hashlib.sha256(password.encode()).hexdigest()
    return hashPass

def get_random_string(length):
    # put your letters in the following string
    sample_letters = 'abcdef123j4k5l6m7n8o9p0qgrhsitxyzuvw'
    return (''.join((random.choice(sample_letters) for i in range(length))))

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
                logout = myProfileInfo.objects.filter(id=userId).update(onlineStatus="offline")
                data = {"status":"success","message":"logged out successfully"}
            else:
                data = {"status":"failure","message":"user not exists"}
            return JsonResponse(data)
        except Exception as error:
            return JsonResponse({"status":"failue","message":str(error)})          

