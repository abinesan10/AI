
from django.db import models
from jsonfield import JSONField
import jsonfield
from django.db.models import Q
# Create your models here.




class registerInfo(models.Model):
    email = models.CharField(max_length=80,null=True)
    password = models.CharField(max_length=80)
    reset_pwd_code = models.CharField(max_length=50,null=True)
    reference_token = models.CharField(max_length=255,null=True)
    access_token = models.CharField(max_length=255,null=True)
    registered_date = models.DateField(null=True)
    onlineStatus = models.CharField(default="offline",max_length=30,null=True)
    mobileNumber = models.CharField(max_length=30,null=True)
    approval_status = models.CharField(default="pending",max_length=30,null=True)
    mobileCountryCode=models.CharField(max_length=30,null=True)


class photoGallery(models.Model):
    dates=models.DateTimeField(auto_now_add=True, blank=True)
    profile_picture = jsonfield.JSONField(null=True)
    image_type = models.CharField(max_length=80)



class projectdetails(models.Model):
    dates=models.DateTimeField(auto_now_add=True, blank=True)
    # videoname = jsonfield.JSONField(null=True)
    projectId = models.IntegerField()
    videoName = models.CharField(max_length=80)
    dates=models.DateTimeField(auto_now_add=True, blank=True)



class projectname(models.Model):
    dates=models.DateTimeField(auto_now_add=True, blank=True)
    # videoname = jsonfield.JSONField(null=True)
    projectName = models.CharField(max_length=255)
    projectDesc =models.TextField(null=True)

    