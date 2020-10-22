
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
    mobileVerifiedStatus=models.IntegerField(null=True)

    