from rest_framework import serializers
from .models import *


class registerInfoSerializer(serializers.ModelSerializer):
    class Meta():
        model = registerInfo
        fields = "__all__"
