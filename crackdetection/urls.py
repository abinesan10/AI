"""crackdetection URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from users import views


urlpatterns = [
    path('AMS/API/user/register',views.register),
    path('AMS/API/user/login',views.login),
    path('AMS/API/user/logout',views.logout),
    path('AMS/API/upload/crack/images',views.Crack_photo),
    path('AMS/API/upload/noncroack/images',views.Noncrack_photo),
    path('AMS/API/create/train/category',views.add_image_category),
    path('AMS/API/rename/train/category',views.rename_image_category),
    path('AMS/API/list/category/names',views.list_category),
    path('AMS/API/upload/train/images/upload',views.train_image_upload),
    path('AMS/API/train/images',views.train_images),
    path('AMS/API/<int:pwd>',views.api_list),
    path('AMS/API/list/noncrack/images',views.list_noncrack_images),
    path('AMS/API/list/crack/images',views.crack_images),
    path('AMS/API/delete/crack/images',views.crack_images_delete),
    path('AMS/API/delete/noncrack/images',views.noncrack_images_delete),
    path('AMS/API/upload/video',views.upload_video),
    path('AMS/API/list/video/<int:id>',views.list_video),
    path('AMS/API/delete/video',views.videos_delete),
    path('AMS/API/project/name/list',views.project_list),
    # path('AMS/API/predict/images',views.predict),
    path('AMS/API/project/name/add',views.project_add),
    path('AMS/API/project/name/update',views.project_update),
    path('AMS/API/project/name/list',views.project_list),
    path('AMS/API/project/video/detect',views.video_detect),
    path('AMS/API/project/dashboard/details/<int:id>',views.dashboard),
    path('AMS/API/project/list/detected/images/<int:id>',views.list_detected_images),
    path('AMS/API/project/list/images/<str:type>',views.list_images),
    path('AMS/API/delete/images',views.images_delete),
    
    # path('AMS/API/project/name/delete',views.project_delete),
    


    # path('AMS/API/change/password',views.change_password),
    # path('AMS/API/forgot/password/token',views.forgot_password_token),
    # path('AMS/API/password/token/validation',views.forgot_password_validation),
    # path('AMS/API/forgot/password/reset',views.forgot_password_reset),
]

    