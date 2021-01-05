"""deeplearn_eassy URL Configuration

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
from extra_apps import xadmin
from django.conf.urls import url, include
from rest_framework_jwt.views import obtain_jwt_token
from rest_framework.authtoken import views
from deeplearn_eassy.settings import MEDIA_ROOT, STATIC_ROOT
from django.views.static import serve
from django.conf import settings  ##新增
from django.conf.urls import url  ##新增

from django.views import static  ##新增

urlpatterns = [
    path('xadmin/', xadmin.site.urls),

    # 接口
    path('api/', include("apps.folder.urls")),
    path('api/', include("apps.userprofile.urls")),
    path('api/', include("apps.document.urls")),
    path('api/back/', include('apps.back.urls')),

    # # drf自带的token认证模式
    # url(r'^api-token-auth/', views.obtain_auth_token),

    # 富文本编辑
    url(r'ueditor/', include('DjangoUeditor.urls')),

    # jwt的认证接口
    path('api/user/login/', obtain_jwt_token),

    url(r"^media/(?P<path>.*)$", serve, {"document_root": MEDIA_ROOT}),
    url(r'^static/(?P<path>.*)$', static.serve,
        {'document_root': settings.STATIC_ROOT}, name='static'),

]
