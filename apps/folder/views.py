from django.shortcuts import render
import re

from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework import viewsets
from .serializers import FolderSerializer
from rest_framework_jwt.authentication import JSONWebTokenAuthentication
from rest_framework import authentication
from .models import Folder
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response
from rest_framework import status
from rest_framework import viewsets
from rest_framework import mixins
from rest_framework.exceptions import *


class P1(PageNumberPagination):
    """
    基于页码
    """
    # 默认每页显示的数据条数
    page_size = 10
    # 获取url参数中设置的每页显示数据条数
    page_size_query_param = 'size'
    # 获取url中传入的页码key
    page_query_param = 'page'
    # 最大支持的每页显示的数据条数
    max_page_size = 50



class FolderViewSet(viewsets.GenericViewSet,mixins.CreateModelMixin,mixins.DestroyModelMixin,mixins.ListModelMixin,mixins.RetrieveModelMixin,mixins.UpdateModelMixin):
    """文件夹增删改查"""

    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)
    permission_classes = (IsAuthenticated,)
    serializer_class = FolderSerializer
    queryset = Folder.objects.all()

    def list(self, request, *args, **kwargs):
        user = request.user

        p1 = P1()

        folders = Folder.objects.filter(user=user)
        page_list = p1.paginate_queryset(queryset=folders, request=request, view=self)

        folder = FolderSerializer(instance=page_list,many=True)

        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": {
                             "date":folder.data,
                             "count":len(folders)
                         },
                         }, status=status.HTTP_200_OK)


    def create(self,request,*args,**kwargs):
        user = request.user

        name = request.data.get("name")
        
        count = Folder.objects.filter(name=name,user=user).count()
        if name is None or count > 0:
            return Response({"status_code": 200,
                         "message": "error",
                         "results": '文件夹名存在',
                         }, status=status.HTTP_200_OK)


        folder = Folder.objects.create(name=name,user=user)

        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": {'name':folder.name,'id':folder.id},
                         }, status=status.HTTP_200_OK)


    def destroy(self,request,*args,**kwargs):

        # 获取数据
        folder_id = kwargs.get("pk",None)

        try:
            folder = Folder.objects.get(id=folder_id)
        except Exception as e:
            return Response({"status_code": 400,
                             "message": "error",
                             "results": "没有该文件夹",
                             }, status=status.HTTP_200_OK)
        # 判断是否存在该文件夹
        if folder.user.id != request.user.id:
            return Response({"status_code": 400,
                             "message": "error",
                             "results": "没有该文件夹",
                             }, status=status.HTTP_200_OK)

        if folder.name == "默认文件夹":
            return Response({"status_code": 200,
                             "message": "error",
                             "results": "不能删除默认文夹",
                             }, status=status.HTTP_200_OK)
        folder.delete()

        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": [],
                         }, status=status.HTTP_200_OK)


    def update(self,request,*args,**kwargs):

        # 获取数据
        folder_id = request.data.get("id",None)

        name = request.data.get("name",None)
        print(folder_id,name)

        # 判断参数完整性
        if not all([folder_id,name]):
            raise ValidationError("参数不全")

        try:
            folder = Folder.objects.get(id=folder_id)
        except Exception as e:
            raise ValidationError("文件夹id错误")

        if folder.name == "默认文件夹":
            raise ValidationError("默认文件夹不允许修改")

        count = Folder.objects.filter(name=name,user=request.user).count()
        if count > 0 and folder.name != name:
            raise ValidationError("已存在该文件夹")
        
        # 修改
        folder.name = name
        folder.save()

        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": [],
                         }, status=status.HTTP_200_OK)


    def retrieve(self, request, *args, **kwargs):
        id = kwargs.get("pk",0)


        try:
            folder = Folder.objects.get(id=id)
        except Exception as e:
            return Response({"status_code": status.HTTP_400_BAD_REQUEST,
                             "message": "error",
                             "results": "找不到文件夹",
                             }, status=status.HTTP_200_OK)



        folder = FolderSerializer(instance=folder)

        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results":  folder.data,
                         }, status=status.HTTP_200_OK)








