import re
from rest_framework import mixins
from rest_framework import viewsets
from rest_framework.exceptions import ValidationError
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework_jwt.authentication import JSONWebTokenAuthentication
from rest_framework import authentication
from .serializers import *
from apps.utils.util import send_email,create_vaildcode
from django.contrib.auth.backends import ModelBackend
from django.db.models import Q


User = get_user_model()

class CumstomBackend(ModelBackend):
    def authenticate(self, request, username=None,email=None, password=None, **kwargs):
        try:
            user = User.objects.get(username=username)
            print(1)
            if user.check_password(password):
                return user
        except Exception as e:
            return None


"""用户"""

class RegisterViewSet(APIView):
    """注册用户"""
    def post(self,request,*args,**kwargs):
        # 校验参数
        serializer = UserLoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # 保存
        serializer.save()

        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": [],
                         }, status=status.HTTP_200_OK)


# 用户修改
class UserViewset(mixins.UpdateModelMixin, mixins.CreateModelMixin,mixins.RetrieveModelMixin,viewsets.GenericViewSet, mixins.ListModelMixin):
    """
    retrieve：查看信息
    update：更新用户，用户修改信息
    """
    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)
    serializer_class = UserDetailSerializer
    permission_classes = (IsAuthenticated,)

    def update(self, request, *args, **kwargs):
        # 获取用户
        user = request.user

        email = request.data.get('email',None)
        username = request.data.get('username',None)

        if not all([email,username]):
            raise ValidationError('参数不全')

        emailcount = UserProfile.objects.filter(email=email).exclude(id=request.user.id).count()
        usernamecount = UserProfile.objects.filter(username=username).exclude(id=request.user.id).count()

        if emailcount > 0:
            raise ValidationError('邮箱存在')

        if usernamecount > 0:
            raise ValidationError('用户名存在')


        user.email = email
        user.username = username
        user.save()


        return Response({"status_code": status.HTTP_200_OK,
                             "message": "ok",
                             "results": "修改成功",
                             }, status=status.HTTP_200_OK)

    def retrieve(self, request, *args, **kwargs):
        user_id = request.user.id
        try:
            user = UserProfile.objects.filter(id=int(user_id)).get()

        except Exception as e:
            print(e)
            raise ValidationError("参数错误")

        ret = self.get_serializer(user)
        ret = ret.data


        # 文案数
        ret["document_count"] = len(ret["document"])

        return Response({"status_code": status.HTTP_200_OK,
                             "message": "ok",
                             "results": [ret],
                             }, status=status.HTTP_200_OK)

    def perform_create(self, serializer):
        return serializer.save()


class PasswordViewset(mixins.UpdateModelMixin,viewsets.GenericViewSet):
    """
    update：更新密码
    """
    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)
    permission_classes = (IsAuthenticated,)
    def update(self, request, *args, **kwargs):
        # 获取用户
        user = request.user


        serializer = PasswordSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # 校验验证码
        try:
            ture_vaildcode = int(cache.get(request.user.email, None))
        except Exception as e:
            print(e)
            raise ValidationError({'error': ['验证码错误']})


        if ture_vaildcode != int(serializer.validated_data["vaildcode"]):
            raise ValidationError({'error': ['验证码错误']})

        # 把缓存删除
        cache.set(request.user.email, '555', 1)
        user.set_password(serializer.validated_data["password"])

        user.save()


        return Response({"status_code": status.HTTP_200_OK,
                             "message": "ok",
                             "results": "修改成功",
                             }, status=status.HTTP_200_OK)




class VaildcodeViewSet(APIView):
    """
    生成验证码
    """

    def post(self,request,*args,**kwargs):
        # 获取email
        email = request.data.get("email","11")

        # 校验email
        result = re.match(r"^[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)+$", email)

        if result is None:
            raise ValidationError("邮箱为空或格式错误")


        # 生成验证码
        code = create_vaildcode(email)

        # 发送验证码
        send_status = send_email(valid_code=code,email=email)
        # send_status = 1
        if send_status == 1:

            return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": "",
                         }, status=status.HTTP_200_OK)
        else:
            return Response({"status_code": '400',
                             "message": "error",
                             "results":"发送失败",
                             }, status=status.HTTP_200_OK)


class HeadPicViewSet(APIView):
    """
    头像
    """
    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)
    permission_classes = (IsAuthenticated,)
    def get(self,request,*args,**kwargs):
        user = request.user

        try:
            pic_url = user.head_pic.url
        except Exception as e:
            print(e)
            pic_url = None
        print(pic_url)
        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": [{"pic":pic_url}],
                         }, status=status.HTTP_200_OK)

    def post(self,request,*args,**kwargs):
        user = request.user

        pic = request.FILES.get('file')

        if pic is None:
            raise ValidationError("未上传文件")

        user.head_pic = pic
        user.save()

        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": [],
                         }, status=status.HTTP_200_OK)














