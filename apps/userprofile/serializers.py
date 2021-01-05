# -*- coding: utf-8 -*-

# Date: 2019/8/5
# Name: serializers


from rest_framework import serializers,exceptions
from django.contrib.auth import get_user_model
from rest_framework.validators import UniqueValidator
from apps.userprofile.models import UserProfile
from django.core.cache import cache
from apps.folder.models import Folder
from apps.document.serializers import DocumentSerializer


User = get_user_model()


class UserDetailSerializer(serializers.ModelSerializer):
    """
    用户序列化类
    """
    document = DocumentSerializer(many=True)

    create_time = serializers.SerializerMethodField()

    last_time = serializers.SerializerMethodField()

    head_pic = serializers.SerializerMethodField()

    def get_last_time(self,row):
        time = row.last_time
        if time == None:
            time = row.create_time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            time = time.strftime("%Y-%m-%d %H:%M:%S")
        return time

    def get_create_time(self, row):

        return row.create_time.strftime("%Y-%m-%d %H:%M:%S")

    def get_head_pic(self,row):
        return 'http://192.168.0.105:8007' + row.head_pic.url

    class Meta:
        model = User
        fields = ("id", "username", "email","head_pic","document","create_time","last_time")



class UserLoginSerializer(serializers.Serializer):
    """
    添加用户序列化类
    """
    email = serializers.EmailField(
                    error_messages={"required":"不能为空"},
                    validators=[UniqueValidator(queryset=User.objects.all(),message="邮箱存在")]
    )

    username = serializers.CharField(max_length=8,min_length=1,error_messages={"required": "邮箱不能为空"}, validators=[UniqueValidator(queryset=User.objects.all(),message="用户名存在")])

    password = serializers.CharField(error_messages={"required": "密码不能为空"},style={'input_type': 'password'},help_text="密码", label="密码", write_only=True)

    re_password = serializers.CharField(error_messages={"required": "重复密码不能为空"},style={'input_type': 'password'},help_text="密码", label="密码", write_only=True)

    vaildcode = serializers.CharField(error_messages={"required":"验证码不能为空"})

    def validate(self, attrs):
        # attrs是除了全局钩子其他校验都通过了的数据
        # 这一步之后，得到的数据就会存放到validated_data中，用于真正的入库，所以要将一些无法入库或者不需要入库的数据剔除，如仅作校验密码一致性的re_password
        password = attrs.get('password')
        re_password = attrs.pop('re_password')
        if password != re_password:
            raise exceptions.ValidationError({'error':'两次密码不一致'})

        # 校验验证码
        try:
            ture_vaildcode = int(cache.get(attrs["email"],None))

        except Exception as e:
            print(e)
            raise exceptions.ValidationError({'error': '验证码错误'})


        if ture_vaildcode != int(attrs["vaildcode"]):
            raise exceptions.ValidationError({'error': '验证码错误'})

        return attrs

    def create(self, validated_data):
        # 去除验证码和二次输入密码
        validated_data.pop("vaildcode")

        user = UserProfile.objects.create(**validated_data)
        user.set_password(validated_data["password"])
        user.save()

        Folder.objects.create(name="默认文件夹",user=user)
        return user





class PasswordSerializer(serializers.Serializer):
    """
    添加用户序列化类
    """
    password = serializers.CharField(error_messages={"required": "不能为空"}, style={'input_type': 'password'},
                                     help_text="密码", label="密码", write_only=True)

    re_password = serializers.CharField(error_messages={"required": "不能为空"}, style={'input_type': 'password'},
                                        help_text="密码", label="密码", write_only=True)

    vaildcode = serializers.CharField(error_messages={"required": "不能为空"})

    def validate(self, attrs):
        # attrs是除了全局钩子其他校验都通过了的数据
        # 这一步之后，得到的数据就会存放到validated_data中，用于真正的入库，所以要将一些无法入库或者不需要入库的数据剔除，如仅作校验密码一致性的re_password
        password = attrs.get('password')
        re_password = attrs.pop('re_password')
        if password != re_password:
            raise exceptions.ValidationError({'error': '两次密码不一致'})


        return attrs




