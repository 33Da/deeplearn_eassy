from rest_framework.views import exception_handler
from rest_framework.exceptions import APIException
from rest_framework import status
from django.http import HttpResponse
import random
from django.core.mail import send_mail
from django.core.cache import cache
from rest_framework.response import Response
from rest_framework.exceptions import *

def custom_exception_handler(exc, context):
    # 先调用DRF默认的 exception_handler 方法, 对异常进行处理，
    # 如果处理成功，会返回一个`Response`类型的对象
    response = exception_handler(exc, context)

    # 一般异常处理
    if response is None:
        return {"status_code": "500", "message": "error", "results": "服务器错误"}

    # 捕捉ValidationError
    if isinstance(exc, ValidationError):
        return Response({"status_code": 401, "message": "error", "results": response.data})

    # 捕捉NotAuthenticated
    if isinstance(exc, NotAuthenticated):
        return Response({"status_code": 402, "message": "error", "results": "用户未验证"})

    if isinstance(exc, NotFound):
        return Response({"status_code": 403, "message": "error", "results": "没有接口"})

    if isinstance(exc, MethodNotAllowed):
        return Response({"status_code": 404, "message": "error", "results": "请求方式错误"})

    if isinstance(exc, NotFound):
        return Response({"status_code": 403, "message": "error", "results": "请求方式错误"})

    return response




def jwt_response_username_userid_token(token, user=None, request=None):
    """
    自定义验证成功后的返回数据处理函数
    :param token:
    :param user:
    :param request:
    :return:
    """

    data = {
        # jwt令牌
        'status_code': 200,
        'token': token,
        'user_id': user.id,
        'username': user.username,
    }
    return data


def create_vaildcode(email):
    """
    生成验证码保存redis
    :return:
    """
    # 生成随机六位数
    vaildcode = random.randrange(100000, 999999, 1)

    print(email)

    # 保存到redis 5分钟
    cache.set(email,vaildcode,60*10)


    return str(vaildcode)

def send_email(email,valid_code):
    """
    发送邮件
    :param email:
    :param valid_code:
    :return:
    """

    # 发送邮件
    email_title = '您的验证码为:*****'
    email_body = valid_code

    try:
        send_status = send_mail(email_title, email_body, "764720843@qq.com", [email])
    except:
        send_status = 0


    return send_status


# 修改jwt返回的错误中间件
class JWTException:
    def __init__(self,get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        return response

    def process_template_response(self,request,response):

        if hasattr(response,"data"):
            if response.data.get('non_field_errors') is not None:
                del response.data['non_field_errors']
                response.data['status_code'] = '101'
                response.data["message"] = "error"
                response.data["results"] = "用户名或密码错误"
        return response


import fitz




def PDF_to_imgs(PDF_path, save_path):
    # 打开PDF文件，生成一个对象
    doc = fitz.open(PDF_path)

    # 将PDF文件的每一页都转化为图片
    for pg in range(doc.pageCount):
        page = doc[pg]
        rotate = int(0)
        # 每个尺寸的缩放系数为2，这将为我们生成分辨率提高4倍的图像。
        zoom_x = 1
        zoom_y = 1
        trans = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
        pm = page.getPixmap(matrix=trans, alpha=False)
        pm.writePNG(save_path +'%s.png' % pg)
    return save_path