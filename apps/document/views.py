from lxml import etree
from rest_framework.exceptions import *
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework_jwt.authentication import JSONWebTokenAuthentication
from rest_framework import authentication
from .models import Document,Folder
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response
from rest_framework import status
from .serializers import DocumentSerializer
from deeplearn_eassy.settings import BASE_DIR
import os
import re
from pydocx import PyDocX
import datetime
import pdfkit
from rest_framework import viewsets
from rest_framework import mixins
from django.db.models import Q
from aip import AipNlp
from ..back.models import AdminCountModel
import win32com.client as wc
import pythoncom
from deeplearn_eassy.settings import MEDIA_ROOT
# from multiprocessing import Process
# import multiprocessing
# from Algorithmpa.demo3 import detection,recg
import OCR
import pytesseract
from PIL import Image
from ..utils.util import PDF_to_imgs


class P1(PageNumberPagination):
    """
    基于页码
    """
    # 默认每页显示的数据条数
    page_size = 10
    # 获取url参数中设置的每页显示数据条数
    page_size_query_param = 'pagesize'
    # 获取url中传入的页码key
    page_query_param = 'page'
    # 最大支持的每页显示的数据条数
    max_page_size = 50


def type_auto(title,content):
    """文章类型识别"""
    APP_ID = '18786667'
    API_KEY = 'sVHpAWkQSR67jI0eb0ZFhrAL'
    SECRET_KEY = 'QocG018D51lWDtyzlEF5FiFnUtnPxkkp '

    # 去除标签存文本内容

    client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

    result = client.topic(title, content)

    types = ''
    for k, v in result["item"].items():
        if len(v) != 0:
            types = types + v[0]['tag'] + ","

    return types


def sentiment_auto(content):
    """情感识别"""
    APP_ID = '18786667'
    API_KEY = 'sVHpAWkQSR67jI0eb0ZFhrAL'
    SECRET_KEY = 'QocG018D51lWDtyzlEF5FiFnUtnPxkkp '


    client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

    result = client.sentimentClassify(content)

    sentiment_type = result['items'][0]['sentiment']
    # 0: 负向，1: 中性，2: 正向

    if sentiment_type == 0:
        sentiment = '负面'
    elif sentiment_type == 1:
        sentiment = '中性'
    else:
        sentiment = '正向'

    return sentiment


class DocumentViewSet(APIView):
    """
    文章逻辑

    post：新增文章

    put：修改文章
    """
    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)
    permission_classes = (IsAuthenticated,)

    def post(self, request, *args, **kwargs):

        file = request.FILES.get('file')

        folder_id = request.data.get("folder_id", 0)

        # 获取今天日期
        data = datetime.datetime.now()

        # 上传方式
        upload_type = 1

        if file is None:
            raise ValidationError("未上传文件")
        # 去掉后缀
        file_name = file.name.split(".")[0]
        file_type = file.name.split(".")[1]

        # 去掉"
        file_type = file_type.split("\"")[0]

        # 检查该用户是否存在该文件夹
        try:
            if folder_id == 0:
                folder = Folder.objects.filter(name="默认文件夹", user=request.user).first()
            else:
                folder = Folder.objects.get(id=folder_id)

        except Exception as e:
            print(e)
            raise ValidationError("文件夹不存在")

        if folder.user != request.user:
            raise ValidationError("文件夹不存在")

        documents = [d.title for d in Document.objects.filter(folder=folder).all()]
        if file_name in documents:
            raise ValidationError("文件夹名不允许重复")

        # 先在本地临时存下来
        if file_type == "txt":

            with open("media/" + file_name + ".txt", 'wb+') as destination:
                for line in file.chunks():
                    destination.write(line)

            with open("media/" + file_name + ".txt") as f:
                # 读出内容
                txt = f.read()

                if txt == None:
                    txt = ''

        elif file_type == "docx":
            with open("media/" + file_name + ".docx", 'wb+') as destination:
                for line in file.chunks():
                    destination.write(line)
            print(11)
            txt = PyDocX.to_html("media/" + file_name + ".docx")

            # 去前面的头
            res_tr = r'<body>(.*?)</body>'
            txt = ''.join(re.findall(res_tr, txt, re.S | re.M))

            # 删除本地存储的文件
            os.remove("media/" + file_name + ".docx")
        elif file_type == "doc":
            with open("media/" + file_name + ".doc", 'wb+') as destination:
                for line in file.chunks():
                    destination.write(line)
            pythoncom.CoInitialize()
            # doc文件另存为docx
            word = wc.Dispatch("Word.Application")
            # F:\Tencent Files\2026411509\FileRecv\deeplearn_essay_3.31\media
            doc = word.Documents.Open(MEDIA_ROOT + file_name + ".doc")
            # 上面的地方只能使用完整绝对地址，相对地址找不到文件，且，只能用“\\”，不能用“/”，哪怕加了 r 也不行，涉及到将反斜杠看成转义字符。
            doc.SaveAs(MEDIA_ROOT + file_name + ".docx", 12, False, "", True, "", False, False, False,
                       False)  # 转换后的文件,12代表转换后为docx文件
            doc.Close()
            word.Quit()
            txt = PyDocX.to_html("media/" + file_name + ".docx")

            # 去前面的头
            res_tr = r'<body>(.*?)</body>'
            txt = ''.join(re.findall(res_tr, txt, re.S | re.M))

            # 删除本地存储的文件
            os.remove("media/" + file_name + ".docx")
            os.remove("media/" + file_name + ".doc")
        elif file_type == "png":

            upload_type = 0
            """
            other版本：
            with open("media/" + file_name + ".png", 'wb+') as destination:
                for line in file.chunks():
                    destination.write(line)
            img_path = MEDIA_ROOT + file_name + ".png"
            print("img_path:",img_path)
            q = multiprocessing.Queue()
            p = Process(target=detection, args=(q, img_path,))
            p.start()
            rec = q.get()
            p.join()  # 进程结束后，GPU显存会自动释放

            w = multiprocessing.Queue()
            p = Process(target=recg, args=(w, rec, img_path))  # 重新识别
            p.start()
            txt = ''.join(w.get())
            p.join()
            """
            with open("media/" + file_name + ".png", 'wb+') as destination:
                for line in file.chunks():
                    destination.write(line)
            img_path = MEDIA_ROOT + file_name + ".png"
            txt_path = MEDIA_ROOT
            print(img_path, txt_path)
            txt = OCR.img_to_txt(img_path, txt_path)
            request.user.knowpic_count += 1

            try:
                data_count = AdminCountModel.objects.get(data=data)
                data_count.knowpic_count += 1
                data_count.save()
            except Exception as e:
                AdminCountModel.objects.create(knowpic_count=1)
        elif file_type == "jpg":
            upload_type = 0
            """
            other版本：
            with open("media/" + file_name + ".png", 'wb+') as destination:
                for line in file.chunks():
                    destination.write(line)
            img_path = MEDIA_ROOT + file_name + ".png"
            print("img_path:",img_path)
            q = multiprocessing.Queue()
            p = Process(target=detection, args=(q, img_path,))
            p.start()
            rec = q.get()
            p.join()  # 进程结束后，GPU显存会自动释放

            w = multiprocessing.Queue()
            p = Process(target=recg, args=(w, rec, img_path))  # 重新识别
            p.start()
            txt = ''.join(w.get())
            p.join()
            """
            with open("media/" + file_name + ".jpg", 'wb+') as destination:
                for line in file.chunks():
                    destination.write(line)
            img_path = MEDIA_ROOT + file_name + ".jpg"
            txt_path = MEDIA_ROOT
            print(img_path, txt_path)
            txt = OCR.img_to_txt(img_path, txt_path)
            request.user.knowpic_count += 1

            try:
                data_count = AdminCountModel.objects.get(data=data)
                data_count.knowpic_count += 1
                data_count.save()
            except Exception as e:
                AdminCountModel.objects.create(knowpic_count=1)

        elif file_type == "pdf":
            upload_type = 0
            """
            other版本：
            with open("media/" + file_name + ".png", 'wb+') as destination:
                for line in file.chunks():
                    destination.write(line)
            img_path = MEDIA_ROOT + file_name + ".png"
            print("img_path:",img_path)
            q = multiprocessing.Queue()
            p = Process(target=detection, args=(q, img_path,))
            p.start()
            rec = q.get()
            p.join()  # 进程结束后，GPU显存会自动释放

            w = multiprocessing.Queue()
            p = Process(target=recg, args=(w, rec, img_path))  # 重新识别
            p.start()
            txt = ''.join(w.get())
            p.join()
            """
            # 下载pdf
            with open("media/" + file_name + ".pdf", 'wb+') as destination:
                for line in file.chunks():
                    destination.write(line)
            # pdf转png
            PDF_path = MEDIA_ROOT + file_name + ".pdf"
            if not os.path.exists('media/' + file_name):
                os.mkdir('media/' + file_name)
            imgs_save_path = 'media/' + file_name + '/'
            png_path = PDF_to_imgs(PDF_path, imgs_save_path)
            img_path = png_path
            txt_path = MEDIA_ROOT + file_name + '/'
            print(img_path, txt_path)
            txt = ''
            for files, _, file_names in os.walk(img_path):
                print("files:", files)
                print("file_names:", file_names)
                for filename in file_names:
                    image = Image.open(files + filename)
                    # chi_sim 是中文识别包，equ 是数学公式包，eng 是英文包
                    content = pytesseract.image_to_string(image, lang="chi_sim")
                    txt += content
            request.user.knowpic_count += 1

            try:
                data_count = AdminCountModel.objects.get(data=data)
                data_count.knowpic_count += 1
                data_count.save()
            except Exception as e:
                AdminCountModel.objects.create(knowpic_count=1)

        else:
            raise ValidationError("文件格式错误")

        # 去除标签存文本内容
        response = etree.HTML(text=txt)
        content = response.xpath('string(.)').strip()

        word_count = len(content)

        type = type_auto(file_name, content)
        sentiment = sentiment_auto(content)

        Document.objects.create(folder=folder, user=request.user, title=file_name, htmlcontent=txt,
                                wordcount=word_count, content=content, sentiment=sentiment, type=type,upload_type=upload_type)
        request.user.document_count += 1
        request.user.save()


        # 写入记录
        try:
            data_count = AdminCountModel.objects.get(data=data)
            data_count.upload_count += 1
            data_count.save()
        except Exception as e:
            AdminCountModel.objects.create(upload_count=1)

        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": [],
                         }, status=status.HTTP_200_OK)



    def put(self, request, *args, **kwargs):
        d_id = request.data.get("id",0)
        d_title = request.data.get("d_title",None)
        d_htmlcontent = request.data.get("d_htmlcontent", None)

        folder_id = request.data.get('folder_id', 0)
        # 校验参数
        if not all([d_title,d_htmlcontent]):
            raise ValidationError("参数不全")

        # 去除标签存文本内容
        response = etree.HTML(text=d_htmlcontent)
        d_content = response.xpath('string(.)').strip()


        word_count = len(d_content)
        type = type_auto(d_title, d_content)
        sentiment = sentiment_auto(d_content)

        if d_id != 0:
            try:
                document = Document.objects.get(id=d_id)
            except Exception as e:
                print(e)
                raise ValidationError("找不到该文章")

            if document.user.id != request.user.id:
                raise ValidationError("找不到该文章")

            count = Document.objects.filter(title=d_title, folder=document.folder).exclude(id=d_id).count()
            if count > 0:
                raise ValidationError("文章名存在")


            document.title = d_title
            document.content = d_content
            document.htmlcontent = d_htmlcontent
            document.wordcount = word_count
            document.type = type
            document.sentiment = sentiment
            document.save()
        else:

            folder = Folder.objects.get(id=folder_id)

            count = Document.objects.filter(title=d_title, folder=folder).count()
            if count > 0:
                raise ValidationError("文章名存在")


            Document.objects.create(title=d_title,content=d_content,wordcount=word_count,user=request.user,folder=folder,htmlcontent=d_htmlcontent,type=type,sentiment=sentiment)


        return Response({"status_code": status.HTTP_200_OK,
                             "message": "ok",
                             "results": [],
                             }, status=status.HTTP_200_OK)


class DocumentDetailViewSet(APIView):
    """ 获取文章详情"""

    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)
    permission_classes = (IsAuthenticated,)
    def get(self,request,*args,**kwargs):
        id = kwargs.get("id",None)

        if id is None:
            raise ValidationError("参数不全")

        try:
            document = Document.objects.get(id=id)
        except Exception as e:
            print(e)
            raise ValidationError("没有该文章")

        if document.user != request.user:
            raise ValidationError("找不到该文章")

        serializer = DocumentSerializer(instance=document)

        return Response({"status_code": status.HTTP_200_OK,
                             "message": "ok",
                             "results": serializer.data,
                             }, status=status.HTTP_200_OK)


    def delete(self,request,*args,**kwargs):
        id = kwargs.get("id",None)

        # 校验参数
        try:
            document = Document.objects.get(id=id)
        except Exception as e:
            print(e)
            raise ValidationError("找不到该文章")

        if document.user != request.user:

            raise ValidationError("找不到该文章")

        document.delete()
        request.user.document_count -= 1
        request.user.save()

        return Response({"status_code": status.HTTP_200_OK,
                             "message": "ok",
                             "results": [],
                             }, status=status.HTTP_200_OK)


class DocumentListViewSet(APIView):
    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)
    permission_classes = (IsAuthenticated,)
    def get(self,request,*args,**kwargs):

        # 获取参数
        folder_id = kwargs.get("id",0)

        try:
            if folder_id == 0:
                folder = Folder.objects.filter(user=request.user,name="默认文件夹")
            else:
                folder = Folder.objects.get(id=folder_id)
        except Exception as e:
            raise ValidationError("文件夹不存在")

        if folder_id is None:
            raise ValidationError("参数不全")

        if folder.user != request.user:
            raise ValidationError("文件夹不存在")

        p1 = P1()
        document = Document.objects.filter(folder=folder)

        document_count = len(document)

        page_list = p1.paginate_queryset(queryset=document, request=request, view=self)

        document = DocumentSerializer(instance=page_list, many=True)

        return Response({"status_code": status.HTTP_200_OK,
                             "message": "ok",
                             "results": {"documents":document.data,"count":document_count},
                             }, status=status.HTTP_200_OK)


class DownloadDocumenttxt(APIView):
    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)
    permission_classes = (IsAuthenticated,)
    def post(self,request,*args,**kwargs):

        response = etree.HTML(text=request.data.get("content"))
        txt = response.xpath('string(.)').strip()

        # 时间字符串
        timestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


        # with open("media/download/" + timestr +".txt", "w+",encoding="utf-8") as f:
        #     f.write(txt)
        #
        # # 获取今天日期
        # data = datetime.datetime.now()
        # # 写入记录
        # try:
        #     data_count = AdminCountModel.objects.get(data=data)
        #     data_count.download_count += 1
        #     data_count.save()
        # except Exception as e:
        #     AdminCountModel.objects.create(download_count=1)

        # 这里其实就是去html给前端就好
        return Response({"status_code": status.HTTP_200_OK,
                             "message": "ok",
                             "results": txt,
                             }, status=status.HTTP_200_OK)


class DownloadDocumentPDF(APIView):
    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)
    permission_classes = (IsAuthenticated,)
    def post(self,requset,*args,**kwargs):
        content = requset.data.get('content')

        # 时间字符串
        timestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        options = {
            'encoding': "utf-8",
            'page-size': 'A4',
            'margin-top': '0mm',
            'margin-right': '0mm',
            'margin-bottom': '0mm',
            'margin-left': '0mm'
        }
        path_wk = BASE_DIR + "\wkhtmltopdf.exe"  # 安装位置
        config = pdfkit.configuration(wkhtmltopdf=path_wk)

        pdfkit.from_string(content, "media/download/" + timestr + '.pdf',options=options,configuration=config)

        requset.user.download_count += 1
        requset.user.save()

        # 获取今天日期
        data = datetime.datetime.now()
        # 写入记录
        try:
            data_count = AdminCountModel.objects.get(data=data)
            data_count.download_count += 1
            data_count.save()
        except Exception as e:
            AdminCountModel.objects.create(download_count=1)


        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": '/media/download/' + timestr + '.pdf',
                         }, status=status.HTTP_200_OK)


class DocumentLastViewset(mixins.ListModelMixin,viewsets.GenericViewSet,mixins.RetrieveModelMixin):
    """获取最近的文章"""
    serializer_class = DocumentSerializer
    pagination_class = P1
    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)
    permission_classes = (IsAuthenticated,)
    def get_queryset(self):
        return Document.objects.filter(user=self.request.user).all().order_by("-create_time")

    def retrieve(self, request, *args, **kwargs):
        """搜索"""
        search = request.GET.get("search"," ")

        if search != " ":
            documents = Document.objects.filter(user=self.request.user).filter(Q(title__contains=search) | Q(content__contains=search)).all().order_by("-create_time")

            p1 = P1()

            page_list = p1.paginate_queryset(queryset=documents, request=request, view=self)

            serializer = DocumentSerializer(instance=page_list, many=True)
            documents_data = serializer.data

            count = len(documents)
        else:
            documents_data = []
            count = 0

        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results":documents_data,
                         }, status=status.HTTP_200_OK)


class Changedocument(APIView):
    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)
    permission_classes = (IsAuthenticated,)
    """保存文件夹"""
    def put(self,request,*args,**kwargs):
        folder_id = request.data.get('folder')

        document_id = request.data.get('document')


        try:
            folder = Folder.objects.get(id=int(folder_id))
        except:
            raise ValidationError({'error': ['找不到文件夹']})

        try:
            document = Document.objects.get(id=int(document_id))
        except:
            raise ValidationError({'error': ['找不到文章']})

        if document.folder.id == folder_id:
            return Response({"status_code": status.HTTP_200_OK,
                             "message": "error",
                             "results": '不用修改',
                             }, status=status.HTTP_200_OK)
        else:
            document.folder = folder
            document.save()


        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": [],
                         }, status=status.HTTP_200_OK)










