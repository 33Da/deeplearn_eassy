import time

from django.shortcuts import render
from lxml import etree
from rest_framework.exceptions import ValidationError
from rest_framework.views import APIView
from translate import Translator
from rest_framework.response import Response
from rest_framework import status
import paddlehub as hub
from apps.document.models import Document
from apps.folder.models import Folder
from rest_framework_jwt.authentication import JSONWebTokenAuthentication
from rest_framework import authentication
from rest_framework.permissions import IsAuthenticated
from aip import AipSpeech, AipNlp
import datetime
from .models import AdminCountModel



class TranslatViewset(APIView):
    """翻译"""
    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)

    def post(self, request, *args, **kwargs):
        word = request.data.get("word")

        # 0 英-》中  1 中 -》英
        type = request.data.get("type", 0)



        if int(type) is 0:
            translator = Translator(to_lang="chinese")
            translation = translator.translate(word)
            print(translation)
        elif int(type) is 1:
            translator = Translator(from_lang="chinese", to_lang="english")
            translation = translator.translate(word)
            print(translation)
        else:
            raise ValidationError("参数错误")

        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": {"result": translation},
                         }, status=status.HTTP_200_OK)

        request.user.translate_count += 1
        request.user.save()

        # 获取今天日期
        data = datetime.datetime.now()
        # 写入记录
        try:
            data_count = AdminCountModel.objects.get(data=data)
            data_count.translate_count += 1
            data_count.save()
        except Exception as e:
            AdminCountModel.objects.create(translate_count=1)

        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": {"result": translation},
                         }, status=status.HTTP_200_OK)


class EntitynamingViewset(APIView):
    """实体类命名"""

    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)

    def post(self, request, *args, **kwargs):

        d_id = request.data.get("id",0)

        if d_id != 0:
            try:
                document = Document.objects.get(id=d_id)
                content = document.content

            except Exception as e:
                raise ValidationError("文章不存在")
        else:
            # 去除标签存文本内容
            response = etree.HTML(text=request.data.get("content"))
            content = response.xpath('string(.)').strip()

        type = request.data.get("type",None)

        lac = hub.Module(name='lac')
        test_text = [content]

        inputs = {"text": test_text}
        results = lac.lexical_analysis(data=inputs)[0]

        result_dict = {
            "n": set(),  # 名词
            "r": set(),  # 代词
            "v": set(),  # 动词
            "a": set(),  # 形容词
            "d": set(),  # 副词
            "PER": set(),  # 人名
            "LOC": set(),  # 地名
            "nw": set(),  # 作品名
            "f": set(),  # 方位名词
            "s": set(),  # 处所名
            "ORG": set(),  # 机构名
            "TIME": set(),  # 时间

        }
        for i in range(len(results['word'])):
            if results['tag'][i] in result_dict.keys():
                result_dict[results['tag'][i]].add(results['word'][i])

        for k in list(result_dict.keys()):
            result_dict[k + "_count"] = len(result_dict[k])

        if type != None:
            result_dict = result_dict[type]


        request.user.pickword_count += 1
        request.user.save()

        # 获取今天日期
        data = datetime.datetime.now()
        # 写入记录
        try:
            data_count = AdminCountModel.objects.get(data=data)
            data_count.entitynaming_count += 1
            data_count.save()
        except Exception as e:
            AdminCountModel.objects.create(entitynaming_count=1)

        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": result_dict,
                         }, status=status.HTTP_200_OK)


class ReadViewset(APIView):
    """
    语音读
    """
    APP_ID = '18180166'
    API_KEY = 'hitfqUFmVbR8afjpGMzO0Vp2'
    SECRET_KEY = 'LZKNpdf0z2WRVIIM18dQHRH7AZ3goLe8'

    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)

    def get(self, request, *args, **kwargs):

        id = kwargs.get("id")

        try:
            document = Document.objects.get(id=id)
        except Exception as e:
            print(e)
            raise ValidationError("文章不存在")

        if document.user != request.user:
            raise ValidationError("文章不存在")

        client = AipSpeech(self.APP_ID, self.API_KEY, self.SECRET_KEY)

        result = client.synthesis(document.content,
                                  'zh', 1, {
                                      'vol': 5, 'per': 4
                                  })

        # 时间字符串
        timestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # 识别正确返回语音二进制 错误则返回dict 参照下面错误码
        if not isinstance(result, dict):
            with open("media/mp3/" + timestr + '.mp3', 'wb') as f:
                f.write(result)

        request.user.read_count += 1
        request.user.save()

        # 获取今天日期
        data = datetime.datetime.now()

        # 写入记录
        try:
            data_count = AdminCountModel.objects.get(data=data)
            data_count.read_count += 1
            data_count.save()
        except Exception as e:
            AdminCountModel.objects.create(read_count=1)

        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": [{"result": "/media/mp3/" + timestr + ".mp3"}],
                         }, status=status.HTTP_200_OK)


class CheckViewset(APIView):
    """文章审核"""

    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)

    def post(self, request, *args, **kwargs):
        APP_ID = '18786667'
        API_KEY = 'sVHpAWkQSR67jI0eb0ZFhrAL'
        SECRET_KEY = 'QocG018D51lWDtyzlEF5FiFnUtnPxkkp '

        # 去除标签存文本内容
        response = etree.HTML(text=request.data.get("content"))
        content = response.xpath('string(.)').strip()

        client = AipNlp(APP_ID, API_KEY, SECRET_KEY)


        result = client.ecnet(content)

        if result.get("error_code") == 282131:  # 文章过长
            result_list = []
            count = len(content)//200
            for i in range(0,count+1):
                time.sleep(1)
                res = client.ecnet(content[i*200:i*200+200])
                result_list.append(res)

            result = result_list[0]
            for r in result_list:
                print(result)
                result['item']['vec_fragment'] += r['item']['vec_fragment']


        request.user.checkdocument_count += 1
        request.user.save()

        # 获取今天日期
        data = datetime.datetime.now()
        # 写入记录
        try:
            data_count = AdminCountModel.objects.get(data=data)
            data_count.checkdocument_count += 1
            data_count.save()
        except Exception as e:
            AdminCountModel.objects.create(checkdocument_count=1)

        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": {"result": result},
                         }, status=status.HTTP_200_OK)


class TypeViewset(APIView):
    """自动分类"""

    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)

    def post(self, request, *args, **kwargs):
        APP_ID = '18786667'
        API_KEY = 'sVHpAWkQSR67jI0eb0ZFhrAL'
        SECRET_KEY = 'QocG018D51lWDtyzlEF5FiFnUtnPxkkp '

        # 去除标签存文本内容
        response = etree.HTML(text=request.data.get("content"))
        content = response.xpath('string(.)').strip()

        title = request.data.get("title")

        client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

        result = client.topic(title, content)

        types = ''
        for k, v in result["item"].items():
            if len(v) != 0:
                types = types + v[0]['tag'] + ","

        request.user.type_count += 1
        request.user.save()

        # 获取今天日期
        data = datetime.datetime.now()
        # 写入记录
        try:
            data_count = AdminCountModel.objects.get(data=data)
            data_count.type_count += 1
            data_count.save()
        except Exception as e:
            AdminCountModel.objects.create(type_count=1)

        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": types,
                         }, status=status.HTTP_200_OK)


class SentimentViewset(APIView):
    """情感倾向"""

    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)

    def post(self, request, *args, **kwargs):
        APP_ID = '18786667'
        API_KEY = 'sVHpAWkQSR67jI0eb0ZFhrAL'
        SECRET_KEY = 'QocG018D51lWDtyzlEF5FiFnUtnPxkkp '

        response = etree.HTML(text=request.data.get("content"))
        content = response.xpath('string(.)').strip()

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

        request.user.sentiment_count += 1
        request.user.save()

        # 获取今天日期
        data = datetime.datetime.now()
        # 写入记录
        try:
            data_count = AdminCountModel.objects.get(data=data)
            data_count.sentiment_count += 1
            data_count.save()
        except Exception as e:
            AdminCountModel.objects.create(sentiment_count=1)

        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": sentiment,
                         }, status=status.HTTP_200_OK)


class SummaryViewset(APIView):
    """摘要"""

    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)

    def post(self, request, *args, **kwargs):
        APP_ID = '18786667'
        API_KEY = 'sVHpAWkQSR67jI0eb0ZFhrAL'
        SECRET_KEY = 'QocG018D51lWDtyzlEF5FiFnUtnPxkkp '

        # 去除标签存文本内容
        response = etree.HTML(text=request.data.get("content"))
        content = response.xpath('string(.)').strip()

        client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

        result = client.newsSummary(content, 200)

        request.user.summary_count += 1
        request.user.save()

        # 获取今天日期
        data = datetime.datetime.now()
        # 写入记录
        try:
            data_count = AdminCountModel.objects.get(data=data)
            data_count.summary_count += 1
            data_count.save()
        except Exception as e:
            AdminCountModel.objects.create(summary_count=1)

        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": {"result": result},
                         }, status=status.HTTP_200_OK)


class DocumentsViewset(APIView):
    """给前端图表分析的，返回用户所有文章的情况"""
    authentication_classes = (JSONWebTokenAuthentication, authentication.SessionAuthentication)
    permission_classes = (IsAuthenticated,)
    def get(self,request,*args,**kwargs):

        documenttype= {}
        sentiments = {}
        documents = Document.objects.filter(user=request.user).all()
        for document in documents:
            # 种类
            if document.type != None:
                types = [t for t in document.type.split(',') if t != '']
                for t in types:
                    try:
                        documenttype[t] += 1
                    except:
                        documenttype[t] = 1
            # 情感
            if document.sentiment != None:
                try:
                    sentiments[document.sentiment] += 1
                except:
                    sentiments[document.sentiment] = 1

        documentcount = {}
        wordcount = {}
        folders = Folder.objects.filter(user=request.user).all()
        for folder in folders:
            documents = folder.document.all()
            if len(documents) == 0:
                wordcount[folder.name] = 0
                documentcount[folder.name] = 0
            else:
                for d in documents:
                    # 字数
                    try:
                        wordcount[folder.name] += d.wordcount
                    except:
                        wordcount[folder.name] = d.wordcount

                    # 文章数
                    try:
                        documentcount[folder.name] += 1
                    except:
                        documentcount[folder.name] = 1



        results = {
            "documentcount":{
                'x':documentcount.keys(),
                'y': documentcount.values(),
            },
            'wordcount':{
                'x': wordcount.keys(),
                'y': wordcount.values(),
            },
            'sentiments':{
                'x': sentiments.keys(),
                'y': sentiments.values(),
                'value':[{'value':v,'name':k} for k,v in sentiments.items()]
            },
            'documenttype':{
                'x': documenttype.keys(),
                'y': documenttype.values(),
            }
        }


        return Response({"status_code": status.HTTP_200_OK,
                         "message": "ok",
                         "results": results,
                         }, status=status.HTTP_200_OK)
