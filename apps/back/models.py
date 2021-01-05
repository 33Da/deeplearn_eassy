from django.db import models

# Create your models here.
import datetime

class AdminCountModel(models.Model):
    # 日期
    data = models.DateField(verbose_name="时间", auto_now_add=True, help_text="时间")

    # 上传次数
    upload_count = models.IntegerField(default=0,verbose_name="上传次数",help_text="上传次数")

    # 下载量
    download_count = models.IntegerField(default=0,verbose_name="下载次数",help_text="下载次数")

    # 使用识别次数
    knowpic_count = models.IntegerField(default=0,verbose_name="使用识别次数",help_text="使用识别次数")

    # 调用语音功能次数
    # read_count = models.IntegerField(default=0,verbose_name="语音功能",help_text="语音功能")

    # 调用翻译功能次数
    translate_count = models.IntegerField(default=0,verbose_name="翻译功能",help_text="翻译功能")

    # 调用实体命名次数
    entitynaming_count = models.IntegerField(default=0,verbose_name="实体命名",help_text="实体命名")

    #文章审核
    checkdocument_count = models.IntegerField(default=0,verbose_name="文章审核",help_text="文章审核")

    # 自动分类
    type_count = models.IntegerField(default=0,verbose_name="自动分类",help_text="自动分类")

    # 情感倾向
    sentiment_count = models.IntegerField(default=0,verbose_name="情感倾向",help_text="情感倾向")

    # 摘要
    summary_count = models.IntegerField(default=0,verbose_name="摘要",help_text="摘要")



    class Meta:
        verbose_name = "功能调用统计"
        verbose_name_plural = verbose_name

    def __str__(self):
        return str(self.data)