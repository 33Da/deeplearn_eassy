from django.db import models
from django.contrib.auth.models import AbstractUser
from datetime import datetime
# Create your models here.
class UserProfile(AbstractUser):
    """
    用户表
    """
    document_count = models.IntegerField(verbose_name="文章数量", help_text="文章数量", default=0)
    head_pic = models.ImageField(
        upload_to="icons",  # 指定文件保存的路径名 系统自动创建
        verbose_name = "头像",
        help_text = "头像",
        blank = True,
        null = True
    )
    create_time = models.DateTimeField(verbose_name="创建时间", help_text="创建时间", default=datetime.now)
    last_time = models.DateTimeField(verbose_name="最新时间", auto_now=True, help_text="最新时间")

    translate_count = models.IntegerField(default=0, verbose_name="调用翻译次数", help_text="调用翻译次数")


    pickword_count = models.IntegerField(default=0, verbose_name="关键词提取", help_text="关键词提取")

    # 上传次数
    upload_count = models.IntegerField(default=0, verbose_name="上传次数", help_text="上传次数")

    # 下载量
    download_count = models.IntegerField(default=0, verbose_name="下载次数", help_text="下载次数")

    # 使用识别次数
    knowpic_count = models.IntegerField(default=0, verbose_name="使用识别次数", help_text="使用识别次数")


    # 文章审核
    checkdocument_count = models.IntegerField(default=0, verbose_name="文章审核", help_text="文章审核")

    # 自动分类
    type_count = models.IntegerField(default=0, verbose_name="自动分类", help_text="自动分类")

    # 情感倾向
    sentiment_count = models.IntegerField(default=0, verbose_name="情感倾向", help_text="情感倾向")

    # 摘要
    summary_count = models.IntegerField(default=0, verbose_name="摘要", help_text="摘要")


    class Meta:
        verbose_name = '用户'
        verbose_name_plural = verbose_name


