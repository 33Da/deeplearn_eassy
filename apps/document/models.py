from django.db import models

from datetime import datetime
from apps.userprofile.models import UserProfile
from apps.folder.models import Folder

from DjangoUeditor.models import UEditorField

# Create your models here.
class Document(models.Model):
    """
    文章表
    """
    CHECK_TYPE = (
        (0, "识别上传"),
        (1, "非识别上传"),

    )

    STAR_TYPE = (
        (1, "星级1"),
        (2, "星级2"),
        (3, "星级3"),
        (4, "星级4"),
        (5, "星级5"),
    )

    title = models.CharField(max_length=100, verbose_name="文章标题", help_text="文章标题")

    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, help_text="文章所属用户", verbose_name="文章所属用户",
                             related_name="document")


    folder = models.ForeignKey(Folder, on_delete=models.CASCADE, verbose_name="所属文件夹", help_text="所属文件夹", default=1,
                               related_name="document", null=True, blank=True)

    htmlcontent = UEditorField(verbose_name="文案html内容", help_text="文案html内容", null=True, blank=True ,filePath='ueditor/file/',imagePath='ueditor/images/')

    content = models.TextField(verbose_name="文案内容", help_text="文案内容", null=True, blank=True)

    upload_type = models.IntegerField(choices=CHECK_TYPE, verbose_name="上传方式", help_text="上传方式(0,识别上传; 1,非识别上传)",
                                default=1)

    translate_count = models.IntegerField(default=0,verbose_name="调用翻译次数", help_text="调用翻译次数")

    read_count = models.IntegerField(default=0,verbose_name="调用语音次数", help_text="调用语音次数")

    pickword_count = models.IntegerField(default=0,verbose_name="调用实体命名次数", help_text="调用实体命名次数")

    starlevel = models.IntegerField(choices=STAR_TYPE, verbose_name="星级", help_text="星级(1,2,3,4,5)",
                                default=1)

    wordcount = models.IntegerField(default=0,verbose_name="文章字数", help_text="文章字数")

    type = models.CharField(default='',verbose_name="文章种类", help_text="文章种类",max_length=50)

    sentiment = models.CharField(default='',verbose_name="文章情感", help_text="文章情感",max_length=50)
    create_time = models.DateTimeField(verbose_name="添加时间", default=datetime.now, help_text="添加时间")
    last_time = models.DateTimeField(verbose_name="修改时间", auto_now=True, help_text="修改时间")



    class Meta:
        verbose_name = "文章"
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.title



