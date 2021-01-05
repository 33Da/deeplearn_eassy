from django.db import models
from datetime import datetime
from apps.userprofile.models import UserProfile

# Create your models here.

class Folder(models.Model):
    """
    文件夹
    """
    name = models.CharField(max_length=120, verbose_name="文件夹名字", help_text="文件夹名字",)

    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE,verbose_name="拥有者",help_text="拥有者")
    create_time = models.DateTimeField(verbose_name="添加时间", default=datetime.now, help_text="添加时间")
    last_time = models.DateTimeField(verbose_name="最新时间", auto_now=True, help_text="最新时间")

    class Meta:
        verbose_name = "文件夹"
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.name



