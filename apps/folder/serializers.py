from .models import Folder
from rest_framework import serializers


class FolderSerializer(serializers.ModelSerializer):
    """
    添加用户序列化类
    """


    document = serializers.SerializerMethodField(read_only=True)


    def get_document(self,attr):
        date = []
        for document in attr.document.all():
            last_time = document.last_time.strftime("%Y-%m-%d %H:%M:%S")
            date.append({"id":document.id,"title":document.title,'wordcount':document.wordcount,"uploadtype":document.get_upload_type_display(),"uploadtime":last_time})
        return {"count":len(date),"data":date}

    class Meta:
        model = Folder
        fields = '__all__'