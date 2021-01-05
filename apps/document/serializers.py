from .models import Document
from rest_framework import serializers



class DocumentSerializer(serializers.ModelSerializer):
    folder_name = serializers.CharField(source="folder.name")

    class Meta:
        model = Document  #模型类

        fields = ["id","title","content","htmlcontent",'starlevel',"wordcount","folder","folder_name","create_time","last_time"]

