import xadmin
from xadmin import views
from .models import Document




class DocumentAdmin(object):

    list_display = ['title', 'user', 'folder', 'upload_type','translate_count','pickword_count','wordcount','create_time','last_time']
    search_fields = ['title']
    list_filter = ['title', 'user', 'folder', 'upload_type','translate_count','pickword_count','wordcount','create_time','last_time']
    list_per_page = 10
    show_bookmarks = False      # 不显示书签
    readonly_fields = ('user', "upload_type","content","translate_count", "wordcount","pickword_count","create_time","last_time")
    list_editable = ['title']  # 在列表页可修改
    style_fields = {"content": "ueditor"}







xadmin.site.register(Document, DocumentAdmin)
