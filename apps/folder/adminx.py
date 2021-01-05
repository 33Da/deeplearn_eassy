import xadmin
from xadmin import views
from .models import Folder



class FolderAdmin():
    list_display = ['name', 'user', 'create_time', 'last_time']
    search_fields = ['name']
    list_filter = ['name', 'user', 'create_time', 'last_time']
    list_per_page = 10
    show_bookmarks = False  # 不显示书签
    readonly_fields = ('user',"create_time","last_time")
    list_editable = ['name']  # 在列表页可修改


xadmin.site.register(Folder, FolderAdmin)

