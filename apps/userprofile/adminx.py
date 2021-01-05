import xadmin
from xadmin import views
from .models import UserProfile
from django.contrib.admin.models import LogEntry


class BaseSetting(object):
    enable_themes = True
    use_booswatch = True


class GlobalSettings(object):
    site_title = '文章分析平台'
    site_footer = '文章分析平台'
    menu_style = 'accordion'




class UserAdmin:
    list_display = ['username', 'email', 'create_time','last_login',"is_superuser",'upload_count','download_count','knowpic_count']
    search_fields = ['username', 'email']
    list_filter = ['username', 'email', 'create_time','last_login',"is_superuser",'upload_count','download_count','knowpic_count']
    list_per_page = 10
    show_bookmarks = False  # 不显示书签
    readonly_fields = ('email', "translate_count",  "pickword_count","create_time",'last_login','upload_count','download_count','knowpic_count','checkdocument_count','type_count','sentiment_count','summary_count')
    exclude = ['is_staff',"is_active","first_name","last_name","last_time","password","last_login","is_superuser","groups","user_permissions"]  # 修改时隐藏的字段
    list_editable = ['username',"is_superuser"]  # 在列表页可修改





xadmin.site.unregister(UserProfile)
xadmin.site.register(UserProfile, UserAdmin)

# 全局设置
xadmin.site.register(views.BaseAdminView, BaseSetting)
xadmin.site.register(views.CommAdminView, GlobalSettings)



# 隐藏授权和管理
from django.contrib.auth.models import Group,Permission
xadmin.site.unregister(Group)
xadmin.site.unregister(Permission)
