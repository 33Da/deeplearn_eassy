import xadmin


from .models import AdminCountModel
import datetime
from django.utils.safestring import mark_safe
from django.contrib import admin

class CountAdmin:

    data_charts = {
        "order_download_upload": {'title': '上传下载统计', "x-field": "data",
                         "y-field": ('upload_count', "download_count"),
                         "order": ('data',)},

        "knowpic_function": {'title': '识别调用分析', "x-field": "data",
                                "y-field": ('knowpic_count',),
                                "order": ('data',)},

        "order_back_function": {'title': '文章功能调用', "x-field": "data",
                        "y-field": ( "translate_count", "entitynaming_count",'checkdocument_count','type_count','sentiment_count','summary_count'),
                        "order": ('data',)},
    }
    date_hierarchy = "data"
    list_display = ['data', 'upload_count', 'download_count', 'knowpic_count',  'translate_count', 'entitynaming_count','checkdocument_count','type_count','sentiment_count','summary_count']
    search_fields = ['data']
    list_filter = ['data', 'upload_count', 'download_count', 'knowpic_count',  'translate_count', 'entitynaming_count','checkdocument_count','type_count','sentiment_count','summary_count']
    list_per_page = 10
    show_bookmarks = False  # 不显示书签
    readonly_fields = ('data', 'upload_count', 'download_count', 'knowpic_count', 'translate_count', 'entitynaming_count','checkdocument_count','type_count','sentiment_count','summary_count')
    refresh_times = [10, 60]



xadmin.site.register(AdminCountModel, CountAdmin)
