from . import views
from django.urls import path

translate_urls = views.TranslatViewset.as_view()


entitynaming_urls = views.EntitynamingViewset.as_view()

reading_urls = views.ReadViewset.as_view()

check_urls = views.CheckViewset.as_view()

type_urls = views.TypeViewset.as_view()

sentiment_urls = views.SentimentViewset.as_view()

summary_urls = views.SummaryViewset.as_view()

document_urls = views.DocumentsViewset.as_view()

urlpatterns = [
    # 翻译
    path("translation/", translate_urls),

    # 实体命名
    path("entityname/",entitynaming_urls),

    # 读文章
    path("readdocument/<int:id>/",reading_urls),

    # 文章审核
    path("check/",check_urls),

    # 文章种类
    path("type/",type_urls),

    # 情感倾向
    path("sentiment/", sentiment_urls),

    # 摘要
    path("summary/", summary_urls),

    # 图数据
    path("chart/", document_urls),



]
