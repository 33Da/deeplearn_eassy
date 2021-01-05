from . import views
from django.urls import path

document_urls = views.DocumentViewSet.as_view()

documentdetail_urls = views.DocumentDetailViewSet.as_view()

documentlist_urls = views.DocumentListViewSet.as_view()

downloadtxt_url = views.DownloadDocumenttxt.as_view()

documentpdf_urls = views.DownloadDocumentPDF.as_view()

document_last_urls = views.DocumentLastViewset.as_view({
    "get":"list"
})

document_find_urls = views.DocumentLastViewset.as_view({
    "get":"retrieve"
})

document_change_urls = views.Changedocument.as_view()
urlpatterns = [
    # 文章逻辑
    path("document/", document_urls),
    path("document/<int:id>/", documentdetail_urls),
    path("documentlist/<int:id>/",documentlist_urls),
    path("document/last/",document_last_urls),
    path("document/search/", document_find_urls),
    path("document/folder/change/", document_change_urls),

    path("downloadtxt/",downloadtxt_url),

    path("downloadpdf/",documentpdf_urls)


]
