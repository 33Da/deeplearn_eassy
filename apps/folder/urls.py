from . import views
from django.urls import path

folder_urls = views.FolderViewSet.as_view({
    "post":"create",
    "get":"list",
    "put": "update",
})

folder_detail_urls = views.FolderViewSet.as_view({
    "delete":"destroy",
    "get":"retrieve",
})


urlpatterns = [

    path("folder/", folder_urls),
    path("folder/<int:pk>/", folder_detail_urls),


]
