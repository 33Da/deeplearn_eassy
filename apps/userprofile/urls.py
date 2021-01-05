from . import views
from django.urls import path


user_urls = views.UserViewset.as_view({
    "put": "update",
    "get": "retrieve",
})

register = views.RegisterViewSet.as_view()

# 头像
head_pic = views.HeadPicViewSet.as_view()


# 验证码
create_vaildcode = views.VaildcodeViewSet.as_view()

# 密码
password_urls = views.PasswordViewset.as_view({
    "put":"update"

})


urlpatterns = [
    # 用户逻辑
    path("user/", user_urls),
    path("register/",register),
    path("vaildcode/",create_vaildcode),
    path("headpic/",head_pic),
    path("password/",password_urls)

]
