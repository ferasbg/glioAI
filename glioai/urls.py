
from django.contrib import admin
from django.urls import path, url
from django.contrib import admin
from glioai import views 
import glioai.urls


urlpatterns = [
    path('classify/', views.call_model.as_view())
    path('admin/', admin.site.urls),
    path('index/', views.index, name='mainview')
    path('glioai/', glioai.urls)
]