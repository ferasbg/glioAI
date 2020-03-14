
from django.contrib import admin
from django.urls import path, url
from django.contrib import admin
from django import views



urlpatterns = [
    path('detect/', views.tumor_prediction.as_view()),
    path('admin/', admin.site.urls),
    path('index/', views.index, name='mainview')
]