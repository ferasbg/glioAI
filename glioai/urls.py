
from django.contrib import admin
from django.urls import path
from django.conf.urls import include
from django.conf.urls import url
from django import views
from django.contrib import admin


urlpatterns = [
    path('classify/', views.call_model.as_view())
]