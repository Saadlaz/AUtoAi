# core/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('visualize/', views.visualize, name='visualize'),
    path('preprocess/', views.preprocess, name='preprocess'),
    path('classify/', views.classify, name='classify'),
    path('predict/', views.predict, name='predict'),
]

