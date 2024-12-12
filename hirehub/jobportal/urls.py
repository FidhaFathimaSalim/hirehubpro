from django.urls import path,include
from . import views

urlpatterns = [
    path('', views.work,name='work'),
    path('recommend_jobs/', views.recommend_jobs, name='recommend_jobs'),
    path('hirebot/', views.hirebot, name='hirebot'),

]