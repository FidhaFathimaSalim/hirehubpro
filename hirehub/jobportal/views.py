from django.shortcuts import render
from django.http import HttpResponse
from .apps import JobportalConfig

# Create your views here.
def work(request):
    return render(request,'jobportal\work.html')
def recommend_jobs(request):
    return render(request, 'recommend_jobs.html')
def hirebot(request):
    return render(request, 'hirebot.html')
