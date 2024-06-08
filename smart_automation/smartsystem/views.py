from django.shortcuts import render
from rest_framework import viewsets
from rest_framework import status
from rest_framework.response import Response
from utils import temperature_pred
# Create your views here.
class PredTempViewset(viewsets.ViewSet):
    def list(self, request,*args, **kwargs):
        '''returns the predicted temperature for the next hour'''
        temp = temperature_pred.temp_pred()
        return Response({'temperature':temp},status=status.HTTP_200_OK)
