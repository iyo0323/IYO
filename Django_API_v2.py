
#----------------------------------------------------------------------------------
# Def
#----------------------------------------------------------------------------------
#{aaa}: Objectのコードネーム（ex. customer）
#{Aaa}: Objectのコードネーム（ex. Customer）
#{AppName}: APPの名称（ex. customermanage）
#{Table_Name}: DB上Tableの物理名（ex. master_customer）
#{TableField}: DB上Fieldの物理名（ex. birthday, sex ... etc）



#----------------------------------------------------------------------------------
# {AppName}\apps.py
#----------------------------------------------------------------------------------
from django.apps import AppConfig
class {AppName}Config(AppConfig):
    name = '{AppName}'



#----------------------------------------------------------------------------------
# {AppName}\urls.py
#----------------------------------------------------------------------------------
from django.conf.urls import url
from {AppName} import views
from rest_framework import routers

router = routers.DefaultRouter()
router.register(r'{aaa}s', views.{Aaa}ViewSet)



#----------------------------------------------------------------------------------
# {AppName}\views\__init__.py
#----------------------------------------------------------------------------------
from .{aaa} import {Aaa}ViewSet




#----------------------------------------------------------------------------------
# {AppName}\views\base.py
#----------------------------------------------------------------------------------
from .mixin import {Bbb}CreateModelMixin
from .mixin import {Bbb}UpdateModelMixin
from rest_framework import status, mixins
from rest_framework.viewsets import GenericViewSet



class {Bbb}ModelViewSet({Bbb}CreateModelMixin,
                       mixins.RetrieveModelMixin,
                       {Bbb}UpdateModelMixin,
                       mixins.DestroyModelMixin,
                       mixins.ListModelMixin,
                       GenericViewSet):
    """
    A viewset that provides default `create()`, `retrieve()`, `update()`,
    `partial_update()`, `destroy()` and `list()` actions.
    """

    pass



#----------------------------------------------------------------------------------
# {AppName}\views\mixins.py
#----------------------------------------------------------------------------------








