
#----------------------------------------------------------------------------------
# Def
#----------------------------------------------------------------------------------
#{aaa}: Objectのコードネーム（ex. customer）
#{Aaa}: Objectのコードネーム（ex. Customer）
#{AppName}: APPの名称（ex. customermanage）
#{Table_Name}: DB上Tableの物理名（ex. master_customer）
#{TableField}: DB上Fieldの物理名（ex. birthday, sex ... etc）


#----------------------------------------------------------------------------------
# {AppName}\urls.py
#----------------------------------------------------------------------------------
from django.urls import path
from {AppName} import views
urlpatterns = [
    # 全件データ取得（Filterの指定で条件の設定も可能）
    path('{aaa}s', views.{Aaa}List.as_view()),
    # 一件データ取得
    path('{aaa}/<int:pk>', views.{Aaa}.as_view()),
    # 一件データ更新
    path('{aaa}/<int:pk>/update/', views.{Aaa}Update.as_view()),
]



#----------------------------------------------------------------------------------
# {AppName}\views\XXX.py
#----------------------------------------------------------------------------------
from rest_framework import generics
from django_filters import rest_framework as filters
from {AppName}.serializer import {Aaa}Serializer
import django_filters

class {Aaa}IdFilter(filters.FilterSet):
    # id　→　DB上の実体Field名
    # {aaa}_id　→　Django側で自命名の名前
    # urlの最後に「?{aaa}_id=1」と指定すれば、id=1のデータだけ抽出になる
    {aaa}_id = django_filters.NumberFilter(field_name="id", lookup_expr='exact')

    class Meta:
        model = {Table_Name}
        fields = ['{aaa}_id']


class {Aaa}List(generics.ListCreateAPIView):
    """
    REST API (Select List. You need to set filter)
    """
    queryset = {Table_Name}.objects.all()
    serializer_class = {Aaa}Serializer

    filter_class = {Aaa}IdFilter
    # permission_classes = [IsAdminUser]


class {Aaa}(generics.RetrieveAPIView):
    """
    REST API (Select One Entry. You need to set <pk> in urls file)
    """
    queryset = {Table_Name}.objects.all()
    serializer_class = {Aaa}Serializer


class {Aaa}Update(generics.RetrieveUpdateAPIView):
    """
    REST API (Update. You need to set <pk> in urls file)
    """
    queryset = {Table_Name}.objects.all()
    serializer_class = {Aaa}Serializer



#----------------------------------------------------------------------------------
# {AppName}\serializer.py
#----------------------------------------------------------------------------------
from rest_framework import serializers
from {AppName}.models import {Table_Name}

class {Aaa}Serializer(serializers.ModelSerializer):
    class Meta:
        model = {Table_Name}
        fields = (
            'id',
            '{TableField1}',
            '{TableField2}',
            '{TableField3}',
        )

