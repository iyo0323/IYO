
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
from rest_framework import status, mixins
from rest_framework.response import Response
from dateutil import parser
from logging import getLogger
logger = getLogger(__name__)


class GetRequestInfoMixin():
    """
    Get Request User ID
    """

    def get_user_id(self):
        user = self.request.user

        if user:
            return user.operator_control_key

        return None

    def last_updated_dt(self):
        data = self.request.data

        if data and 'last_updated_dt' in data:
            return parser.parse(data['last_updated_dt'])

        return None


class {Bbb}CreateModelMixin(mixins.CreateModelMixin, GetRequestInfoMixin):
    """
    Create a model instance.
    """

    def create(self, request, *args, **kwargs):
        user_id = self.get_user_id()
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.validated_data["created_by_id"] = user_id
        serializer.validated_data["last_updated_by_id"] = user_id
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


class {Bbb}UpdateModelMixin(mixins.UpdateModelMixin, GetRequestInfoMixin):
    """
    Update a model instance.
    """

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        user_id = self.get_user_id()
        request_update_dt = self.last_updated_dt()
        logger.debug(request_update_dt.timestamp())
        logger.debug(instance.last_updated_dt.timestamp())
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        serializer.validated_data["last_updated_by_id"] = user_id
        serializer.validated_data["last_updated_dt"] = request_update_dt
        self.perform_update(serializer)

        if getattr(instance, '_prefetched_objects_cache', None):
            # If 'prefetch_related' has been applied to a queryset, we need to
            # forcibly invalidate the prefetch cache on the instance.
            instance._prefetched_objects_cache = {}

        return Response(serializer.data)



#----------------------------------------------------------------------------------
# {AppName}\views\{aaa}.py
#----------------------------------------------------------------------------------
from django_filters import rest_framework as filters
from rest_framework import serializers
from .bases import {Bbb}ModelViewSet
from {AppName}.models import {Aaa}
from logging import getLogger
logger = getLogger(__name__)


class {Aaa}Serializer(serializers.ModelSerializer):

    class Meta:
        model = {Aaa}
        fields = '__all__'
        read_only_fields = (
            'created_dt',
            'last_updated_dt',
        )


class {Aaa}Filter(filters.FilterSet):
    # Field Look up してぶん回す、型に応じたFilter 作る
    filter_1 = filters.CharFilter(lookup_expr='contains')
    filter_2 = filters.CharFilter(lookup_expr='contains')

    class Meta:
        model = {Aaa}
        fields = ['filter_1', 'filter_2']


class {Aaa}ViewSet({Bbb}ModelViewSet):
    queryset = {Aaa}.objects.all()
    serializer_class = {Aaa}Serializer
    filter_class = {Aaa}Filter





