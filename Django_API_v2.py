
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

