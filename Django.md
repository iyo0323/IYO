[公式学習サイト](https://docs.djangoproject.com/ja/2.0/intro/)



DjangoのRoot
```sh
$ cd C:\Users\{username}\AppData\Local\Continuum\anaconda3\Scripts
```


Project作成（ProjectName=mysite）
```sh
$ django-admin startproject mysite
```


Module作成（ModuleName=polls）
```sh
$ python manage.py startapp polls
```



Migrationファイル作成
```sh
$ python manage.py makemigrations polls
```

Migrationファイルの中身をSQLで表示（表示のみ）
```sh
$ python manage.py sqlmigrate polls 0001
```

Table実作成
```sh
$ python manage.py migrate
```

ShellでDjangoをいじる
```sh
$ python manage.py shell
```



-- router
mysite.urls.py 
	→　polls.urls.py 
		→ polls.views.py 
			→ pools.templates.polls.{index}.html




$ python manage.py createsuperuser
{username}
{email}
{password}



-- Do Test
$ python manage.py test polls



-- building package
$ python setup.py sdist (run from inside django-polls)
-- install package
$ pip install --user django-polls/dist/django-polls-0.1.tar.gz
-- uninstall package
$ pip uninstall django-polls


-- Server起動
$ python manage.py runserver

http://127.0.0.1:8000/admin/
http://localhost:8000/polls/


--Integrating Django with a legacy database（既存DBからModelを作る）
$ python manage.py inspectdb > models.py --include-views

Django migrate コマンドで、SQLを直接実行する
https://www.monotalk.xyz/blog/django-migrate-%E3%82%B3%E3%83%9E%E3%83%B3%E3%83%89%E3%81%A7insert-%E6%96%87%E3%82%92%E5%AE%9F%E8%A1%8C%E3%81%99%E3%82%8B/
