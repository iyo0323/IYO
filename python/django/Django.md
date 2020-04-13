[公式学習サイト](https://docs.djangoproject.com/ja/2.0/intro/)



Django Root
```sh
$ cd C:\Users\{username}\AppData\Local\Continuum\anaconda3\Scripts
```


Create Project（ProjectName=mysite）
```sh
$ django-admin startproject mysite
```


Create App（AppName=polls）
```sh
$ python manage.py startapp polls
```



Create Migration File
```sh
$ python manage.py makemigrations polls
```

Show the detail of Migration File (SQL)
```sh
$ python manage.py sqlmigrate polls 0001
```

Do Migrate (Create Table)
```sh
$ python manage.py migrate
```

Run Django by Shell Mode
```sh
$ python manage.py shell
```



-- router
mysite.urls.py 
	→　polls.urls.py 
		→ polls.views.py 
			→ pools.templates.polls.{index}.html



Create Superuser
```sh
$ python manage.py createsuperuser
{username}
{email}
{password}
```




Do Test
```sh
$ python manage.py test polls
```


Building Package
```sh
$ python setup.py sdist (run from inside django-polls)
```

Install Package
```sh
$ pip install --user django-polls/dist/django-polls-0.1.tar.gz
```

Uninstall Package
```sh
$ pip uninstall django-polls
```

Server起動
```sh
$ python manage.py runserver
```

http://127.0.0.1:8000/admin/

http://localhost:8000/polls/


Integrating Django with a legacy database (Create Model from DB Table)
```sh
$ python manage.py inspectdb > models.py --include-views
```

[Django migrate コマンドで、SQLを直接実行する](https://www.monotalk.xyz/blog/django-migrate-%E3%82%B3%E3%83%9E%E3%83%B3%E3%83%89%E3%81%A7insert-%E6%96%87%E3%82%92%E5%AE%9F%E8%A1%8C%E3%81%99%E3%82%8B/
)
