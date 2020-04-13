Install Sphinx
```sh
$ pip install -U Sphinx
```

Create Project
```sh
$ sphinx-quickstart
```

Build Html Files
```sh
$ make html
```

Install Theme
```sh
$ easy_install sphinxjp.themes.sphinx_rtd_theme
$ pip install sphinx_rtd_theme
```

[Read the Docs Sphinx Theme](https://sphinx-rtd-theme.readthedocs.io/en/latest/demo/demo.html)


Themeの使用設定

conf.py　→　html_theme = 'sphinx_rtd_theme'

conf.py　→　

import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]


CSS custmize
conf.py　→　 
def setup(app):
    app.add_stylesheet('custom.css')


Pygments
```sh
$ pip install Pygments
```

-- Html Icon
.. raw:: html

   <i class="fa fa-inbox"></i>

.. |inbox| raw:: html

   <i class="fa fa-inbox"></i>

oo |inbox| kk

-- Replace
.. |ex1| replace:: 例1


Code with Sidebar
-----------------

.. sidebar:: A code example

    With a sidebar on the right.2





[mermaid.js](https://ryuta46.com/516)

.. mermaid::

    graph TB
    
        open(Request)
        
        subgraph 初期処理
         init1(確認)
         init2(CSV)
         init3(明細)
        end
        subgraph 繰り返しコミット
         regist1(判定)
         regist2(更新項目)
         regist3(更新相関)
        end
		
        close(Response)
        
        open-->init1
        
        init1-->init2
        init2-->init3
        init3-->regist1
        
        regist1-->regist2
        regist2-->regist3
        regist3-->close
