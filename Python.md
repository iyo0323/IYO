

```py
# Get an input fron CUI
#########################################################
num = int(input("Give me an integer "))
```


```py
# Sample of Function docstring
#########################################################
def function_name(x):
    '関数の註解はここに書く'
    #...
    return
help(function_name)

#Result
#########################################################
#Help on function function_name in module __main__:
#function_name(x)
#   関数の註解はここに書く
```



```py
# Use Global variable in function
#########################################################
a = 10
b = 0

def f():
    global b
    c = a*a
    b = c
f()
print(b, a)

#Result
#########################################################
#100 10
```



```py
# Pass a function as the parameter to another function
#########################################################
def func_sub():
    print("sub says Hello")

# 関数を引数でもらって実行する関数
def func_main(para):
    print("In main, ", end="")
    para()

func_sub()
func_main(func_sub)

#Result
#########################################################
#sub says Hello
#In main, sub says Hello
```






Class
=========================================================


```py
# Sample of a Class
#########################################################
# クラスの練習
class MyClass():
    # クラス変数(static)
    a = "マイクラス" # public variable
    __b = 0         # private variable
    
    # Constructor
    def __init__(self, data):
        # インスタンス変数
        self.__number = MyClass.__b
        self.mydata = data
    
    # 通し番号を表示するメソッド
    def show_number(self):
        print(self.__number)

if __name__ == "__main__":
    print("MyClass のクラス変数 a: ",MyClass.a)
```






Turtle
=========================================================

```py
# Draw a Pentagon in turtle
#########################################################
from turtle import *
n = 5
for i in range(n):
    forward(100)
    left(72)
done()
```


```py
# Draw two cycles in turtle
#########################################################
from turtle import *
t1 = Turtle()
t2 = Turtle()
t1.color('red')
t2.color('blue')
for i in range(180):
    t1.forward(5)
    t2.forward(3)
    t1.left(2)
    t2.left(2)
done()
```


```py
# Click event in turtle
#########################################################
from turtle import *
def come(x,y):
    (xx,yy) = pos()
    newxy = ((xx+x)/2,(yy+y)/2)
    print(x,y)
    goto(newxy)
onscreenclick(come)
done()
```

