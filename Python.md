
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
