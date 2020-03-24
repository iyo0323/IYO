
```py
# Sample of Function docstring
def function_name(x):
    '関数の註解はここに書く'
    #...
    return
help(function_name)
```



```py
# Use Global variable in function
a = 10
b = 0

def f():
    global b
    c = a*a
    b = c
f()
print(b, a)

#Result
#100 10
```



```py
# Pass a function as the parameter to another function
def f():
    print("f says Hello")

# 関数を引数でもらって実行する関数
def F(y):
    print("In F, ", end="")
    y()

# f を実行
f()
# f を F に渡して F を実行
F(f)

#Result
#f says Hello
#In F, f says Hello
```

