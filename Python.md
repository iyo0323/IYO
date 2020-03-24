
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
```


