
Basics
=========================================================

```py
# 
#########################################################

# Get Type
type(2.0)
# <class 'float'>

# 分数
import fractions
x = fractions.Fraction(1, 3)
```


List
=========================================================


```py
# Add
#########################################################
# << ['a']

a_list = a_list + [2.0, 3]
# ['a', 2.0, 3]

a_list.append(True)
# ['a', 2.0, 3, True]

a_list.extend(['four', 'Ω'])
# ['a', 2.0, 3, True, 'four', 'Ω']

a_list.insert(0, 'Ω')
# ['Ω', 'a', 2.0, 3, True, 'four', 'Ω']
```


```py
# 
#########################################################
# Get count of element
a_list.count('new')

# Check if element in list
'new' in a_list

# Get the index number of element in list
a_list.index('mpilgrim')
```


```py
# Delete
#########################################################
# ['a', 'b', 'new', 'mpilgrim', 'new']

del a_list[1]
# ['a', 'new', 'mpilgrim', 'new']

a_list.remove('new')
# ['a', 'mpilgrim', 'new']
```


```py
# Pop
#########################################################
# ['a', 'b', 'new', 'mpilgrim']

a_list.pop()
# ['a', 'b', 'new']

a_list.pop(1)
# ['a', 'new']
```



```py
# Get an input fron CUI
#########################################################
num = int(input("Give me an integer "))
```




Function
=========================================================

```py
# Write docstring
#########################################################
def function_name(x):
    '関数の註解はここに書く'
    #...
    return
    
# Show docstring
help(function_name)
function_name.__doc__

#Result
#########################################################
#Help on function function_name in module __main__:
#function_name(x)
#   関数の註解はここに書く
#関数の註解はここに書く
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
class MyClass():
    # class variable(static)
    a = "my class" # public variable
    __b = 0         # private variable
    
    # Constructor
    def __init__(self, data):
        # instance variable
        self.__number = MyClass.__b
        self.mydata = data
    
    def show_number(self):
        print(self.__number)

# If you import the module, then __name__ is the module’s filename
if __name__ == "__main__":
    # Do something
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

