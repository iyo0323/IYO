
# Top

* [Basics](#Basics)
* [List](#List)
* [Tuple](#Tuple)
* [Sets](#Sets)
* [Comprehension](#Comprehension)
* [String](#String)
* [Regular Expressions](#RegularExpressions)
* [Closures](#Closures)
* [Generator](#Generator)
* [Iterator](#Iterator)
* [Function](#Function)
* [Class](#Class)
* [Turtle](#Turtle)


# Basics

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

```py
# OS
#########################################################
import os
print(os.path.join(os.path.expanduser('~'), 'diveintopython3', 'examples', 'humansize.py'))
# C:\Users\iyo\diveintopython3\examples\humansize.py

(dirname, filename) = os.path.split(pathname)
(shortname, extension) = os.path.splitext(filename)

print(os.getcwd())
# C:\Users\iyo\AppData\Local\Continuum\anaconda3\Scripts
print(os.path.realpath('feed.xml'))
# C:\Users\iyo\AppData\Local\Continuum\anaconda3\Scripts\feed.xml

metadata = os.stat('feed.xml')
import time
time.localtime(metadata.st_mtime)
```


```py
# Get an input fron CUI
#########################################################
num = int(input("Give me an integer "))
```


```py
# Decorator
#########################################################
def deco_tag(tag):
    def _deco_tag(func):
        def wrapper(*args, **kwargs):
            res = '<'+tag+'>'
            res = res + func(*args, **kwargs)
            res = res + '</'+tag+'>'
            return res
        return wrapper
    return _deco_tag

@deco_tag('html')
@deco_tag('body')
def test():
    return 'Hello Decorator!'

print(test())

#Result
#########################################################
<html><body>Hello Decorator!</body></html>
```

[To Top](#Top)


# List

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
# Remove
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

[To Top](#Top)


# Tuple

```py
# 
#########################################################
v = ('a', 2, True)
(x, y, z) = v

(MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY) = range(7)
```

[To Top](#Top)


# Sets

```py
# From list to set
#########################################################
a_list = ['a', 'b', 'mpilgrim', True, False, 42]
a_set = set(a_list)
# {'a', False, 'b', True, 'mpilgrim', 42}


# Set vs Dictionary
#########################################################
a_set = set()
type(a_set)
# <class 'set'>
len(a_set)
# 0

not_sure = {}
type(not_sure)
# <class 'dict'>


# Set Operations
#########################################################
a_set.update({2, 4, 6})
a_set.discard(10)
a_set.remove(21)
a_set.pop()
a_set.clear()

a_set.union(b_set)
a_set.intersection(b_set)
a_set.difference(b_set)
a_set.symmetric_difference(b_set)

a_set.issubset(b_set)
b_set.issuperset(a_set)
```

[To Top](#Top)


# Comprehension

```py
# List Comprehension
#########################################################
import os, glob
import humansize

[elem * 2 for elem in a_list]

# Get all fullpath of .XML in glob.glob()
[os.path.realpath(f) for f in glob.glob('*.xml')]

# Get all files of .PY in glob.glob() which size larger than 6000
[f for f in glob.glob('*.py') if os.stat(f).st_size > 6000]

# Get all size & fullpath of .XML in glob.glob()
[(os.stat(f).st_size, os.path.realpath(f)) for f in glob.glob('*.xml')]

# Get all size & files of .XML in glob.glob()
[(humansize.approximate_size(os.stat(f).st_size), f) for f in glob.glob('*.xml')]
```

```py
# Dictionary Comprehension
#########################################################
import os, glob, humansize

metadata_dict = {f:os.stat(f) for f in glob.glob('*test*.py')}
list(metadata_dict.keys())

humansize_dict = {
    os.path.splitext(f)[0]:humansize.approximate_size(meta.st_size) 
        for f, meta in metadata_dict.items() if meta.st_size > 6000
}

{value:key for key, value in a_dict.items()}
```

```py
# Set Comprehension
#########################################################
{x ** 2 for x in a_set}
{x for x in a_set if x % 2 == 0}
{2**x for x in range(10)}
```

[To Top](#Top)


# String

```py
# Format Specifiers
#########################################################
"{0}'s password is {1}".format(username, password)

si_suffixes
# ['KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
'1000{0[0]} = 1{0[1]}'.format(si_suffixes)
# '1000KB = 1MB'

# From modules 'humansize'
import sys
'1MB = 1000{0.modules[humansize].SUFFIXES[1000][0]}'.format(sys)
# '1MB = 1000KB'

'{0:.1f} {1}'.format(698.24, 'GB')
# '698.2 GB'

s.splitlines()
s.lower().count('f')

query = 'user=pilgrim&database=master&password=PapayaWhip'
a_list = query.split('&')
# ['user=pilgrim', 'database=master', 'password=PapayaWhip']
a_list_of_lists = [v.split('=', 1) for v in a_list]
# [['user', 'pilgrim'], ['database', 'master'], ['password', 'PapayaWhip']]
a_dict = dict(a_list_of_lists)
# {'password': 'PapayaWhip', 'user': 'pilgrim', 'database': 'master'}
```

```py
# String vs Bytes
#########################################################
by = b'abcd\x65'
# b'abcde' # len = 5
type(by)
# <class 'bytes'>
by += b'\xff'
# b'abcde\xff' # len = 6
```

```py
# Encode
#########################################################
by = a_string.encode('utf-8')
roundtrip = by.decode('big5')
```

[To Top](#Top)


# RegularExpressions

```py
# Regular Expressions (Roman Numerals)
#########################################################
import re
pattern = '^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
re.search(pattern, 'MMDCLXVI')
```

```py
# Verbose Regular Expressions (Roman Numerals)
#########################################################
pattern = '''
^                   # beginning of string
M{0,3}              # thousands - 0 to 3 Ms
(CM|CD|D?C{0,3})    # hundreds - 900 (CM), 400 (CD), 0-300 (0 to 3 Cs),
                    #           or 500-800 (D, followed by 0 to 3 Cs)
(XC|XL|L?X{0,3})    # tens - 90 (XC), 40 (XL), 0-30 (0 to 3 Xs),
                    #           or 50-80 (L, followed by 0 to 3 Xs)
(IX|IV|V?I{0,3})    # ones - 9 (IX), 4 (IV), 0-3 (0 to 3 Is),
                    #           or 5-8 (V, followed by 0 to 3 Is)
$                   # end of string
'''

import re
re.search(pattern, 'MCMLXXXIX', re.VERBOSE)
```

```py
# Regular Expressions (Phone Pattern)
#########################################################
import re
phonePattern = re.compile(r'(\d{3})\D*(\d{3})\D*(\d{4})\D*(\d*)$')
phonePattern.search('work 1-(800) 555.1212 #1234').groups()
# ('800', '555', '1212', '1234')

phonePattern = re.compile(r'''
            # don't match beginning of string, number can start anywhere
(\d{3})     # area code is 3 digits (e.g. '800')
\D*         # optional separator is any number of non-digits
(\d{3})     # trunk is 3 digits (e.g. '555')
\D*         # optional separator
(\d{4})     # rest of number is 4 digits (e.g. '1212')
\D*         # optional separator
(\d*)       # extension is optional and can be any number of digits
$           # end of string
''', re.VERBOSE)

phonePattern.search('800-555-1212')
# ('800', '555', '1212', '')
```

```py
# Findall
#########################################################
import re
re.findall('[0-9]+', '16 2-by-4s in rows of 8')
# ['16', '2', '4', '8']

re.findall('[A-Z]+', 'SEND + MORE == MONEY')
# ['SEND', 'MORE', 'MONEY']

re.findall(' s.*? s', "The sixth sick sheikh's sixth sheep's sick.")
# [' sixth s', " sheikh's s", " sheep's s"]

words = ['SEND', 'MORE', 'MONEY']
set(''.join(words))
# {'E', 'D', 'M', 'O', 'N', 'S', 'R', 'Y'}

unique_characters = {'E', 'D', 'M', 'O', 'N', 'S', 'R', 'Y'}
tuple(ord(c) for c in unique_characters)
# (69, 68, 77, 79, 78, 83, 82, 89)

```

[To Top](#Top)


# Closures
(using the values of outside parameters within a dynamic function is called closures.)

```py
# Plural Rule (by List)
#########################################################
import re
def build_match_and_apply_functions(pattern, search, replace):
    def matches_rule(word):
        return re.search(pattern, word)
    def apply_rule(word):
        return re.sub(search, replace, word)
    return (matches_rule, apply_rule)

patterns = \
    (
        ('[sxz]$', '$', 'es'),
        ('[^aeioudgkprt]h$', '$', 'es'),
        ('(qu|[^aeiou])y$', 'y$', 'ies'),
        ('$', '$', 's')
    )
rules = [build_match_and_apply_functions(pattern, search, replace)
    for (pattern, search, replace) in patterns]

def plural(noun):
    for matches_rule, apply_rule in rules:
        if matches_rule(noun):
            return apply_rule(noun)
```

```py
# Plural Rule (by File)
#########################################################
# plural4-rules.txt
# [sxz]$              $ es
# [^aeioudgkprt]h$    $ es
# [^aeiou]y$         y$ ies
# $                   $ s

rules = []
with open('plural4-rules.txt', encoding='utf-8') as pattern_file:
    for line in pattern_file:
        pattern, search, replace = line.split(None, 3)
        rules.append(build_match_and_apply_functions(pattern, search, replace))
```

[To Top](#Top)


# Generator

```py
# Fibonacci
#########################################################
def fib(max):
    a, b = 0, 1
    while a < max:
        yield a
        a, b = b, a + b

for n in fib(1000):
    print(n, end=' ')
# 0 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987

list(fib(1000))
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
```

```py
# Plural Rule
#########################################################
def rules(rules_filename):
    with open(rules_filename, encoding='utf-8') as pattern_file:
        for line in pattern_file:
            pattern, search, replace = line.split(None, 3)
            yield build_match_and_apply_functions(pattern, search, replace)

def plural(noun, rules_filename='plural5-rules.txt'):
    for matches_rule, apply_rule in rules(rules_filename):
        if matches_rule(noun):
            return apply_rule(noun)
    raise ValueError('no matching rule for {0}'.format(noun))
```

[To Top](#Top)


# Iterator

```py
# Fibonacci
#########################################################
class Fib:
    '''iterator that yields numbers in the Fibonacci sequence'''
    
    def __init__(self, max):
        self.max = max
    
    def __iter__(self):
        '''
        The __iter__() method is called whenever someone calls iter(fib). 
        (A for loop will call this automatically, but you can also call it yourself manually.)
        '''
        self.a = 0
        self.b = 1
        return self
    
    def __next__(self):
        '''The __next__() method is called whenever someone calls next() on an iterator of an instance of a class.'''
        fib = self.a
        if fib > self.max:
            raise StopIteration
        self.a, self.b = self.b, self.a + self.b
        return fib

for n in Fib(1000):
    print(n, end=' ')
# 0 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987
```

```py
# Plural Rule (Lazy Rules)
#########################################################
class LazyRules:
    rules_filename = 'plural6-rules.txt'
    
    def __init__(self):
        self.pattern_file = open(self.rules_filename, encoding='utf-8')
        self.cache = []
    
    def __iter__(self):
        self.cache_index = 0
        return self
    
    def __next__(self):
        self.cache_index += 1
        if len(self.cache) >= self.cache_index:
            # Just use the cache, because the file was loaded at first time.
            return self.cache[self.cache_index - 1]
            
        if self.pattern_file.closed:
            raise StopIteration
            
        line = self.pattern_file.readline()
        if not line:
            self.pattern_file.close()
            raise StopIteration
            
        pattern, search, replace = line.split(None, 3)
        funcs = build_match_and_apply_functions(pattern, search, replace)
        self.cache.append(funcs)
        return funcs

rules = LazyRules()
```

```py
# Puzzle
#########################################################
import re
import itertools

def solve(puzzle):
    words = re.findall('[A-Z]+', puzzle.upper())
    unique_characters = set(''.join(words))
    assert len(unique_characters) <= 10, 'Too many letters'
    
    first_letters = {word[0] for word in words}
    n = len(first_letters)
    sorted_characters = ''.join(first_letters) + ''.join(unique_characters - first_letters)
    
    characters = tuple(ord(c) for c in sorted_characters)
    digits = tuple(ord(c) for c in '0123456789')
    
    zero = digits[0]
    for guess in itertools.permutations(digits, len(characters)):
        if zero not in guess[:n]:
            equation = puzzle.translate(dict(zip(characters, guess)))
            if eval(equation):
                return equation

if __name__ == '__main__':
    import sys
    for puzzle in sys.argv[1:]:
        print(puzzle)
        solution = solve(puzzle)
        if solution:
            print(solution)

# $ python3 alphametics.py "SEND + MORE == MONEY"
# SEND + MORE == MONEY
# 9567 + 1085 == 10652
```

```py
# Iteretor tools
#########################################################
import itertools

perms = itertools.permutations([1, 2, 3], 2)
next(perms)
#(1, 2)
next(perms)
#(1, 3)

perms = itertools.permutations('ABC', 3)
next(perms)
# ('A', 'B', 'C')
next(perms)
# ('A', 'C', 'B')
list(itertools.permutations('ABC', 3))
# [('A', 'B', 'C'), ('A', 'C', 'B'), ('B', 'A', 'C'), ('B', 'C', 'A'), ('C', 'A', 'B'), ('C', 'B', 'A')]

list(itertools.product('ABC', '123'))
# [('A', '1'), ('A', '2'), ('A', '3'), ('B', '1'), ('B', '2'), ('B', '3'), ('C', '1'), ('C', '2'), ('C', '3')]

list(itertools.combinations('ABC', 2))
# [('A', 'B'), ('A', 'C'), ('B', 'C')]
```


[To Top](#Top)

# Function

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

[To Top](#Top)


# Class

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

[To Top](#Top)


# Turtle

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

[To Top](#Top)

