
# Top

* [Basics](#Basics)
* [NumPy Arrays](#NumPyArrays)
* [Aggregations: Min, Max, and Everything in Between](#AggregationsMinMaxAndEverythingInBetween)
* [Broadcasting](#Broadcasting)
* [Comparisons, Masks, and Boolean Logic](#ComparisonsMasksAndBooleanLogic)
* [Indexing](#Indexing)
* [Sorting Arrays](#SortingArrays)
* [Structured Data: NumPy’s Structured Arrays](#StructuredArrays)


# Basics

```py
# Version
#########################################################
import numpy
numpy.__version__
```

```py
# Creating Arrays from Python Lists
#########################################################
import numpy as np

np.array([3.14, 4, 2, 3])
# array([ 3.14, 4. , 2. , 3. ])

np.array([1, 2, 3, 4], dtype='float32')
# array([ 1., 2., 3., 4.], dtype=float32)

np.array([range(i, i + 3) for i in [2, 4, 6]])
# array([[2, 3, 4], [4, 5, 6], [6, 7, 8]])
```

```py
# Creating Arrays from Scratch
#########################################################
# Create a length-10 integer array filled with zeros
np.zeros(10, dtype=int)
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Create a 3x5 floating-point array filled with 1s
np.ones((3, 5), dtype=float)
# array([[ 1., 1., 1., 1., 1.],
#        [ 1., 1., 1., 1., 1.],
#        [ 1., 1., 1., 1., 1.]])

# Create a 3x5 array filled with 3.14
np.full((3, 5), 3.14)
# array([[ 3.14, 3.14, 3.14, 3.14, 3.14],
#        [ 3.14, 3.14, 3.14, 3.14, 3.14],
#        [ 3.14, 3.14, 3.14, 3.14, 3.14]])

np.arange(0, 20, 2)
# array([ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18])

np.linspace(0, 1, 5)
# array([ 0. , 0.25, 0.5 , 0.75, 1. ])

# Create a 3x3 array of uniformly distributed
# random values between 0 and 1
np.random.random((3, 3))
# array([[ 0.99844933, 0.52183819, 0.22421193],
#        [ 0.08007488, 0.45429293, 0.20941444],
#        [ 0.14360941, 0.96910973, 0.946117 ]])

# Create a 3x3 array of normally distributed random values
# with mean 0 and standard deviation 1
np.random.normal(0, 1, (3, 3))
# array([[ 1.51772646, 0.39614948, -0.10634696],
#        [ 0.25671348, 0.00732722,  0.37783601],
#        [ 0.68446945, 0.15926039, -0.70744073]])

# Create a 3x3 array of random integers in the interval [0, 10)
np.random.randint(0, 10, (3, 3))
# array([[2, 3, 4],
#        [5, 7, 8],
#        [0, 5, 0]])

# Create a 3x3 identity matrix
np.eye(3)
# array([[ 1., 0., 0.],
#        [ 0., 1., 0.],
#        [ 0., 0., 1.]])

# Create an uninitialized array of three integers
# The values will be whatever happens to already exist at that
# memory location
np.empty(3)
# array([ 1., 1., 1.])
```

[To Top](#Top)


# NumPyArrays

```py
# NumPy Array Attributes
#########################################################
import numpy
np.random.seed(0) # seed for reproducibility

x1 = np.random.randint(10, size=6) # One-dimensional array
x2 = np.random.randint(10, size=(3, 4)) # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5)) # Three-dimensional array

print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)
# x3 ndim: 3
# x3 shape: (3, 4, 5)
# x3 size: 60

print("dtype:", x3.dtype)
# dtype: int64

print("itemsize:", x3.itemsize, "bytes")
print("nbytes:", x3.nbytes, "bytes")
# itemsize: 8 bytes
# nbytes: 480 bytes
```

```py
# Array Indexing
#########################################################
x1
# array([5, 0, 3, 3, 7, 9])

x1[-1]
# 9
x1[0] = 3.14159 # this will be truncated!
# array([3, 0, 3, 3, 7, 9])
```

```py
# Array Slicing
#########################################################

# One-dimensional subarrays
###################################
x = np.arange(10)
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

x[::2] # every other element
# array([0, 2, 4, 6, 8])
x[1::2] # every other element, starting at index 1
# array([1, 3, 5, 7, 9])
x[::-1] # all elements, reversed
# array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
x[5::-2] # reversed every other from index 5
# array([5, 3, 1])

# Multidimensional subarrays
###################################
x2
# array([[12, 5, 2, 4],
#        [ 7, 6, 8, 8],
#        [ 1, 6, 7, 7]])

x2[:2, :3] # two rows, three columns
# array([[12, 5, 2],
#        [ 7, 6, 8]])
x2[:3, ::2] # all rows, every other column
# array([[12, 2],
#        [ 7, 8],
#        [ 1, 7]])
x2[::-1, ::-1]
# array([[ 7, 7, 6,  1],
#        [ 8, 8, 6,  7],
#        [ 4, 2, 5, 12]])

# Accessing array rows and columns
###################################
print(x2[:, 0]) # first column of x2
# [12 7 1]
print(x2[0, :]) # first row of x2
# [12 5 2 4]
print(x2[0]) # equivalent to x2[0, :]
# [12 5 2 4]

# Subarrays as no-copy views
###################################
x2_sub = x2[:2, :2]
print(x2_sub)
# [[12 5]
#  [ 7 6]]
x2_sub[0, 0] = 99
print(x2_sub)
# [[99 5]
#  [ 7 6]]
print(x2)
# [[99 5 2 4]
#  [ 7 6 8 8]
#  [ 1 6 7 7]]

# Creating copies of arrays
###################################
x2_sub_copy = x2[:2, :2].copy()
print(x2_sub_copy)
# [[99 5]
#  [ 7 6]]
x2_sub_copy[0, 0] = 42
print(x2_sub_copy)
# [[42 5]
#  [ 7 6]]
print(x2)
# [[99 5 2 4]
#  [ 7 6 8 8]
#  [ 1 6 7 7]]
```

```py
# Reshaping of Arrays
#########################################################
grid = np.arange(1, 10).reshape((3, 3))
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

x = np.array([1, 2, 3])
x.reshape((1, 3)) # row vector via reshape
# array([[1, 2, 3]])
x[np.newaxis, :]  # row vector via newaxis
# array([[1, 2, 3]])

x.reshape((3, 1)) # column vector via reshape
# array([[1],
#        [2],
#        [3]])
x[:, np.newaxis]  # column vector via newaxis
# array([[1],
#        [2],
#        [3]])
```

```py
# Array Concatenation and Splitting
#########################################################

# Concatenation of arrays
##############################
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])
# array([1, 2, 3, 3, 2, 1])
z = [99, 99, 99]
print(np.concatenate([x, y, z]))
# [ 1 2 3 3 2 1 99 99 99]

grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
np.concatenate([grid, grid])  # concatenate along the first axis
# array([[1, 2, 3],
#        [4, 5, 6],
#        [1, 2, 3],
#        [4, 5, 6]])
np.concatenate([grid, grid], axis=1)  # concatenate along the second axis (zero-indexed)
# array([[1, 2, 3, 1, 2, 3],
#        [4, 5, 6, 4, 5, 6]])

x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])
np.vstack([x, grid])  # vertically stack the arrays
# array([[1, 2, 3],
#        [9, 8, 7],
#        [6, 5, 4]])
y = np.array([[99], # horizontally stack the arrays
              [99]])
np.hstack([grid, y])
# array([[ 9, 8, 7, 99],
#        [ 6, 5, 4, 99]])

# Splitting of arrays
##############################
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)
# [1 2 3] [99 99] [3 2 1]

grid = np.arange(16).reshape((4, 4))
# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11],
#        [12, 13, 14, 15]])

upper, lower = np.vsplit(grid, [2])
print(upper)
print(lower)
# [[0 1 2 3]
#  [4 5 6 7]]
# [[ 8  9 10 11]
#  [12 13 14 15]]

left, right = np.hsplit(grid, [2])
print(left)
print(right)
# [[ 0  1]
#  [ 4  5]
#  [ 8  9]
#  [12 13]]
# [[ 2  3]
#  [ 6  7]
#  [10 11]
#  [14 15]]
```

```py
# NumPy’s UFuncs
#########################################################

# Array arithmetic
##############################
x = np.arange(4)
# x = [0 1 2 3]
print("x + 5 =", x + 5)
# x + 5 = [5 6 7 8]
np.add(x, 2)
# array([2, 3, 4, 5])

# Absolute value
##############################
x = np.array([-2, -1, 0, 1, 2])
abs(x)
# array([2, 1, 0, 1, 2])
np.absolute(x)
# array([2, 1, 0, 1, 2])
np.abs(x)
# array([2, 1, 0, 1, 2])

x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
np.abs(x)
# array([ 5., 5., 2., 1.])

# Trigonometric functions
##############################
theta = np.linspace(0, np.pi, 3)
print("theta = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))
# theta = [ 0. 1.57079633 3.14159265]
# sin(theta) = [ 0.00000000e+00 1.00000000e+00  1.22464680e-16]
# cos(theta) = [ 1.00000000e+00 6.12323400e-17 -1.00000000e+00]
# tan(theta) = [ 0.00000000e+00 1.63312394e+16 -1.22464680e-16]

x = [-1, 0, 1]
print("x = ", x)
print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))
# x = [-1, 0, 1]
# arcsin(x) = [-1.57079633 0.         1.57079633]
# arccos(x) = [ 3.14159265 1.57079633 0.        ]
# arctan(x) = [-0.78539816 0.         0.78539816]

# Exponents and logarithms
##############################
x = [1, 2, 3]
print("x =", x)
print("e^x =", np.exp(x))
print("2^x =", np.exp2(x))
print("3^x =", np.power(3, x))
# x = [1, 2, 3]
# e^x = [ 2.71828183 7.3890561 20.08553692]
# 2^x = [ 2. 4. 8.]
# 3^x = [ 3 9 27]

x = [1, 2, 4, 10]
print("x =", x)
print("ln(x) =", np.log(x))
print("log2(x) =", np.log2(x))
print("log10(x) =", np.log10(x))
# x = [1, 2, 4, 10]
# ln(x) = [ 0. 0.69314718 1.38629436 2.30258509]
# log2(x) = [ 0. 1. 2. 3.32192809]
# log10(x) = [ 0. 0.30103 0.60205999 1. ]

# When x is very small, these functions give more precise values than if the raw np.log or np.exp were used.
x = [0, 0.001, 0.01, 0.1]
print("exp(x) - 1 =", np.expm1(x))
print("log(1 + x) =", np.log1p(x))
# exp(x) - 1 = [ 0. 0.0010005 0.01005017 0.10517092]
# log(1 + x) = [ 0. 0.0009995 0.00995033 0.09531018]

# Specialized ufuncs
##############################
from scipy import special
# Gamma functions (generalized factorials) and related functions
x = [1, 5, 10]
print("gamma(x) =", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2) =", special.beta(x, 2))
# gamma(x) = [ 1.00000000e+00 2.40000000e+01 3.62880000e+05]
# ln|gamma(x)| = [ 0. 3.17805383 12.80182748]

# Error function (integral of Gaussian)
# its complement, and its inverse
x = np.array([0, 0.3, 0.7, 1.0])
print("erf(x) =", special.erf(x))
print("erfc(x) =", special.erfc(x))
print("erfinv(x) =", special.erfinv(x))
# erf(x) = [ 0. 0.32862676 0.67780119 0.84270079]
# erfc(x) = [ 1. 0.67137324 0.32219881 0.15729921]
# erfinv(x) = [ 0. 0.27246271 0.73286908 inf]
```

```py
# Advanced Ufunc Features
#########################################################

# Specifying output
##############################
x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)
# [ 0. 10. 20. 30. 40.]

y = np.zeros(10)
np.power(2, x, out=y[::2])
print(y)
# [ 1. 0. 2. 0. 4. 0. 8. 0. 16. 0.]

# Aggregates
##############################
x = np.arange(1, 6)
np.add.reduce(x)
# 15
np.multiply.reduce(x)
# 120
np.add.accumulate(x)
# array([ 1, 3, 6, 10, 15])
np.multiply.accumulate(x)
# array([ 1, 2, 6, 24, 120])

# Outer products
##############################
x = np.arange(1, 6)
np.multiply.outer(x, x)
# array([[ 1,  2,  3,  4, 5],
#        [ 2,  4,  6,  8, 10],
#        [ 3,  6,  9, 12, 15],
#        [ 4,  8, 12, 16, 20],
#        [ 5, 10, 15, 20, 25]])
```

[To Top](#Top)


# AggregationsMinMaxAndEverythingInBetween

```py
# Summing the Values in an Array
#########################################################
import numpy as np
L = np.random.random(100)
sum(L)
# 55.61209116604941
np.sum(L)
# 55.612091166049424

big_array = np.random.rand(1000000)
%timeit sum(big_array)
%timeit np.sum(big_array)
# 10 loops, best of 3: 104 ms per loop
# 1000 loops, best of 3: 442 μs per loop
```

```py
# Minimum and Maximum
#########################################################
min(big_array), max(big_array)
# (1.1717128136634614e-06, 0.9999976784968716)
np.min(big_array), np.max(big_array)
# (1.1717128136634614e-06, 0.9999976784968716)
%timeit min(big_array)
%timeit np.min(big_array)
# 10 loops, best of 3: 82.3 ms per loop
# 1000 loops, best of 3: 497 μs per loop

print(big_array.min(), big_array.max(), big_array.sum())
# 1.17171281366e-06 0.999997678497 499911.628197

# Multidimensional aggregates
##############################
M = np.random.random((3, 4))
print(M)
# [[ 0.8967576  0.03783739 0.75952519 0.06682827]
#  [ 0.8354065  0.99196818 0.19544769 0.43447084]
#  [ 0.66859307 0.15038721 0.37911423 0.6687194 ]]

M.sum()
# 6.0850555667307118
M.min(axis=0)
# array([ 0.66859307, 0.03783739, 0.19544769, 0.06682827])
M.max(axis=1)
# array([ 0.8967576 , 0.99196818, 0.6687194 ])
# The axis keyword specifies the dimension of the array that will be collapsed, rather than the dimension that will be returned.
```

```py
# Example: What Is the Average Height of US Presidents?
#########################################################
!head -4 data/president_heights.csv
# order,name,height(cm)
# 1,George Washington,189
# 2,John Adams,170
# 3,Thomas Jefferson,189

import pandas as pd
data = pd.read_csv('data/president_heights.csv')
heights = np.array(data['height(cm)'])
print(heights)
# [189 170 189 163 183 171 185 168 173 183 173 173 175 178 183 193 178 173
#  174 183 183 168 170 178 182 180 183 178 182 188 175 179 183 193 182 183
#  177 185 188 188 182 185]

print("Mean height: ", heights.mean())
print("Standard deviation:", heights.std())
print("Minimum height: ", heights.min())
print("Maximum height: ", heights.max())
# Mean height: 179.738095238
# Standard deviation: 6.93184344275
# Minimum height: 163
# Maximum height: 193

print("25th percentile: ", np.percentile(heights, 25))
print("Median: ", np.median(heights))
print("75th percentile: ", np.percentile(heights, 75))
# 25th percentile: 174.25
# Median: 182.0
# 75th percentile: 183.0

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set() # set plot style
plt.hist(heights)
plt.title('Height Distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number');
```

[To Top](#Top)


# Broadcasting

```py
# Introducing Broadcasting
#########################################################
a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
print(a)
print(b)
# [0 1 2]
# [[0]
#  [1]
#  [2]]
a + b
# array([[0, 1, 2],
         [1, 2, 3],
         [2, 3, 4]])
```

```py
# Rules of Broadcasting
#########################################################
# • Rule 1: If the two arrays differ in their number of dimensions, the shape of the one with fewer dimensions is padded with ones on its leading (left) side.
# • Rule 2: If the shape of the two arrays does not match in any dimension, the array with shape equal to 1 in that dimension is stretched to match the other shape.
# • Rule 3: If in any dimension the sizes disagree and neither is equal to 1, an error is raised.

# Broadcasting example 1
##############################
M = np.ones((2, 3))
a = np.arange(3)
M + a
# array([[ 1., 2., 3.],
#        [ 1., 2., 3.]])

# Broadcasting example 2
##############################
a = np.arange(3).reshape((3, 1))
b = np.arange(3)
a + b
# array([[0, 1, 2],
#        [1, 2, 3],
#        [2, 3, 4]])

# Broadcasting example 3
##############################
M = np.ones((3, 2))
a = np.arange(3)
M + a
# ValueError Traceback (most recent call last)
# <ipython-input-13-9e16e9f98da6> in <module>()
# ValueError: operands could not be broadcast together with shapes (3,2) (3,)

a[:, np.newaxis].shape
# (3, 1)
M + a[:, np.newaxis]
# array([[ 1., 1.],
#        [ 2., 2.],
#        [ 3., 3.]])

np.logaddexp(M, a[:, np.newaxis]) # logaddexp(a, b) = log(exp(a) + exp(b))
# array([[ 1.31326169, 1.31326169],
#        [ 1.69314718, 1.69314718],
#        [ 2.31326169, 2.31326169]])
```

```py
# Broadcasting in Practice
#########################################################

# Centering an array
##############################
X = np.random.random((10, 3))

# Compute the mean of each feature using the mean aggregate across the first dimension
Xmean = X.mean(0)
Xmean
# array([ 0.53514715, 0.66567217, 0.44385899])
X_centered = X - Xmean
X_centered.mean(0)
# array([ 2.22044605e-17, -7.77156117e-17, -1.66533454e-17])

# Plotting a two-dimensional function
##############################
# x and y have 50 steps from 0 to 5
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

%matplotlib inline
import matplotlib.pyplot as plt
plt.imshow(z, origin='lower', extent=[0, 5, 0, 5], cmap='viridis')
plt.colorbar();
```

[To Top](#Top)


# ComparisonsMasksAndBooleanLogic

```py
# Example: Counting Rainy Days
#########################################################
import numpy as np
import pandas as pd
# use Pandas to extract rainfall inches as a NumPy array
rainfall = pd.read_csv('data/Seattle2014.csv')['PRCP'].values
inches = rainfall / 254 # 1/10mm -> inches
inches.shape

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set() # set plot styles
plt.hist(inches, 40);
# (365,)
```

```py
# Comparison Operators as ufuncs
#########################################################
x = np.array([1, 2, 3, 4, 5])
x < 3 # less than
# array([ True, True, False, False, False], dtype=bool)
x == 3 # equal
# array([False, False, True, False, False], dtype=bool)
(2 * x) == (x ** 2)
# array([False, True, False, False, False], dtype=bool)

rng = np.random.RandomState(0)
x = rng.randint(10, size=(3, 4))
# array([[5, 0, 3, 3],
#        [7, 9, 3, 5],
#        [2, 4, 7, 6]])
x < 6
# array([[ True, True, True, True],
#        [False, False, True, True],
#        [ True, True, False, False]], dtype=bool)
```

```py
# Working with Boolean Arrays
#########################################################
print(x)
# [[5 0 3 3]
#  [7 9 3 5]
#  [2 4 7 6]]

# Counting entries
##############################
# how many values less than 6?
np.count_nonzero(x < 6) # To count the number of True entries in a Boolean array
# 8
np.sum(x < 6) # False is interpreted as 0, and True is interpreted as 1
# 8
np.sum(x < 6, axis=1) # how many values less than 6 in each row?
# array([4, 2, 2])
np.any(x > 8) # are there any values greater than 8?
# True
np.all(x == 6)  # are all values equal to 6?
# False
np.all(x < 8, axis=1) # are all values in each row less than 8?
# array([ True, False, True], dtype=bool)

# Boolean operators
##############################
np.sum((inches > 0.5) & (inches < 1))
# 29
np.sum(~( (inches <= 0.5) | (inches >= 1) ))
# 29

print("Number days without rain: ", np.sum(inches == 0))
print("Number days with rain: ", np.sum(inches != 0))
print("Days with more than 0.5 inches:", np.sum(inches > 0.5))
print("Rainy days with < 0.1 inches :", np.sum((inches > 0) & (inches < 0.2)))
# Number days without rain: 215
# Number days with rain: 150
# Days with more than 0.5 inches: 37
# Rainy days with < 0.1 inches : 75
```

```py
# Boolean Arrays as Masks
#########################################################
x
# array([[5, 0, 3, 3],
#        [7, 9, 3, 5],
#        [2, 4, 7, 6]])

x < 5
# array([[False,  True,  True,  True],
#        [False, False,  True, False],
#        [ True,  True, False, False]], dtype=bool)
x[x < 5]
# array([0, 3, 3, 3, 2, 4])

# construct a mask of all rainy days
rainy = (inches > 0)
# construct a mask of all summer days (June 21st is the 172nd day)
summer = (np.arange(365) - 172 < 90) & (np.arange(365) - 172 > 0)

print("Median precip on rainy days in 2014 (inches): ", np.median(inches[rainy]))
print("Median precip on summer days in 2014 (inches): ", np.median(inches[summer]))
print("Maximum precip on summer days in 2014 (inches): ", np.max(inches[summer]))
print("Median precip on non-summer rainy days (inches):", np.median(inches[rainy & ~summer]))
# Median precip on rainy days in 2014 (inches): 0.194881889764
# Median precip on summer days in 2014 (inches): 0.0
# Maximum precip on summer days in 2014 (inches): 0.850393700787
# Median precip on non-summer rainy days (inches): 0.200787401575
```

```py
# Using the Keywords and/or Versus the Operators &/|
#########################################################

# and and or gauge the truth or falsehood of entire object, while & and | refer to bits within each object.

bool(42), bool(0)
# (True, False)
bool(42 and 0)
# False
bool(42 or 0)
# True

bin(42)
# '0b101010'
bin(59)
# '0b111011'
bin(42 & 59)
# '0b101010'
bin(42 | 59)
# '0b111011'

A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
A | B
# array([ True, True, True, False, True, True], dtype=bool)
A or B
# ValueError Traceback (most recent call last)
# <ipython-input-38-5d8e4f2e21c0> in <module>()
# ValueError: The truth value of an array with more than one element is...

x = np.arange(10)
(x > 4) & (x < 8)
# array([False, False, ..., True, True, False, False], dtype=bool)
(x > 4) and (x < 8)
# ValueError Traceback (most recent call last)
# <ipython-input-40-3d24f1ffd63d> in <module>()
# ValueError: The truth value of an array with more than one element is...
```

[To Top](#Top)


# Indexing

```py
# Exploring Fancy Indexing
#########################################################
import numpy as np
rand = np.random.RandomState(42)
x = rand.randint(100, size=10)
print(x)
# [51 92 14 71 60 20 82 86 74 74]

[x[3], x[7], x[2]]
# [71, 86, 14]
ind = [3, 7, 4]
x[ind]
# array([71, 86, 60])

ind = np.array([[3, 7],
                [4, 5]])
x[ind]
# array([[71, 86],
#        [60, 20]])

X = np.arange(12).reshape((3, 4))
# array([[ 0, 1, 2, 3],
#        [ 4, 5, 6, 7],
#        [ 8, 9, 10, 11]])

row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
X[row, col]
# array([ 2, 5, 11])

X[row[:, np.newaxis], col]
# array([[ 2, 1, 3],
#        [ 6, 5, 7],
#        [10, 9, 11]])

row[:, np.newaxis] * col
# array([[0, 0, 0],
#        [2, 1, 3],
#        [4, 2, 6]])
```

```py
# Combined Indexing
#########################################################
print(X)
# [[ 0 1  2  3]
#  [ 4 5  6  7]
#  [ 8 9 10 11]]

X[2, [2, 0, 1]]
# array([10, 8, 9])

X[1:, [2, 0, 1]]
# array([[ 6, 4, 5],
#        [10, 8, 9]])

mask = np.array([1, 0, 1, 0], dtype=bool)
X[row[:, np.newaxis], mask]
# array([[ 0,  2],
#        [ 4,  6],
#        [ 8, 10]])
```

```py
# Example: Selecting Random Points
#########################################################
mean = [0, 0]
cov = [[1, 2],
       [2, 5]]
X = rand.multivariate_normal(mean, cov, 100)
X.shape
# (100, 2)

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set() # for plot styling
plt.scatter(X[:, 0], X[:, 1]);

indices = np.random.choice(X.shape[0], 20, replace=False)
# array([93, 45, 73, 81, 50, 10, 98, 94, 4, 64, 65, 89, 47, 84, 82, 80, 25, 90, 63, 20])
selection = X[indices] # fancy indexing here
selection.shape
# (20, 2)

plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
plt.scatter(selection[:, 0], selection[:, 1],
facecolor='none', s=200);
```

```py
# Modifying Values with Fancy Indexing
#########################################################
x = np.arange(10)
i = np.array([2, 1, 8, 4])
x[i] = 99
print(x)
# [ 0 99 99 3 99 5 6 7 99 9]
x[i] -= 10
print(x)
# [ 0 89 89 3 89 5 6 7 89 9]

x = np.zeros(10)
x[[0, 0]] = [4, 6]
print(x)
# [ 6. 0. 0. 0. 0. 0. 0. 0. 0. 0.] 
# The result of this operation is to first assign x[0] = 4, followedby x[0] = 6.
i = [2, 3, 3, 4, 4, 4]
x[i] += 1
x
# array([ 6., 0., 1., 1., 1., 0., 0., 0., 0., 0.])

x = np.zeros(10)
np.add.at(x, i, 1)
print(x)
# [ 0. 0. 1. 2. 3. 0. 0. 0. 0. 0.]
# The at() method does an in-place application of the given operator at the specified indices (here, i) with the specified value (here, 1).
```

```py
# Example: Binning Data
#########################################################
np.random.seed(42)
x = np.random.randn(100)

# compute a histogram by hand
bins = np.linspace(-5, 5, 20)
counts = np.zeros_like(bins)

# find the appropriate bin for each x
i = np.searchsorted(bins, x)

# add 1 to each of these bins
np.add.at(counts, i, 1)

# plot the results
plt.plot(bins, counts, linestyle='steps');
plt.hist(x, bins, histtype='step');

print("NumPy routine:")
%timeit counts, edges = np.histogram(x, bins)
print("Custom routine:")
%timeit np.add.at(counts, np.searchsorted(bins, x), 1)
# NumPy routine:
# 10000 loops, best of 3: 97.6 μs per loop
# Custom routine:
# 10000 loops, best of 3: 19.5 μs per loop

x = np.random.randn(1000000)
print("NumPy routine:")
%timeit counts, edges = np.histogram(x, bins)
print("Custom routine:")
%timeit np.add.at(counts, np.searchsorted(bins, x), 1)
# NumPy routine:
# 10 loops, best of 3: 68.7 ms per loop
# Custom routine:
# 10 loops, best of 3: 135 ms per loop
```

[To Top](#Top)


# SortingArrays

```py
# np.sort and np.argsort
#########################################################
# By default np.sort uses quicksort algorithm
x = np.array([2, 1, 4, 3, 5])
np.sort(x)
# array([1, 2, 3, 4, 5])
x.sort()
print(x)
# [1 2 3 4 5]

# A related function is argsort, which instead returns the indices of the sorted elements
x = np.array([2, 1, 4, 3, 5])
i = np.argsort(x)
print(i)
# [1 0 3 2 4]
x[i]
# array([1, 2, 3, 4, 5])

# Sorting along rows or columns
##############################
rand = np.random.RandomState(42)
X = rand.randint(0, 10, (4, 6))
print(X)
# [[6 3 7 4 6 9]
#  [2 6 7 4 3 7]
#  [7 2 5 4 1 7]
#  [5 1 4 0 9 5]]

# sort each column of X
np.sort(X, axis=0)
# array([[2, 1, 4, 0, 1, 5],
#        [5, 2, 5, 4, 3, 7],
#        [6, 3, 7, 4, 6, 7],
#        [7, 6, 7, 4, 9, 9]])

# sort each row of X
np.sort(X, axis=1)
# array([[3, 4, 6, 6, 7, 9],
#        [2, 3, 4, 6, 7, 7],
#        [1, 2, 4, 5, 7, 7],
#        [0, 1, 4, 5, 5, 9]])
```

```py
# Partial Sorts: Partitioning
#########################################################
# np.partition takes an array and a number K; the result is a new array with the smallest K values to the left of the partition, and the remaining values to the right, in arbitrary order
x = np.array([7, 2, 3, 1, 6, 5, 4])
np.partition(x, 3)
# array([2, 1, 3, 4, 6, 5, 7])

np.partition(X, 2, axis=1)
# array([[3, 4, 6, 7, 6, 9],
#        [2, 3, 4, 7, 6, 7],
#        [1, 2, 4, 5, 7, 7],
#        [0, 1, 4, 5, 9, 5]])
```

```py
# Example: k-Nearest Neighbors
#########################################################
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set() # Plot styling

X = rand.rand(10, 2)
plt.scatter(X[:, 0], X[:, 1], s=100);

dist_sq = np.sum((X[:,np.newaxis,:] - X[np.newaxis,:,:]) ** 2, axis=-1)

# for each pair of points, compute differences in their coordinates
differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
differences.shape
# (10, 10, 2)

# square the coordinate differences
sq_differences = differences ** 2
sq_differences.shape
# (10, 10, 2)

dist_sq.diagonal()
# array([ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

nearest = np.argsort(dist_sq, axis=1)
print(nearest)
# [[0 3 9 7 1 4 2 5 6 8]
#  [1 4 7 9 3 6 8 5 0 2]
#  [2 1 4 6 3 0 8 9 7 5]
#  [3 9 7 0 1 4 5 8 6 2]
#  [4 1 8 5 6 7 9 3 0 2]
#  [5 8 6 4 1 7 9 3 2 0]
#  [6 8 5 4 1 7 9 3 2 0]
#  [7 9 3 1 4 0 5 8 6 2]
#  [8 5 6 4 1 7 9 3 2 0]
#  [9 7 3 0 1 4 5 8 6 2]]
# the first column gives the numbers 0 through 9 in order: this is due to the fact that each point’s closest neighbor is itself

K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)
plt.scatter(X[:, 0], X[:, 1], s=100)

# draw lines from each point to its two nearest neighbors
K = 2
for i in range(X.shape[0]):
  for j in nearest_partition[i, :K+1]:
    # plot a line from X[i] to X[j]
    # use some zip magic to make it happen:
    plt.plot(*zip(X[j], X[i]), color='black')
```

[To Top](#Top)


# StructuredArrays

```py
# Basic
#########################################################
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]
x = np.zeros(4, dtype=int)

# Use a compound data type for structured arrays
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'), 'formats':('U10', 'i4', 'f8')})
print(data.dtype)
# [('name', '<U10'), ('age', '<i4'), ('weight', '<f8')]
# 'U10' translates to “Unicode string of maximum length 10,” 'i4' translates to “4-byte (i.e., 32 bit) integer,” and 'f8' translates to “8-byte (i.e., 64 bit) float.”

data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)
# [('Alice', 25, 55.0) ('Bob', 45, 85.5) ('Cathy', 37, 68.0) ('Doug', 19, 61.5)]

# Get all names
data['name']
# array(['Alice', 'Bob', 'Cathy', 'Doug'], dtype='<U10')

# Get first row of data
data[0]
# ('Alice', 25, 55.0)

# Get the name from the last row
data[-1]['name']
# 'Doug'

# Get names where age is under 30
data[data['age'] < 30]['name']
# array(['Alice', 'Doug'], dtype='<U10')
```

```py
# Creating Structured Arrays
#########################################################
np.dtype({'names':('name', 'age', 'weight'), 
  'formats':('U10', 'i4', 'f8')})
# dtype([('name', '<U10'), ('age', '<i4'), ('weight', '<f8')])

np.dtype({'names':('name', 'age', 'weight'), 
  'formats':((np.str_, 10), int, np.float32)})
# dtype([('name', '<U10'), ('age', '<i8'), ('weight', '<f4')])

np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])
# dtype([('name', 'S10'), ('age', '<i4'), ('weight', '<f8')])

np.dtype('S10, i4, f8')
# dtype([('f0', 'S10'), ('f1', '<i4'), ('f2', '<f8')])
```

```py
# More Advanced Compound Types
#########################################################
tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
X = np.zeros(1, dtype=tp)
print(X[0])
print(X['mat'][0])
# (0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
# [[ 0. 0. 0.]
#  [ 0. 0. 0.]
#  [ 0. 0. 0.]]
```

```py
# RecordArrays: Structured Arrays with a Twist
#########################################################
# The np.recarray class, which is almost identical to the structured arrays just described, but with one additional feature: fields can be accessed as attributes rather than as dictionary keys.
data['age']
# array([25, 45, 37, 19], dtype=int32)

data_rec = data.view(np.recarray)
data_rec.age
# array([25, 45, 37, 19], dtype=int32)

%timeit data['age']
%timeit data_rec['age']
%timeit data_rec.age
# 1000000 loops, best of 3: 241 ns per loop
# 100000 loops, best of 3: 4.61 μs per loop
# 100000 loops, best of 3: 7.27 μs per loop
```

[To Top](#Top)
