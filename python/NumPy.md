
# Top

* [Basics](#Basics)
* [NumPy Arrays](#NumPyArrays)
* [Aggregations: Min, Max, and Everything in Between](#AggregationsMinMaxAndEverythingInBetween)
* [Computation on Arrays: Broadcasting](#ComputationOnArraysBroadcasting)


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


# ComputationOnArraysBroadcasting

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
M = np.ones((2, 3))
a = np.arange(3)
M + a
# array([[ 1., 2., 3.],
#        [ 1., 2., 3.]])

# Broadcasting example 2
a = np.arange(3).reshape((3, 1))
b = np.arange(3)
a + b
# array([[0, 1, 2],
#        [1, 2, 3],
#        [2, 3, 4]])

# Broadcasting example 3
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

[To Top](#Top)
