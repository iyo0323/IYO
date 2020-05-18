
# Top

* [Basics](#Basics)
* [NumPy Arrays](#NumPyArrays)


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
#        [ 0.25671348, 0.00732722, 0.37783601],
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
# array([[ 7, 7, 6, 1],
#        [ 8, 8, 6, 7],
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
# array([[ 0, 1, 2, 3],
#        [ 4, 5, 6, 7],
#        [ 8, 9, 10, 11],
#        [12, 13, 14, 15]])

upper, lower = np.vsplit(grid, [2])
print(upper)
print(lower)
# [[0 1 2 3]
#  [4 5 6 7]]
# [[ 8 9 10 11]
#  [12 13 14 15]]

left, right = np.hsplit(grid, [2])
print(left)
print(right)
# [[ 0 1]
#  [ 4 5]
#  [ 8 9]
#  [12 13]]
# [[ 2 3]
#  [ 6 7]
#  [10 11]
#  [14 15]]
```


