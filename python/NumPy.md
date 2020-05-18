
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

# NumPy Arrays

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

x1
# array([5, 0, 3, 3, 7, 9])
x1[0] = 3.14159 # this will be truncated!
# array([3, 0, 3, 3, 7, 9])
```



