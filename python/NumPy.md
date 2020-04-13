
# Top

* [Basics](#Basics)


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

np.array([1, 4, 2, 5, 3])
# array([1, 4, 2, 5, 3])

np.array([3.14, 4, 2, 3])
# array([ 3.14, 4. , 2. , 3. ])

np.array([1, 2, 3, 4], dtype='float32')
# array([ 1., 2., 3., 4.], dtype=float32)

np.array([range(i, i + 3) for i in [2, 4, 6]])
# array([[2, 3, 4], [4, 5, 6], [6, 7, 8]])
```
