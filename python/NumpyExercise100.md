
# Top

[Numpy Officail Site](https://numpy.org/doc/stable/index.html)

[GitHub - 100 numpy exercises](https://github.com/rougier/numpy-100)

[解答](https://osawat.hatenablog.com/entry/2018/12/24/044047)

[解説](https://minus9d.hatenablog.com/entry/2017/05/21/221242)


* [01~10](#01)
* [11~20](#11)
* [21~30](#21)
* [31~40](#31)
* [41~50](#41)
* [51~60](#51)
* [61~70](#61)
* [71~80](#71)
* [81~90](#81)
* [91~100](#91)


# 01

01. Import the numpy package under the name np (★☆☆)
```py
# 01. numpy パッケージを `np` の名前でインポートする (★☆☆)
#########################################################
import numpy as np
```

02. Print the numpy version and the configuration (★☆☆)
```py
# 02. numpy バージョンとその設定を表示する (★☆☆)
#########################################################
print(np.__version__)
np.show_config()
```

03. Create a null vector of size 10 (★☆☆)
```py
# 03. 大きさ 10 の零ベクトルを生成する (★☆☆)
#########################################################
Z = np.zeros(10)
print(Z)
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
```

04. How to find the memory size of any array (★☆☆)
```py
# 04. 配列のメモリーサイズを知る方法は (★☆☆)
#########################################################
Z = np.zeros((10,10))
print(Z.size*Z.itemsize)
# 800
```

05. How to get the documentation of the numpy add function from the command line? (★☆☆)
```py
# 05. コマンドラインからの、numpy add 関数のドキュメントの取得方法は? (★☆☆)
#########################################################
python -c "import numpy; numpy.info(numpy.add)"
```

06. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)
```py
# 06. 5番目の値だけ 1 の大きさ 10 の零ベクトルを生成する (★☆☆)
#########################################################
Z = np.zeros(10)
Z[4] = 1
print(Z)
# [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
```

07. Create a vector with values ranging from 10 to 49 (★☆☆)
```py
# 07. 値の範囲が 10 から 49 であるようなベクトルを生成する (★☆☆)
#########################################################
Z = np.arange(10,50)
print(Z)
# [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49]
```

08. Reverse a vector (first element becomes last) (★☆☆)
```py
# 08. ベクトルを逆転する (最初の要素が最後に) (★☆☆)
#########################################################
Z = np.arange(50)
Z = Z[::-1]
print(Z)
# [49 48 47 46 45 44 43 42 41 40 39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0]
```

09. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)
```py
# 09. 値の範囲が 0 から 8 であるような 3x3 マトリクスを生成する (★☆☆)
#########################################################
Z = np.arange(9).reshape(3,3)
print(Z)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]
```

10. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)
```py
# 10. [1,2,0,0,4,0] からゼロでない要素の添え字を見つける (★☆☆)
#########################################################
nz = np.nonzero([1,2,0,0,4,0])
print(nz)
# (array([0, 1, 4], dtype=int64),)
```

* [To Top](#Top)


# 11

11. Create a 3x3 identity matrix (★☆☆)
```py
# 11. 3x3 の単位行列を生成する (★☆☆)
#########################################################
np.eye(3)
# array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]])
```

12. Create a 3x3x3 array with random values (★☆☆)
```py
# 12. 乱数で 3x3x3 配列を生成する (★☆☆)
#########################################################
Z = np.random.random((3,3,3))
print(Z)
# [[[0.52139159 0.71417179 0.98566575]
#   [0.71946288 0.55356541 0.46827844]
#   [0.14190383 0.78303101 0.42199173]]
#
#  [[0.79905955 0.23618193 0.6525496 ]
#   [0.78334409 0.34081869 0.50622032]
#   [0.23197788 0.27511517 0.58518794]]
#
#  [[0.71064867 0.88496634 0.44757271]
#   [0.36761105 0.29457499 0.26001356]
#   [0.40253981 0.13853578 0.56675828]]]
```

13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)
```py
# 13. 乱数で 10x10 配列を生成して、その最小値と最大値を見つける (★☆☆)
#########################################################
Z = np.random.random((10,10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)
# 0.0013772822154449749 0.9929879247278675
```

14. Create a random vector of size 30 and find the mean value (★☆☆)
```py
# 14. 大きさ 30 の乱数のベクトルを生成して、その平均を求める (★☆☆)
#########################################################
Z = np.random.random(30)
m = Z.mean()
print(m)
# 0.5043710617045838
```

15. Create a 2d array with 1 on the border and 0 inside (★☆☆)
```py
# 15. 周囲が 1 で内部が 0 であるような2次元配列生成する (★☆☆)
#########################################################
Z = np.ones((10,10))
Z[1:-1,1:-1] = 0
print(Z)
# [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
```

16. How to add a border (filled with 0's) around an existing array? (★☆☆)
```py
# 16. 既存の配列の周囲を 0 で囲む方法は? (★☆☆)
#########################################################
Z = np.ones((5,5))
Z = np.pad(Z, pad_width=1, mode="constant", constant_values=0)
print(Z)
# [[0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 1. 1. 1. 1. 0.]
#  [0. 1. 1. 1. 1. 1. 0.]
#  [0. 1. 1. 1. 1. 1. 0.]
#  [0. 1. 1. 1. 1. 1. 0.]
#  [0. 1. 1. 1. 1. 1. 0.]
#  [0. 0. 0. 0. 0. 0. 0.]]
```

17. What is the result of the following expression? (★☆☆)
```py
# 17. 以下の式の結果は何か? (★☆☆)
#########################################################
# 0 * np.nan
# np.nan == np.nan
# np.inf > np.nan
# np.nan - np.nan
# 0.3 == 3 * 0.1
#########################################################
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(0.3 == 3 * 0.1)
# nan
# False
# False
# nan
# False
```

18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)
```py
# 18. 対角成分の直下に 1,2,3,4 の成分を持つ 5x5 行列を生成する (★☆☆)
#########################################################
Z = np.diag(1+np.arange(4), k=-1)
print(Z)
# [[0 0 0 0 0]
#  [1 0 0 0 0]
#  [0 2 0 0 0]
#  [0 0 3 0 0]
#  [0 0 0 4 0]]
```

19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)
```py
# 19. 8x8 行列を生成して、市松模様で埋める (★☆☆)
#########################################################
Z = np.zeros((8,8), dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)
# [[0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]]
```

20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?
```py
# 20. shape属性が (6,7,8) の配列のとき、100 番目の要素の添え字 (x,y,z)は？
#########################################################
np.unravel_index(100, (6,7,8))
# (1, 5, 4)
#########################################################
# ■ 解説
# 100 = (1 * 7 * 8) + (5 * 8) + 4
```

* [To Top](#Top)


# 21

21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)
```py
# 21. tile 関数を使って 8x8 の市松模様の行列を生成する (★☆☆)
#########################################################
Z = np.tile(np.array([[0,1],[1,0]]),(4,4))
print(Z)
# [[0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]]
```

22. Normalize a 5x5 random matrix (★☆☆)
```py
# 22. 5x5 の乱数の行列を正規化する (★☆☆)
#########################################################
Z = np.random.random((5,5))
Zmin, Zmax = Z.min(), Z.max()
Z = (Z - Zmin)/(Zmax - Zmin)
print(Z)
# [[0.76356044 0.03799654 0.21609297 0.37122944 0.17555795]
#  [0.00609923 0.14550604 0.8295974  0.34273491 0.50677197]
#  [0.35251777 0.04047856 0.22341527 1.         0.50534599]
#  [0.37293817 0.57006785 0.80328607 0.33740204 0.07917562]
#  [0.4646811  0.07425622 0.91839516 0.21776284 0.        ]]
```

23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)
```py
# 23. 色を 4 個の符号なしバイト型 (RGBA) で表現するカスタム dtype を作成する (★☆☆)
#########################################################
color = np.dtype( [('r', np.ubyte, 1),
                   ('g', np.ubyte, 1),
                   ('b', np.ubyte, 1),
                   ('a', np.ubyte, 1)])
```

24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)
```py
# 24. 5x3 行列と 3x2 行列の掛け算 (実数の行列積) (★☆☆)
#########################################################
Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)
# [[3. 3.]
#  [3. 3.]
#  [3. 3.]
#  [3. 3.]
#  [3. 3.]]
# 別解, Python3.5 以降
Z = np.ones((5,3)) @ np.ones((3,2))
```

25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)
```py
# 25. 1次元配列が与えられたとき、３番目から８番目の全要素を-1にする (★☆☆)
#########################################################
Z = np.arange(11)
Z[(Z>=3) & (Z<=8)] = -1
print(Z)
# [ 0  1  2 -1 -1 -1 -1 -1 -1  9 10]
```

26. What is the output of the following script? (★☆☆)
```py
# 26. 以下のスクリプトの出力は? (★☆☆)
#########################################################
# print(sum(range(5),-1))
# from numpy import *
# print(sum(range(5),-1))
#########################################################
print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
# 9
# 10
#########################################################
# ■ 解説
# python組み込みのsum()関数は、sum(iterable[, start])という引数を持ちます。これは以下のように計算されます。
# ------------------------------------
# def sum(values, start = 0):
#     total = start
#     for value in values:
#         total = total + value
#     return total
# ------------------------------------
# なので、print(sum(range(5),-1))を計算すると -1 + 0 + 1 + 2 + 3 + 4 = 9となります。

# 一方、from numpy as npすると、python組み込みのsum()関数の代わりに、np.sum()が呼ばれます。
# print(np.sum(range(5), axis=-1))
# 0 + 1 + 2 + 3 + 4 = 10となります。
```

27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)
```py
# 27. Z が整数型のベクトルのとき、これらの式のどれが適切か？ (★☆☆)
#########################################################
# Z**Z
# 2 << Z >> 2
# Z <- Z
# 1j*Z
# Z/1/1
# ZZ
#########################################################
Z = np.arange(3)
# array([0, 1, 2])
Z**Z        # array([1, 1, 4], dtype=int32)
2 << Z >> 2 # array([0, 1, 2], dtype=int32)
Z <- Z      # array([False, False, False])
1j*Z        # array([0.+0.j, 0.+1.j, 0.+2.j])
Z/1/1       # array([0., 1., 2.])
ZZ # NameError: name 'ZZ' is not defined
```

28. What are the result of the following expressions?
```py
# 28. 以下の式の結果は何か?
#########################################################
# print(np.array(0) / np.array(0))
# print(np.array(0) // np.array(0))
# print(np.array([np.nan]).astype(int).astype(float))
#########################################################
print(np.array(0) / np.array(0))
print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))
# nan
# 0
# [-9.22337204e+18]
#########################################################
# ■ 解説
# np.array([0])は長さ1のベクトルです。
# np.array(0)は「0という値を持つ0次元のarray」を表すようです。
# 0次元のarrayというのは数学的にはスカラーだと思うのですが、NumPyにおける0次元のndarrayは、NumPyにおけるscalarとは別物のようです。
```

29. How to round away from zero a float array ? (★☆☆)
```py
# 29. ゼロから遠くなるように浮動小数点型の配列の小数点を丸める方法は? (★☆☆)
#########################################################
# Author: Charles R Harris
Z = np.random.uniform(-10,+10,10)
print(Z)
print (np.copysign(np.ceil(np.abs(Z)), Z))
# [-4.75375598  8.72993389 -2.08680534 -4.60164073 -1.92325826 -2.04734398 -4.27149975  2.62664347 -0.47571341 -1.84556154]
# [-5.  9. -3. -5. -2. -3. -5.  3. -1. -2.]
#########################################################
# ■ 解説
# np.copysignは第一引数の符号が第二引数の符号に置き換えられる。
```

30. How to find common values between two arrays? (★☆☆)
```py
# 30. 2つの配列に共通する値の見つけ方は? (★☆☆)
#########################################################
Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(Z1)
print(Z2)
print(np.intersect1d(Z1, Z2))
# [0 8 4 9 3 8 4 5 1 8]
# [8 8 8 2 1 7 4 8 2 3]
# [1 3 4 8]
```

* [To Top](#Top)


# 31

31. How to ignore all numpy warnings (not recommended)? (★☆☆)
```py
# 31. すべての numpy の警告を無視する方法は (非推奨)? (★☆☆)
#########################################################
# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# Back to sanity
_ = np.seterr(**defaults)

# An equivalent way, with a context manager:
with np.errstate(divide='ignore'):
    Z = np.ones(1) / 0
```

32. Is the following expressions true? (★☆☆)
```py
# 32. 以下の式は正しいですか? (★☆☆)
#########################################################
np.sqrt(-1) == np.emath.sqrt(-1)
#########################################################
np.sqrt(-1) == np.emath.sqrt(-1)
# False
#########################################################
# ■ 解説
# np.sqrt(-1)はnanになり、np.emath.sqrt(-1)は1jになります。
# np.sqrt() は引数が複素数の場合は、負の実数が含まれている場合も計算結果を複素数で返すが、引数が負の実数の場合には nan を返す。一方でnp.emath.sqrt() は引数が負の実数の場合にも、複素数を返す。
```

33. How to get the dates of yesterday, today and tomorrow? (★☆☆)
```py
# 33. 昨日、今日、明日の日付の取得方法は? (★☆☆)
#########################################################
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today = np.datetime64('today', 'D')
tomorrow = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print(yesterday)
print(today)
print(tomorrow)
# 2020-05-20
# 2020-05-21
# 2020-05-22
```

34. How to get all the dates corresponding to the month of July 2016? (★★☆)
```py
# 34. 2016年7月のすべての日付の取得方法は? (★★☆)
#########################################################
Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(Z)
# ['2016-07-01' '2016-07-02' '2016-07-03' '2016-07-04' '2016-07-05'
#  '2016-07-06' '2016-07-07' '2016-07-08' '2016-07-09' '2016-07-10'
#  '2016-07-11' '2016-07-12' '2016-07-13' '2016-07-14' '2016-07-15'
#  '2016-07-16' '2016-07-17' '2016-07-18' '2016-07-19' '2016-07-20'
#  '2016-07-21' '2016-07-22' '2016-07-23' '2016-07-24' '2016-07-25'
#  '2016-07-26' '2016-07-27' '2016-07-28' '2016-07-29' '2016-07-30'
#  '2016-07-31']
```

35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)
```py
# 35. ((A+B)\*(-A/2))の計算方法は (copyせずに)? (★★☆)
#########################################################
A = np.ones(3)*1
B = np.ones(3)*2
C = np.ones(3)*3
print( np.add(A,B,out=B)) 
print( np.divide(A,2,out= A))
print( np.negative(A,out= A))
print( np.multiply(A,B,out=A))
# [3. 3. 3.]
# [0.5 0.5 0.5]
# [-0.5 -0.5 -0.5]
# [-1.5 -1.5 -1.5]
```

36. Extract the integer part of a random array using 5 different methods (★★☆)
```py
# 36. 乱数の配列から整数部分を抽出する5種類の方法は (★★☆)
#########################################################
Z = np.random.uniform(0,10,10)
print (Z - Z%1)
print (np.floor(Z))
print (np.ceil(Z)-1)
print (Z.astype(int))
print (np.trunc(Z))
# [4. 6. 5. 2. 9. 2. 6. 6. 1. 9.]
# [4. 6. 5. 2. 9. 2. 6. 6. 1. 9.]
# [4. 6. 5. 2. 9. 2. 6. 6. 1. 9.]
# [4 6 5 2 9 2 6 6 1 9]
# [4. 6. 5. 2. 9. 2. 6. 6. 1. 9.]
```

37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)
```py
# 37. 行の値が 0 から 4 の 5x5 行列を生成する (★★☆)
#########################################################
Z = np.zeros((5, 5))
Z += np.arange(5)
print(Z)
# [[0. 1. 2. 3. 4.]
#  [0. 1. 2. 3. 4.]
#  [0. 1. 2. 3. 4.]
#  [0. 1. 2. 3. 4.]
#  [0. 1. 2. 3. 4.]]
```

38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)
```py
# 38. 10個の整数を生成するジェネレータ関数があるとき、それを使って配列を生成する (★☆☆)
#########################################################
def generate():
   for i in range(10):
       yield i
Z = np.fromiter(generate(), dtype=float, count=-1)
print(Z)
# [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
#########################################################
# ■ 解説
# numpy.fromiter(iterable, dtype, count=-1)
# Create a new 1-dimensional array from an iterable object.
# np.fromiter()は、イテレータからndarrayを生成する関数です。引数のcountは、要素数の上限です。
```

39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)
```py
# 39. 0 から 1 の範囲の値を持つ大きさ 10 のベクトルを生成する (ただし、両端の 0 と 1 は含めない) (★★☆)
#########################################################
Z = np.linspace(0, 1, 11, endpoint=False)[1:]
print(Z)
# [0.09090909 0.18181818 0.27272727 0.36363636 0.45454545 0.54545455 0.63636364 0.72727273 0.81818182 0.90909091]
#########################################################
# ■ 解説
# numpy.linspace()も等差数列を生成するが、間隔（公差）ではなく要素数を指定する。
# 第一引数startに最初の値、第二引数stopに最後の値、第三引数numに要素数を指定する。それらに応じた間隔（公差）が自動的に算出される。
```

40. Create a random vector of size 10 and sort it (★★☆)
```py
# 40. 大きさが 10 の乱数を生成して、それを並びかえる (★★☆)
#########################################################
Z = np.random.random(10)
Z.sort()
print(Z)
# [0.05927249 0.08552114 0.12987335 0.20097557 0.22175876 0.37299276 0.39795321 0.40155124 0.64217058 0.98197582]
```

* [To Top](#Top)


# 41

41. How to sum a small array faster than np.sum? (★★☆)
```py
# 41. np.sumより高速に小さい配列を集計する方法は? (★★☆)
#########################################################
Z = np.arange(10)
%timeit np.add.reduce(Z)
%timeit Z.sum()
# 3.69 µs ± 162 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 6.33 µs ± 362 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
#########################################################
# ■ 解説
# np.sum()は内部でnp.add.reduce()を呼んでいるので、オーバーヘッド分だけnp.sum()のほうが遅くなるようです。
```

42. Consider two random array A and B, check if they are equal (★★☆)
```py
# 42. A と B の2つの乱数配列があるとき、それらが等しいかをチェックする (★★☆)
#########################################################
A = np.random.randint(0, 2, 5)
B = np.random.randint(0, 2, 5)
# Assuming identical shape of the arrays and a tolerance for the comparison of values
equal = np.allclose(A, B)
print(equal)
# Checking both the shape and the element values, no tolerance (values have to be exactly equal)
equal = np.array_equal(A, B)
print(equal)
# False
# False
```

43. Make an array immutable (read-only) (★★☆)
```py
# 43. イミュータブル(読み取り専用)の配列を生成する (★★☆)
#########################################################
Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1
#       1 Z = np.zeros(10)
#       2 Z.flags.writeable = False
# ----> 3 Z[0] = 1
# ValueError: assignment destination is read-only
```

44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)
```py
# 44. 直交座標で 10x2 行列があるとき、それらを極座標に変換する (★★☆)
#########################################################
Z = np.random.random((10,2))
X, Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)
# [0.87681217 0.3161743  0.90135585 0.6531784  0.64847374 0.68742857 0.84485734 1.12235679 0.62017815 1.0990964 ]
# [0.38481736 1.46375703 0.99300613 0.22501077 1.3079256  0.5497876 0.44970742 0.92537148 1.02306295 1.064497 ]
#########################################################
# ■ 解説
# R = np.sqrt(X**2+Y**2)の部分はR = np.hypot(X, Y)としたほうが簡潔です。
# np.arctan2(a, b)	a/bのarctanを返す。戻り値の範囲は[-pi, pi]
```

45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)
```py
# 45. 大きさ 10 の乱数のベクトルを生成して、その最大値を 0 に置き換える (★★☆)
#########################################################
Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)
# [0.29018266 0.2156002  0.68783096 0.51589855 0.10440006 0.10542743 0.5956617  0.06487708 0.         0.3238788 ]
```

46. Create a structured array with x and y coordinates covering the [0,1]x[0,1] area (★★☆)
```py
# 46. [0,1]x[0,1] 領域を網羅する `x` 座標と `y` 座標を持つ構造化配列を生成する (★★☆)
#########################################################
Z = np.zeros((5,5), [('x', float), ('y', float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5), np.linspace(0,1,5))
print(Z)
# [[(0.  , 0.  ) (0.25, 0.  ) (0.5 , 0.  ) (0.75, 0.  ) (1.  , 0.  )]
#  [(0.  , 0.25) (0.25, 0.25) (0.5 , 0.25) (0.75, 0.25) (1.  , 0.25)]
#  [(0.  , 0.5 ) (0.25, 0.5 ) (0.5 , 0.5 ) (0.75, 0.5 ) (1.  , 0.5 )]
#  [(0.  , 0.75) (0.25, 0.75) (0.5 , 0.75) (0.75, 0.75) (1.  , 0.75)]
# [(0.  , 1.  ) (0.25, 1.  ) (0.5 , 1.  ) (0.75, 1.  ) (1.  , 1.  )]]
```

47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))
```py
# 47. 2つの配列 X と Y が与えられたとき、コーシー行列 C (Cij =1/(xi - yj)) を作成する
#########################################################
# Author: Evgeni Burovski
X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))
# 3638.163637117973
#########################################################
# ■ 解説
print(X)
print(Y)
# [0 1 2 3 4 5 6 7]
# [0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5]
print(np.subtract.outer(X, Y))
# [[-0.5 -1.5 -2.5 -3.5 -4.5 -5.5 -6.5 -7.5]
#  [ 0.5 -0.5 -1.5 -2.5 -3.5 -4.5 -5.5 -6.5]
#  [ 1.5  0.5 -0.5 -1.5 -2.5 -3.5 -4.5 -5.5]
#  [ 2.5  1.5  0.5 -0.5 -1.5 -2.5 -3.5 -4.5]
#  [ 3.5  2.5  1.5  0.5 -0.5 -1.5 -2.5 -3.5]
#  [ 4.5  3.5  2.5  1.5  0.5 -0.5 -1.5 -2.5]
#  [ 5.5  4.5  3.5  2.5  1.5  0.5 -0.5 -1.5]
#  [ 6.5  5.5  4.5  3.5  2.5  1.5  0.5 -0.5]]

# 行列式は linalg の det 関数、逆行列は linalg の inv 関数で計算します。
A = np.array([[1, 2], [3, 4]])
d = np.linalg.det(A)
print(d)
# -2.0000000000000004
```

48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)
```py
# 48. numpy スカラー型のそれぞれについて、表現可能な最小値と最大値を表示する (★★☆)
#########################################################
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
# -128
# 127
# -2147483648
# 2147483647
# -9223372036854775808
# 9223372036854775807

for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)
# -3.4028235e+38
# 3.4028235e+38
# 1.1920929e-07
# -1.7976931348623157e+308
# 1.7976931348623157e+308
# 2.220446049250313e-16
#########################################################
# ■ 解説
# eps:float
# The smallest representable positive number 
```

49. How to print all the values of an array? (★★☆)
```py
# 49. 配列のすべての値を表示する方法は? (★★☆)
#########################################################
np.set_printoptions(precision=1)
Z = np.zeros((8, 8))
print(Z)
# [[0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0.]]
```

50. How to find the closest value (to a given scalar) in a vector? (★★☆)
```py
# 50. ベクトルの中で指定したスカラーに最も近い値を見つける方法は? (★★☆)
#########################################################
Z = np.arange(100)
v = np.random.uniform(0, 100)
index = (np.abs(Z-v)).argmin()
print(Z[index])
# 25
#########################################################
# ■ 解説
Z
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
#        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
#        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
#        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])
v
# 25.09584740973364
np.abs(Z-v)
# array([25.1, 24.1, 23.1, 22.1, 21.1, 20.1, 19.1, 18.1, 17.1, 16.1, 15.1,
#        14.1, 13.1, 12.1, 11.1, 10.1,  9.1,  8.1,  7.1,  6.1,  5.1,  4.1,
#         3.1,  2.1,  1.1,  0.1,  0.9,  1.9,  2.9,  3.9,  4.9,  5.9,  6.9,
#         7.9,  8.9,  9.9, 10.9, 11.9, 12.9, 13.9, 14.9, 15.9, 16.9, 17.9,
#        18.9, 19.9, 20.9, 21.9, 22.9, 23.9, 24.9, 25.9, 26.9, 27.9, 28.9,
#        29.9, 30.9, 31.9, 32.9, 33.9, 34.9, 35.9, 36.9, 37.9, 38.9, 39.9,
#        40.9, 41.9, 42.9, 43.9, 44.9, 45.9, 46.9, 47.9, 48.9, 49.9, 50.9,
#        51.9, 52.9, 53.9, 54.9, 55.9, 56.9, 57.9, 58.9, 59.9, 60.9, 61.9,
#        62.9, 63.9, 64.9, 65.9, 66.9, 67.9, 68.9, 69.9, 70.9, 71.9, 72.9,
#        73.9])
(np.abs(Z-v)).argmin()
# 25

# np.random.uniform(1, 10, 3)　1以上10未満の一様乱数3個の配列。
# [ 8.77776521  4.21775806  9.269332  ]
```

* [To Top](#Top)


# 51

51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)
```py
# 51. 位置 (x,y) と 色 (r,g,b) を表現する構造化配列を生成する (★★☆)
#########################################################
Z = np.zeros(10, [ ('position', [ ('x', float, 1), ('y', float, 1)]),
                   ('color',    [ ('r', float, 1), ('g', float, 1), ('b', float, 1)])])
print(Z)
# [((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))
#  ((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))
#  ((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))
#  ((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))
#  ((0., 0.), (0., 0., 0.)) ((0., 0.), (0., 0., 0.))]
```

52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)
```py
# 52. shape属性が (100, 2) の座標を表現する乱数のベクトルがあるとき、各点ごとの距離を求める
#########################################################
Z = np.random.random((3, 2))
print(Z)
X, Y = np.atleast_2d(Z[:,0], Z[:,1])
print(X)
print(Y)
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2 )
print(D)
# [[0.2 0.7]
#  [0.3 0. ]
#  [0.  0.3]]
# [[0.2 0.3 0. ]]
# [[0.7 0.  0.3]]
# [[0.  0.7 0.5]
#  [0.7 0.  0.4]
#  [0.5 0.4 0. ]]

# Much faster with scipy
import scipy
# Thanks Gavin Heverly-Coulson (#issue 1)
import scipy.spatial
Z = np.random.random((3, 2))
D = scipy.spatial.distance.cdist(Z, Z)
print(D)
#########################################################
# ■ 解説
# np.at_least_2d()は、入力されたndarrayの次元数が最低でも2になるように次元を拡張する関数のようです。
Z = np.array([[1, 1], [3, 1], [2, 5]])
X, Y = Z[:,0], Z[:,1]
X.shape, Y.shape
# ((3,), (3,))
X, Y = np.atleast_2d(Z[:, 0], Z[:, 1])
X.shape, Y.shape
# ((1, 3), (1, 3))
X-X.T
# array([[ 0,  2,  1],
#        [-2,  0, -1],
#        [-1,  1,  0]])
```

53. How to convert a float (32 bits) array into an integer (32 bits) in place?
```py
# 53. 浮動小数点 (32 bits) 配列を整数型 (32 bits) 配列に変換する方法は?
#########################################################
Z = np.arange(10, dtype=np.float32)
Z = Z.astype(np.int32, copy=False)
print(Z)
# [0 1 2 3 4 5 6 7 8 9]
```

54. How to read the following file? (★★☆)
```py
# 54. 以下のファイル(文字列)を読む方法は? (★★☆)
#########################################################
 1, 2, 3, 4, 5
 6,  ,  , 7, 8
  ,  , 9,10,11
#########################################################
from io import StringIO
# Fake file
s = StringIO("""1, 2, 3, 4, 5\n
                6,  ,  , 7, 8\n
                 ,  , 9,10,11\n""")
Z = np.genfromtxt(s, delimiter=",", dtype=np.int)
print(Z)
# [[ 1  2  3  4  5]
#  [ 6 -1 -1  7  8]
#  [-1 -1  9 10 11]]
#########################################################
# ■ 解説
# np.genfromtxt()を使うと、欠損値を含んでいたり複数の異なるデータ型を含んでいたりする、より複雑な構造のCSVファイルの読み込みが可能。
```

55. What is the equivalent of enumerate for numpy arrays? (★★☆)
```py
# 55. numpy 配列で enumerate に相当するものは何ですか?
#########################################################
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
   print(index, value)
for index in np.ndindex(Z.shape):
   print(index, Z[index])
# (0, 0) 0
# (0, 1) 1
# (0, 2) 2
# (1, 0) 3
# (1, 1) 4
# (1, 2) 5
# (2, 0) 6
# (2, 1) 7
# (2, 2) 8

# (0, 0) 0
# (0, 1) 1
# (0, 2) 2
# (1, 0) 3
# (1, 1) 4
# (1, 2) 5
# (2, 0) 6
# (2, 1) 7
# (2, 2) 8
```

56. Generate a generic 2D Gaussian-like array (★★☆)
```py
# 56. 一般的な2次元正規分布の配列を生成する (★★☆)
#########################################################
X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
D = np.sqrt(X*X + Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D - mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)
```

57. How to randomly place p elements in a 2D array? (★★☆)
```py
# 57. 2次元配列中に p 個の要素をランダムに配置する方法は? (★★☆)
#########################################################
# Author: Divakar
n = 10
p = 3
Z = np.zeros((n, n))
np.put(Z, np.random.choice(range(n*n), p, replace=False), 1)
print(Z)
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
```

58. Subtract the mean of each row of a matrix (★★☆)
```py
# 58. 行列の各行からその平均を減算する (★★☆)
#########################################################
# Author: Warren Weckesser
X = np.random.rand(5, 10)
# Recent versions of numpy
Y = X - X.mean(axis=1, keepdims=True)
# Older versions of numpy
Y = X - X.mean(axis=1).reshape(-1, 1)
print(Y)
# [[ 0.17467156  0.39610569 -0.15474385 -0.10768295 -0.30852725 -0.23045708 0.30775199 -0.20708909  0.31331994 -0.18334896]
#  [-0.34197514 -0.25747791  0.20978006  0.29652348  0.1252613  -0.01937976 0.19479692 -0.36073304  0.28232472 -0.12912062]
#  [-0.32253107  0.12600989 -0.47445115  0.27678715 -0.04236603 -0.49955442 0.3475865   0.33831996 -0.04925552  0.29945469]
#  [-0.32450816 -0.0818643   0.29232411  0.31703995  0.54894657 -0.31751116 0.48092611 -0.3087898  -0.33459605 -0.27196727]
#  [-0.30280297 -0.23644529 -0.20628806 -0.30144542  0.25597645 -0.01116676 0.45472822  0.0503685  -0.30044417  0.5975195 ]]
```

59. How to sort an array by the nth column? (★★☆)
```py
# 59. 配列を n 番目の列でソートする方法は? (★★☆)
#########################################################
# Author: Steve Tjoa
Z = np.random.randint(0, 10, (3, 3))
print(Z)
print(Z[Z[:, 1].argsort()])
# [[7 2 5]
#  [9 0 3]
#  [0 5 7]]
# [[9 0 3]
#  [7 2 5]
#  [0 5 7]]
#########################################################
# ■ 解説
print(Z[:, 1])
# [2 0 5]
print(Z[:, 1].argsort())
# [1 0 2]
```

60. How to tell if a given 2D array has null columns? (★★☆)
```py
# 60. 与えられた2次元配列に全てがゼロの列があるかを調べる方法は? (★★☆)
#########################################################
# Author: Warren Weckesser
Z = np.random.randint(0, 3, (3, 10))
print(Z)
print((~Z.any(axis=0)).any())
# [[2 1 0 0 1 0 1 0 0 0]
#  [2 1 2 2 2 0 0 0 2 0]
#  [2 0 1 0 1 0 2 2 0 0]]
# True
#########################################################
# ■ 解説
print(~Z.any(axis=0))
# [False False False False False  True False False False  True]
```

* [To Top](#Top)


# 61

61. Find the nearest value from a given value in an array (★★☆)
```py
# 61. 与えられた値から最も近い値を配列の中に見つける (★★☆)
#########################################################
Z = np.random.uniform(0, 1, 10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print(m)
# 0.504222092418414
#########################################################
# ■ 解説
np.abs(Z - z)
# array([0.00422209, 0.19767948, 0.41034326, 0.27362028, 0.43928241, 0.3790531 , 0.30789328, 0.25880853, 0.18907701, 0.13576864])
np.abs(Z - z).argmin()
# 0

# ndarray.flat: A 1-D iterator over the array.
x = np.arange(1, 7).reshape(2, 3)
# array([[1, 2, 3],
#        [4, 5, 6]])
x.flat[3]
# 
x.T
# array([[1, 4],
#        [2, 5],
#        [3, 6]])
x.T.flat[3]
# 5
```

62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)
```py
# 62. shape属性が (1,3) と (3,1) の2つの配列があるとき、イテレータを使ってその和を計算する方法は? (★★☆)
#########################################################
A = np.arange(3).reshape(3, 1)
B = np.arange(3).reshape(1, 3)
it = np.nditer([A, B, None])
for x, y, z in it: z[...] = x + y
print(it.operands[2])
# [[0 1 2]
#  [1 2 3]
#  [2 3 4]]
#########################################################
# ■ 解説
A = np.array([[1],[2],[3]])
B = np.array([[5,6,7]])
it = np.nditer([A,B])
for x, y in it:
    print("x y:", x, y)
# x y: 1 5
# x y: 1 6
# x y: 1 7
# x y: 2 5
# x y: 2 6
# x y: 2 7
# x y: 3 5
# x y: 3 6
# x y: 3 7
```

63. Create an array class that has a name attribute (★★☆)
```py
# 63. name属性を持つ配列クラスを作成する (★★☆)
#########################################################
class NamedArray(np.ndarray):
   def __new__(cls, array, name="no name"):
       obj = np.asarray(array).view(cls)
       obj.name = name
       return obj
   def __array_finalize__(self, obj):
       if obj is None: return
       self.info = getattr(obj, 'name', "no name")

Z = NamedArray(np.arange(10), "range_10")
print (Z.name)
# range_10
#########################################################
# ■ 解説
Z = NamedArray(np.arange(10))
print (Z.name)
# no name
```

64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)
```py
# 64. あるベクトルが与えられたとき、第2のベクトルにより添え字指定されたそれぞれの要素に 1 を加える方法は (添え字は繰り返されることに注意)? (★★★)
#########################################################
# Author: Brett Olsen
Z = np.ones(10)
I = np.random.randint(0, len(Z), 20)
Z += np.bincount(I, minlength=len(Z))
print(Z)
# Another solution
# Author: Bartosz Telenczuk
np.add.at(Z, I, 1)
print(Z)
[3. 2. 3. 3. 5. 2. 1. 3. 5. 3.]
[5. 3. 5. 5. 9. 3. 1. 5. 9. 5.]
#########################################################
# ■ 解説
I
# array([1, 2, 2, 4, 9, 5, 4, 8, 8, 3, 8, 4, 7, 0, 4, 0, 3, 9, 7, 8])
np.bincount(I, minlength=len(Z))
# array([2, 1, 2, 2, 4, 1, 0, 2, 4, 2], dtype=int64)
```

65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)
```py
# 65. 添え字リスト (I) に基づきベクトル (X) の要素を配列 (F) に累積する方法は? (★★★)
#########################################################
# Author: Alan G Isaac
X = [1, 2, 3, 4, 5, 6]
I = [1, 3, 9, 3, 4, 1]
F = np.bincount(I, X)
print(F)
# [0. 7. 0. 6. 5. 0. 0. 0. 0. 3.]
#########################################################
# ■ 解説
# np.bincount(I,X)は、np.bincount(I, weights=X)と等価です。つまり、値の生起回数に重みを与えます。
```

66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★)
```py
# 66. 画像が (w, h, 3) (dtype=ubyte) のとき、ユニークな色の数を計算する(★★★)
#########################################################
# Author: Nadav Horesh
w, h = 4, 4
I = np.random.randint(0, 2, (h, w, 3)).astype(np.ubyte)
F = I[..., 0] * 256 * 256 + I[..., 1] * 256 + I[..., 2]
n = len(np.unique(F))
print(np.unique(I))
[0 1]
#########################################################
# ■ 解説
# F = I[...,0]*(256*256) + I[...,1]*256 +I[...,2]により(R, G, B)の値をR * 256 * 256 + G * 256 + Bという単一の値に変換したあと、np.unique()で同一の色を持つ要素を削除して、残った要素数を数えています。
I
# array([[[0, 0, 1],
#         [0, 0, 0],
#         [1, 1, 0],
#         [0, 0, 1]],
# 
#        [[1, 1, 0],
#         [0, 1, 1],
#         [1, 1, 1],
#         [0, 1, 1]],
# 
#        [[1, 1, 1],
#         [0, 1, 0],
#         [0, 1, 1],
#         [1, 0, 0]],
# 
#        [[1, 0, 1],
#         [1, 1, 1],
#         [0, 0, 0],
#         [1, 0, 0]]], dtype=uint8)
F
# array([[  1,   0, 256,   1],
#        [256, 257, 257, 257],
#        [257, 256, 257,   0],
#        [  1, 257,   0,   0]], dtype=uint16)
```

67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)
```py
# 67. 4次元配列のとき、一気に最後の2軸で合計する方法は? (★★★)
#########################################################
A = np.random.randint(0, 10, (3, 4, 3, 4))
# solution by passing a tuple of axes (introduced in numpy 1.7.0)
sum = A.sum(axis=(-2, -1))
print(sum)
# solution by flattening the last two dimensions into one
# (useful for functions that don't accept tuples for axis argument)
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)
# [[46 56 68 58]
#  [28 47 40 49]
#  [53 53 65 53]]
# [[46 56 68 58]
#  [28 47 40 49]
#  [53 53 65 53]]
```

68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset indices? (★★★)
```py
# 68. 1次元ベクトル D を考えたとき、部分集合を添え字で記述する同じサイズのベクトル S を使用して、D の部分集合の平均を計算する方法は？
#########################################################
# Author: Jaime Fernández del Río
D = np.random.uniform(0, 1, 100)
S = np.random.randint(0, 10, 100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)
# Pandas solution as a reference due to more intuitive code
import pandas as pd
print(pd.Series(D).groupby(S).mean())
# [0.53184825 0.45105339 0.43915429 0.49048519 0.39412586 0.48825946 0.56079188 0.48646984 0.39619038 0.21918379]
# 0    0.531848
# 1    0.451053
# 2    0.439154
# 3    0.490485
# 4    0.394126
# 5    0.488259
# 6    0.560792
# 7    0.486470
# 8    0.396190
# 9    0.219184
# dtype: float64
```

69. How to get the diagonal of a dot product? (★★★)
```py
# 69. ドット積の対角要素を取得する方法は? (★★★)
#########################################################
# Author: Mathieu Blondel
A = np.random.uniform(0, 1, (5, 5))
B = np.random.uniform(0, 1, (5, 5))
# Slow version  
np.diag(np.dot(A, B))
# Fast version
np.sum(A * B.T, axis=1)
# Faster version
np.einsum("ij,ji->i", A, B)
# array([1.13600527, 1.26483936, 1.00529482, 0.75508906, 2.15589049])
#########################################################
# ■ 解説
# 第1の方法 np.diag(np.dot(A, B)) は愚直にAとBの積をとり、その対角成分を取得しています。対角成分以外の要素も計算しているので無駄が多いと言えます。
# 第2の方法 np.sum(A * B.T, axis=1) では、要素ごとの積をとる*を使用しています。紙と鉛筆で確かめるとわかりますが、AとBの積をとったときの対角成分に相当する部分のみを演算しているので高速化が期待できます。
# 第3の方法 np.einsum('ij,ji->i', A, B) は初めて見ました。アインシュタインの縮約記号にのっとって演算を行うための関数だそうです。

a = np.arange(25).reshape(5,5)
b = np.arange(25).reshape(5,5)*2
print("A = ",a)
print("B =", b)

# A = [[ 0  1  2  3  4]
#      [ 5  6  7  8  9]
#      [10 11 12 13 14]
#      [15 16 17 18 19]
#      [20 21 22 23 24]]
# B = [[ 0  2  4  6  8]
#      [10 12 14 16 18]
#      [20 22 24 26 28]
#      [30 32 34 36 38]
#      [40 42 44 46 48]]

np.einsum("ii",a) #60
np.einsum("ii->i",a) #[  0,  6, 12, 18, 24]
np.einsum("ij->i",a) #[ 10, 35, 60, 85, 110]
np.einsum("ij->j",a) #[ 50, 55, 60, 65, 70]
np.einsum("ij->ji",a) 
# array([[ 0,  5, 10, 15, 20],
#        [ 1,  6, 11, 16, 21],
#        [ 2,  7, 12, 17, 22],
#        [ 3,  8, 13, 18, 23],
#        [ 4,  9, 14, 19, 24]])

np.einsum("ij,jk->ik", a, b) #AB
np.einsum("ij,jk->ki", a, b) #BA 足(ik)が変わったのに注意
np.einsum("ij,ij->ij", a, b) #いわゆるアダマール積
```

70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)
```py
# 70. ベクトル \[1, 2, 3, 4, 5\] があるとき、それぞれの値の間に連続する3つのゼロをはさむ新しいベクトルを生成する方法は? (★★★)
#########################################################
# Author: Warren Weckesser
Z = np.array([1, 2, 3, 4, 5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z) - 1) * (nz))
Z0[:: nz + 1] = Z
print(Z0)
# [1. 0. 0. 0. 2. 0. 0. 0. 3. 0. 0. 0. 4. 0. 0. 0. 5.]
```

* [To Top](#Top)


# 71

71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)
```py
# 71. (5,5,3) 次元の配列があるとき、それに (5,5) 次元の配列を掛ける方法は? (★★★)
#########################################################
A = np.ones((5, 5, 3))
B = 2 * np.ones((5, 5))
print(A * B[:, :, None])
# [[[2. 2. 2.]
#   [2. 2. 2.]
#   [2. 2. 2.]
#   [2. 2. 2.]
#   [2. 2. 2.]]
# 
#  [[2. 2. 2.]
#   [2. 2. 2.]
#   [2. 2. 2.]
#   [2. 2. 2.]
#   [2. 2. 2.]]
# 
#  [[2. 2. 2.]
#   [2. 2. 2.]
#   [2. 2. 2.]
#   [2. 2. 2.]
#   [2. 2. 2.]]
# 
#  [[2. 2. 2.]
#   [2. 2. 2.]
#   [2. 2. 2.]
#   [2. 2. 2.]
#   [2. 2. 2.]]
# 
#  [[2. 2. 2.]
#   [2. 2. 2.]
#   [2. 2. 2.]
#   [2. 2. 2.]
#   [2. 2. 2.]]]
#########################################################
# ■ 解説
(5,5)のshapeを持つBに対してB[:,:,None]とすると、(5,5,1)のshapeになるようです。
```

72. How to swap two rows of an array? (★★★)
```py
72. 配列の2つの行を交換する方法は? (★★★)
#########################################################
A = np.arange(25).reshape(5,5)
A
# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19],
#        [20, 21, 22, 23, 24]])

# Slicing
A[0:2, :] = A[1::-1, :]
A
# array([[ 5,  6,  7,  8,  9],
#        [ 0,  1,  2,  3,  4],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19],
#        [20, 21, 22, 23, 24]])

# Advanced Indexing
A = np.arange(25).reshape(5,5)
A[[0, 1]] = A[[1, 0]]
A
# array([[ 5,  6,  7,  8,  9],
#        [ 0,  1,  2,  3,  4],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19],
#        [20, 21, 22, 23, 24]])
#########################################################
# ■ 解説
A[0, 1]
# 1
A[[0, 1]] # このアクセス方法は"Advanced Indexing"と呼ばれる方法です
# array([[0, 1, 2, 3, 4],
#        [5, 6, 7, 8, 9]])
A[0:2, :]
# array([[0, 1, 2, 3, 4],
#        [5, 6, 7, 8, 9]])
# Advanced Indexingでは常にコピーが返るのに対し、slicingではビューが返るという違いがあるので注意です。
```

73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the triangles (★★★)
```py
# 73. 10個の三角形を記述する10個の三つ組みがあるとき (共有された頂点を持つ) 、すべての三角形を構成するユニークな線分の集合を見つける (★★★)
#########################################################
# Author: Nicolas P. Rougier
faces = np.random.randint(0, 100, (10, 3))
F = np.roll(faces.repeat(2, axis=1), -1, axis=1)
F = F.reshape(len(F) * 3, 2)
F = np.sort(F, axis=1)
G = F.view( dtype=[('p0', F.dtype), ('p1', F.dtype)] )
G = np.unique(G)
print(G)
# [( 7, 16) ( 7, 39) ( 7, 48) ( 7, 90) (12, 45) (12, 57) (15, 62) (15, 85)
#  (16, 48) (18, 43) (18, 88) (19, 44) (37, 56) (37, 92) (39, 61) (39, 82)
#  (39, 90) (42, 69) (42, 91) (43, 88) (44, 44) (44, 71) (44, 95) (45, 57)
#  (56, 92) (61, 82) (62, 85) (69, 91) (71, 95)]
#########################################################
# ■ 解説
faces
# array([[56, 92, 37],
#        [90,  7, 39],
#        [19, 44, 44],
#        [ 7, 48, 16],
#        [44, 95, 71],
#        [42, 91, 69],
#        [61, 82, 39],
#        [85, 62, 15],
#        [18, 88, 43],
#        [45, 57, 12]])
F
# array([[56, 92],
#        [37, 92],
#        [37, 56],
#        [ 7, 90],
#        [ 7, 39],
#        [39, 90],
#        [19, 44], #
#        [44, 44],
#        [19, 44], #
#        [ 7, 48],
#        [16, 48],
#        [ 7, 16],
#        [44, 95],
#        [71, 95],
#        [44, 71],
#        [42, 91],
#        [69, 91],
#        [42, 69],
#        [61, 82],
#        [39, 82],
#        [39, 61],
#        [62, 85],
#        [15, 62],
#        [15, 85],
#        [18, 88],
#        [43, 88],
#        [18, 43],
#        [45, 57],
#        [12, 57],
#        [12, 45]])
```

74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)
```py
# 74. bincountで与えられる配列 C があるとき、np.bincount(A) == C であるような配列 A の生成方法は? (★★★)
#########################################################
# Author: Jaime Fernández del Río
C = np.bincount([1, 1, 2, 3, 4, 4, 6])
A = np.repeat(np.arange(len(C)), C)
print(A)
# [1 1 2 3 4 4 6]
#########################################################
# ■ 解説
np.arange(len(C)), C
# (array([0, 1, 2, 3, 4, 5, 6]), array([0, 2, 1, 1, 2, 0, 1], dtype=int64))
```

75. How to compute averages using a sliding window over an array? (★★★)
```py
# 75. 配列にスライドウィンドウを使って平均を計算する方法は? (★★★)
#########################################################
# Author: Jaime Fernández del Río
def moving_average(a, n=3) :
   ret = np.cumsum(a, dtype=float)
   ret[n:] = ret[n:] - ret[:-n]
   return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))
# [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17. 18.]
#########################################################
# ■ 解説
a = np.array([1, 2, 3, 4, 5, 6])
np.cumsum(a)
# array([ 1,  3,  6, 10, 15, 21])
```

76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) (★★★)
```py
# 76. 1次元の配列 Z があるとき、最初の行が (Z\[0\],Z\[1\],Z\[2\]) で、それに続く行はそれぞれ 1 だけシフトされた2次元配列を生成する (最後の行は、(Z\[-3\],Z\[-2\],Z\[-1\]) になる) (★★★)
#########################################################
# Author: Joe Kington / Erik Rigtorp
from numpy.lib import stride_tricks
def rolling(a, window):
   shape = (a.size - window + 1, window)   # この場合 (8, 3)
   strides = (a.itemsize, a.itemsize)      # この場合 (8, 8)
   return stride_tricks.as_strided(a, shape=shape, strides=strides) # (a)
Z = rolling(np.arange(10), 3)
print(Z)
# [[0 1 2]
#  [1 2 3]
#  [2 3 4]
#  [3 4 5]
#  [4 5 6]
#  [5 6 7]
#  [6 7 8]
#  [7 8 9]]
#########################################################
# ■ 解説
# stride_tricks.as_stridedは、stridesの値を任意に指定することで、ポインタの進め方を偽装する関数です。
# 式(a)は、「shapeが(10,)であるndarray aを、stridesを(8, 8)と偽装して、shapeが(8, 3)になるように舐めていく」という意味になります。stridesが(8, 8)ということは、「ポインタを一個右に動かすにも、ポインタを一個下に動かすにも、8バイト移動する」ということになります。
```

77. How to negate a boolean, or to change the sign of a float inplace? (★★★)
```py
# 77. ブール値を否定する方法、また、浮動小数点の符号を変更する方法は? (★★★)
#########################################################
# Author: Nathaniel J. Smith
Z = np.random.randint(0, 2, 10)
np.logical_not(Z, out=Z)
print(Z)
Z = np.random.uniform(-1.0, 1.0, 10)
np.negative(Z, out=Z)
# [0 1 0 0 1 0 1 0 0 0]
# array([ 0.30177043,  0.43012631,  0.93765857,  0.69757218,  0.68864177, -0.79638609,  0.71618958, -0.29460162, -0.19384768, -0.92111143])
```

78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])? (★★★)
```py
# 78. 直線 (2次元) を記述する2組の点 P0、P1そして点 p があるとき、点 p から直線 i (P0\[i\],P1\[i\]) までの距離を計算する方法は? (★★★)
#########################################################
def distance(P0, P1, p):
   T = P1 - P0
   L = (T**2).sum(axis=1)
   U = -((P0[:, 0] - p[...,0]) * T[:, 0] + (P0[:, 1] - p[..., 1]) * T[:, 1]) / L
   U = U.reshape(len(U), 1)
   D = P0 + U*T - p
   return np.sqrt((D**2).sum(axis=1))
P0 = np.random.uniform(-10, 10, (10, 2))
P1 = np.random.uniform(-10, 10, (10, 2))
p  = np.random.uniform(-10, 10, ( 1, 2))
print(distance(P0, P1, p))
# [10.72352647 13.09825264 13.73247846  2.11357729 14.96245468  5.27188098  0.40259183  1.10263579  5.30508889  3.24204977]
```

79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)
```py
# 79. 直線 (2次元) を記述する2組の点 P0、P1 そして1組の点 P があるとき、それぞれの点 (P\[j\]) から各線 i (P0\[i\],P1\[i\]) までの距離を計算する方法は? (★★★)
#########################################################
# Author: Italmassov Kuanysh
# based on distance function from previous question
P0 = np.random.uniform(-10, 10, (5, 2))
P1 = np.random.uniform(-10, 10, (5, 2))
p = np.random.uniform(-10, 10, (5, 2))
print(np.array([distance(P0, P1, p_i) for p_i in p]))
# [[ 5.59509606 11.99011178  4.57584585  1.38373154  2.55252384]
#  [11.37395291  4.92650438  4.16825027 14.5409501   9.81340396]
#  [ 4.89889266 11.51905421  6.64931316  0.24980589  5.4154482 ]
#  [14.3453139   8.24277824  9.56151317 15.77300207  7.01180281]
#  [ 1.88569673  4.68537175  2.38912111  6.30889709  7.51266531]]
```

80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a fill value when necessary) (★★★)
```py
# 80. 任意の配列に対して、定められたshape属性で、指定された要素が中央に配置されるような一部分を抽出する関数を書く（必要に応じて `fill` 値でパディング） (★★★)
#########################################################
# Author: Nicolas Rougier
Z = np.random.randint(0, 10, (10, 10))
shape = (5, 5)
fill  = 0
position = (0, 0)

R = np.ones(shape, dtype=Z.dtype) * fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape), )).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P - Rs // 2)
Z_stop  = (P + Rs // 2) + Rs % 2

R_start = (R_start - np.minimum(Z_start, 0)).tolist()
Z_start = (np.maximum(Z_start, 0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs, 0))).tolist()
Z_stop = (np.minimum(Z_stop, Zs)).tolist()

r = [slice(start, stop) for start, stop in zip(R_start, R_stop)]
z = [slice(start, stop) for start, stop in zip(Z_start, Z_stop)]
R[r] = Z[z]
print(Z)
print(R)

# [[6 3 2 9 2 7 7 4 4 5]
#  [3 9 2 6 4 4 9 4 1 8]
#  [1 9 1 3 3 4 1 8 3 1]
#  [0 9 4 1 9 0 8 6 2 0]
#  [6 3 7 3 6 8 2 7 8 3]
#  [6 2 9 1 4 6 3 1 6 5]
#  [3 4 3 7 0 0 5 0 3 2]
#  [4 2 8 5 8 8 3 8 3 3]
#  [9 8 5 8 2 4 3 5 1 6]
#  [9 5 2 4 7 1 5 6 0 2]]
# [[0 0 0 0 0]
#  [0 0 0 0 0]
#  [0 0 6 3 2]
#  [0 0 3 9 2]
#  [0 0 1 9 1]]
```

* [To Top](#Top)


# 81

81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)
```py
# 81. 配列 Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14] があるとき、配列 R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]] の生成方法は? (★★★)
#########################################################
# Author: Stefan van der Walt
Z = np.arange(1, 15, dtype=np.uint32)
R = stride_tricks.as_strided(Z, (11, 4), (4, 4))
print(R)
# [[ 1  2  3  4]
#  [ 2  3  4  5]
#  [ 3  4  5  6]
#  [ 4  5  6  7]
#  [ 5  6  7  8]
#  [ 6  7  8  9]
#  [ 7  8  9 10]
#  [ 8  9 10 11]
#  [ 9 10 11 12]
#  [10 11 12 13]
#  [11 12 13 14]]
```

82. Compute a matrix rank (★★★)
```py
# 82. 行列のランク計算 (★★★)
#########################################################
# Author: Stefan van der Walt
Z = np.random.uniform(0, 1, (10, 10))
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
rank = np.sum(S > 1e-10)
print(rank)
```

83. How to find the most frequent value in an array?
```py
# 83. 配列の中での最頻値の求め方は?
#########################################################
Z = np.random.randint(0, 10, 50)
print(np.bincount(Z).argmax())
# 9
```

84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)
```py
# 84. 10x10 の乱数の行列から隣接するすべての 3x3 ブロックを抽出する (★★★)
#########################################################
# Author: Chris Barker
Z = np.random.randint(0, 5, (10, 10))
n = 3
i = 1 + (Z.shape[0] - 3)
j = 1 + (Z.shape[1] - 3)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides = Z.strides + Z.strides)
# C = stride_tricks.as_strided(Z, shape=(8, 8, 3, 3), strides = (40, 4, 40, 4))
print(C)
```

85. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)
```py
# 85. Z[i,j] == Z[j,i] となるような2次元配列のサブクラスを作成する (★★★)
#########################################################
# Author: Eric O. Lebigot
# Note: only works for 2d array and value setting using indices
class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i, j = index
        super(Symetric, self).__setitem__((i, j), value)
        super(Symetric, self).__setitem__((j, i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(np.random.randint(0, 10, (5, 5)))
S[2, 3] = 42
print(S)

# [[ 1  4 12  6 13]
#  [ 4  2  6 12  6]
#  [12  6  9 10  4]
#  [ 6 12 10  5  9]
#  [13  6  4  9  5]]
#########################################################
# ■ 解説
print(Z)
# [[1 3 7 6 5]
#  [1 2 6 9 0]
#  [5 0 9 6 4]
#  [0 3 4 5 9]
#  [8 6 0 0 5]]
print(Z.diagonal())
# [1 2 9 5 5]
print(np.diag(Z.diagonal()))
# [[1 0 0 0 0]
#  [0 2 0 0 0]
#  [0 0 9 0 0]
#  [0 0 0 5 0]
#  [0 0 0 0 5]]

# view：元の配列と同じメモリを参照しているもの
# https://deepage.net/features/numpy-copyview.html
```

86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)
```py
# 86. p 個の shape属性 (n,n) の行列と p 個の shape属性 (n,1) のベクトルがあるとき、一度に p 個の行列積の総和を計算する方法は? (結果は shape属性が (n,1)) (★★★)
#########################################################
# Author: Stefan van der Walt
p, n = 10, 5
M = np.ones((p, n, n))
V = np.ones((p, n, 1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)
# [[50.]
#  [50.]
#  [50.]
#  [50.]
#  [50.]]

# It works, because:
# M is (p, n, n)
# V is (p, n, 1)
# Thus, summing over the paired axes 0 and 0 (of M and V independently),
# and 2 and 1, to remain with a (n,1) vector.
#########################################################
# ■ 解説
# The idea with tensordot is pretty simple - We input the arrays and the respective axes along which the sum-reductions are intended. The axes that take part in sum-reduction are removed in the output and all of the remaining axes from the input arrays are spread-out as different axes in the output keeping the order in which the input arrays are fed.
A = np.random.randint(2, size=(2, 6, 5))
B = np.random.randint(2, size=(3, 2, 4))
np.tensordot(A, B, axes=((0), (1))).shape
# (6, 5, 3, 4)

A = np.random.randint(2, size=(2, 3, 5))
B = np.random.randint(2, size=(3, 2, 4))
np.tensordot(A, B, axes=((0, 1), (1, 0))).shape
# (5, 4)

a = np.arange(60.).reshape(3,4,5)
b = np.arange(24.).reshape(4,3,2)
c = np.tensordot(a,b, axes=([1,0],[0,1]))
c.shape
# (5, 2)
c
# array([[4400., 4730.],
#        [4532., 4874.],
#        [4664., 5018.],
#        [4796., 5162.],
#        [4928., 5306.]])
# A slower but equivalent way of computing the same...
d = np.zeros((5,2))
for i in range(5):
  for j in range(2):
    for k in range(3):
      for n in range(4):
        d[i,j] += a[k,n,i] * b[n,k,j]
c == d
# array([[ True,  True],
#        [ True,  True],
#        [ True,  True],
#        [ True,  True],
#        [ True,  True]])
```

87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)
```py
# 87. 16x16 の配列があるとき、ブロック合計 (ブロック・サイズは 4x4 ) の取得方法は? (★★★)
#########################################################
# Author: Robert Kern
Z = np.ones((16, 16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print(S)
# [[16. 16. 16. 16.]
#  [16. 16. 16. 16.]
#  [16. 16. 16. 16.]
#  [16. 16. 16. 16.]]
#########################################################
# ■ 解説
np.arange(0, Z.shape[0], k)
# array([ 0,  4,  8, 12])

x = np.arange(16).reshape(4,4)
# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11],
#        [12, 13, 14, 15]])

np.add.reduceat(x,[0])
# array([[24, 28, 32, 36]], dtype=int32)
np.add.reduceat(x,[1])
# array([[24, 27, 30, 33]], dtype=int32)
np.add.reduceat(x,[2])
# array([[20, 22, 24, 26]], dtype=int32)
np.add.reduceat(x,[3])
# array([[12, 13, 14, 15]], dtype=int32)

np.add.reduceat(x,[0,1])
# array([[ 0,  1,  2,  3],
#        [24, 27, 30, 33]], dtype=int32)
np.add.reduceat(x,[0,2])
# array([[ 4,  6,  8, 10],
#        [20, 22, 24, 26]], dtype=int32)
np.add.reduceat(x,[0,3])
# array([[12, 15, 18, 21],
#        [12, 13, 14, 15]], dtype=int32)
np.add.reduceat(x,[1,3])
# array([[12, 14, 16, 18],
#        [12, 13, 14, 15]], dtype=int32)
np.add.reduceat(x,[2,3])
# array([[ 8,  9, 10, 11],
#        [12, 13, 14, 15]], dtype=int32)
np.add.reduceat(x,[1,2])
# array([[ 4,  5,  6,  7],
#        [20, 22, 24, 26]], dtype=int32)
```

88. How to implement the Game of Life using numpy arrays? (★★★)
```py
# 88. numpy 配列を使ったライフゲームの実装方法は? (★★★)
#########################################################
# Author: Nicolas Rougier
def iterate(Z):
    # Count neighbours
    N = (Z[0:-2, 0:-2] + Z[0:-2, 1:-1] + Z[0:-2, 2:] +
         Z[1:-1, 0:-2]                 + Z[1:-1, 2:] +
         Z[2:  , 0:-2] + Z[2:  , 1:-1] + Z[2:  , 2:])

    # Apply rules
    birth = (N == 3) & (Z[1:-1, 1:-1] == 0)
    survive = ((N == 2) | (N == 3)) & (Z[1:-1, 1:-1] == 1)
    Z[...] = 0
    Z[1:-1, 1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0, 2, (50, 50))
for i in range(100): Z = iterate(Z)
print(Z)
# [[0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [0 1 1 0 0 0 0 0 0 0]
#  [0 1 0 1 0 1 1 1 0 0]
#  [0 0 1 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]]
#########################################################
# ■ 解説
# Game of Life
# https://d.hatena.ne.jp/keyword/%E3%83%A9%E3%82%A4%E3%83%95%E3%82%B2%E3%83%BC%E3%83%A0
```

89. How to get the n largest values of an array (★★★)
```py
# 89. 配列の n 個の最大値の取得方法は (★★★)
#########################################################
Z = np.arange(10000)
np.random.shuffle(Z)
n = 5

# Slow
print (Z[np.argsort(Z)[-n:]])
# [9995 9996 9997 9998 9999]

# Fast
print (Z[np.argpartition(-Z, n)[:n]])
# [9999 9997 9998 9996 9995]
#########################################################
# ■ 解説
# np.argsort(arr)は、配列arrを昇順にソートしたときの、配列のインデックスを返します。
# np.argpartition(arr, idx)は、配列arrのうち、値が小さいものidx + 1個分のインデックスを返します。必ずしも小さい順になっている保証はないことに注意です。
```

90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)
```py
# 90. 任意の数のベクトルがあるとき、そのデカルト積を生成する (すべての要素のすべての組み合わせ) (★★★)
#########################################################
# Author: Stefan Van der Walt
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print (cartesian(([1, 2, 3], [4, 5], [6, 7])))
# [[1 4 6]
#  [1 4 7]
#  [1 5 6]
#  [1 5 7]
#  [2 4 6]
#  [2 4 7]
#  [2 5 6]
#  [2 5 7]
#  [3 4 6]
#  [3 4 7]
#  [3 5 6]
#  [3 5 7]]
#########################################################
# ■ 解説
grid = np.indices((2, 3))
grid.shape
# (2, 2, 3)
row, col = np.indices((2, 3))
print(row)
print(col)
# [[0 0 0]
#  [1 1 1]]
# [[0 1 2]
#  [0 1 2]]
i, j = np.indices((2, 3), sparse=True)
i
# array([[0],
#        [1]])
j
# array([[0, 1, 2]])
```

* [To Top](#Top)



# 91

01. Import the numpy package under the name np (★☆☆)
```py
# 01. numpy パッケージを `np` の名前でインポートする (★☆☆)
#########################################################
import numpy as np
```

* [To Top](#Top)
