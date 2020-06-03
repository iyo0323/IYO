

# Top

* [np.zeros_like](#zeros_like)
* [np.cumsum](#cumsum)
* [np.linspace](#linspace)
* [np.percentile](#percentile)
* [np.interp](#interp)
* [np.einsum](#einsum)


# zeros_like
https://deepage.net/features/numpy-zeros.html
```py
# np.zeros_like
#########################################################
# 元の配列と同じ形状の配列を生成する。
b = np.zeros_like(a)
b = np.zeros(a.shape)

a = np.array([[2, 3, 4], [5, 6, 7]])
np.zeros_like(a)
# array([[0, 0, 0],
#        [0, 0, 0]])
```

# cumsum
https://qiita.com/Sa_qiita/items/fc61f776cef657242e69
```py
# np.cumsum
#########################################################
# 要素を足し合わせたものを、配列として出力する。
a = np.array([1, 2, 3, 4, 5, 6])
#下記どちらの書き方でもOK
np.cumsum(a)
a.cumsum()
# array([ 1,  3,  6, 10, 15, 21])

a = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
# array([[1, 2, 3],
#        [4, 5, 6]])

np.cumsum(a, axis=0)
# array([[1, 2, 3],
#        [5, 7, 9]])

np.cumsum(a, axis=1)
# array([[ 1,  3,  6],
#        [ 4,  9, 15]])

#ちなみに多次元のarrayに対してaxis指定せずにcumsum()実行すると
np.cumsum(a)
# array([ 1,  3,  6, 10, 15, 21])
```

# linspace
https://note.nkmk.me/python-numpy-arange-linspace/
```py
# np.cumsum
#########################################################
# 第一引数startに最初の値、第二引数stopに最後の値、第三引数numに要素数を指定する。それらに応じた間隔（公差）が自動的に算出される。
print(np.linspace(0, 10, 3))
# [ 0.  5. 10.]
print(np.linspace(0, 10, 5))
# [ 0.   2.5  5.   7.5 10. ]

# 引数endpoint
print(np.linspace(0, 10, 5, endpoint=False))
# [0. 2. 4. 6. 8.]

# 逆順に変換
print(np.linspace(0, 10, 5)[::-1])
# [10.   7.5  5.   2.5  0. ]

# 多次元配列に変換
print(np.linspace(0, 10, 12).reshape(3, 4))
# [[ 0.          0.90909091  1.81818182  2.72727273]
#  [ 3.63636364  4.54545455  5.45454545  6.36363636]
#  [ 7.27272727  8.18181818  9.09090909 10.        ]]
```

# percentile
https://analytics-note.xyz/programming/numpy-percentile/
```py
# np.percentile
#########################################################
# 中央値や四分位数を一般化した概念に分位数ってのがあります。
# その中でも特にq/100分位数をqパーセンタイルといい、numpyに専用の関数が用意されています。
import numpy

# 5個の値の3番目の数を返す
data_1 = np.array([3, 12, 3, 7, 10])
print(np.percentile(data_1, 50))  # 7.0

# 6個の値の3番目の数と4番目の数の平均を返す
data_2 = np.array([3, 12, 3, 7, 10, 20])
print(np.percentile(data_2, 50))  # 8.5

data_3 = np.random.randint(0, 2000, 11)
print(data_3)
# [1306  183 1323  266  998 1263 1503 1986  250  305 1397]
for p in range(0, 101, 10):
    print(p, "パーセンタイル・・・", np.percentile(data_3, p))
# 0 パーセンタイル・・・ 183.0
# 10 パーセンタイル・・・ 250.0
# 20 パーセンタイル・・・ 266.0
# 30 パーセンタイル・・・ 305.0
# 40 パーセンタイル・・・ 998.0
# 50 パーセンタイル・・・ 1263.0
# 60 パーセンタイル・・・ 1306.0
# 70 パーセンタイル・・・ 1323.0
# 80 パーセンタイル・・・ 1397.0
# 90 パーセンタイル・・・ 1503.0
# 100 パーセンタイル・・・ 1986.0

data_4 = np.array([15, 52, 100, 73, 102])
print(np.percentile(data_4, 17))
# 40.16
```

# interp
https://numpy.org/doc/stable/reference/generated/numpy.interp.html
```py
# np.interp
#########################################################
# One-dimensional linear interpolation.
xp = [1, 2, 3]
fp = [3, 2, 0]
np.interp(2.5, xp, fp)
# 1.0
np.interp([0, 1, 1.5, 2.72, 3.14], xp, fp)
# array([3.  , 3.  , 2.5 , 0.56, 0.  ])

# Plot an interpolant to the sine function
x = np.linspace(0, 2*np.pi, 10)
y = np.sin(x)
xvals = np.linspace(0, 2*np.pi, 50)
yinterp = np.interp(xvals, x, y)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o')
plt.plot(xvals, yinterp, '-x')
plt.show()

# Complex interpolation:
x = [1.5, 4.0]
xp = [2,3,5]
fp = [1.0j, 0, 2+3j]
np.interp(x, xp, fp)
# array([0.+1.j , 1.+1.5j])

# A simple check for xp being strictly increasing is:
np.all(np.diff(xp) > 0)
```

# einsum
https://www.procrasist.com/entry/einsum
```py
# np.einsum
#########################################################
# Einsteinの縮約記号
# np.einsum('今ある足'->'残したい足')

a = np.arange(25).reshape(5, 5)
b = np.arange(25).reshape(5, 5) * 2
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

# 行列(単体)
np.einsum("ii", a)     #60
np.einsum("ii->i", a)  #[ 0,  6, 12, 18, 24]
np.einsum("ij->i", a)  #[ 10,  35,  60,  85, 110]
np.einsum("ij->j", a)  #[50, 55, 60, 65, 70]
np.einsum("ij->ji", a) #a.T 
# array([[ 0,  5, 10, 15, 20],
#        [ 1,  6, 11, 16, 21],
#        [ 2,  7, 12, 17, 22],
#        [ 3,  8, 13, 18, 23],
#        [ 4,  9, 14, 19, 24]])

# 行列演算
np.einsum("ij, jk->ik", a, b) #AB
# array([[ 300,  320,  340,  360,  380],
#        [ 800,  870,  940, 1010, 1080],
#        [1300, 1420, 1540, 1660, 1780],
#        [1800, 1970, 2140, 2310, 2480],
#        [2300, 2520, 2740, 2960, 3180]])
np.einsum("ij, jk->ki", a, b) #BA 足(ik)が変わったのに注意
# array([[ 300,  800, 1300, 1800, 2300],
#        [ 320,  870, 1420, 1970, 2520],
#        [ 340,  940, 1540, 2140, 2740],
#        [ 360, 1010, 1660, 2310, 2960],
#        [ 380, 1080, 1780, 2480, 3180]])
np.einsum("ij, ij->ij", a, b) #いわゆるアダマール積
# array([[   0,    2,    8,   18,   32],
#        [  50,   72,   98,  128,  162],
#        [ 200,  242,  288,  338,  392],
#        [ 450,  512,  578,  648,  722],
#        [ 800,  882,  968, 1058, 1152]])

# ベクトル演算
v1 = np.arange(3)       #[0, 1, 2]
v2 = np.arange(3) + 1   #[1, 2, 3]
np.einsum("i, i", v1, v2)     #内積
# 8
np.einsum("i, j->ij", v1, v2) #直積
# array([[0, 0, 0],
#        [1, 2, 3],
#        [2, 4, 6]])

# 外積(Cross Product)
# a = (a1, a2, a3)
# b = (b1, b2, b3)
# a × b = (a2b3 - a3b2, a3b1 - a1b3, a1b2 - a2b1)
eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
# array([[[ 0.,  0.,  0.],
#         [ 0.,  0.,  1.],
#         [ 0., -1.,  0.]],
#        [[ 0.,  0., -1.],
#         [ 0.,  0.,  0.],
#         [ 1.,  0.,  0.]],
#        [[ 0.,  1.,  0.],
#         [-1.,  0.,  0.],
#         [ 0.,  0.,  0.]]])
cross_numpy  = np.cross(v1, v2))
# array([-1,  2, -1])
cross_einsum = np.einsum('ijk, i, j->k', eijk, v1, v2))
# array([-1., 2., -1.])

# 行列式(Determinant)
A = np.arange(9).reshape(3, 3)
det_numpy  = np.linalg.det(A)
det_einsum = np.einsum('ijk, i, j, k', eijk, A[0], A[1], A[2]) #0
```

[To Top](#Top)
