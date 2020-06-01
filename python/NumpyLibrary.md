

# Top

* [np.zeros_like](#zeros_like)
* [np.cumsum](#cumsum)
* [np.linspace](#linspace)
* [np.percentile](#percentile)
* [np.interp](#interp)


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

# A simple check for xp being strictly increasing is:
np.all(np.diff(xp) > 0)
```

[To Top](#Top)
