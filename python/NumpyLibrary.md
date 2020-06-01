

# Top

* [np.percentile](#percentile)


# percentile
(https://analytics-note.xyz/programming/numpy-percentile/)
```py
# np.percentile
#########################################################
# 中央値や四分位数を一般化した概念に分位数ってのがあります。その中でも特にq/100分位数をqパーセンタイルといい、numpyに専用の関数が用意されています。
import numpy

# 5個の値の3番目の数を返す
data_1 = np.array([3, 12, 3, 7, 10])
print(np.percentile(data_1, 50))  # 7.0

# 6個の値の3番目の数と4番目の数の平均を返す
data_2 = np.array([3, 12, 3, 7, 10, 20])
print(np.percentile(data_2, 50))  # 8.5
```

[To Top](#Top)
