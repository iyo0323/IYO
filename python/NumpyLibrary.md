

# Top

* [np.percentile](#percentile)


# percentile
(https://analytics-note.xyz/programming/numpy-percentile/)
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

[To Top](#Top)
