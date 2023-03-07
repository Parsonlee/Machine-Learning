# Pandas

## 关于DataFrame的大小问题
* 设置对应的dtype，可以有效减少DataFrame的大小，但是当写入csv文件时，dtype数据将会丢失。
* `df.to_pickle(df.pickle)`pickle文件读写比csv更快，且文件更小，并且会保存所有的dtype信息。
* `df.to_feather(df.feather)`feather文件读写比csv更快，文件更小。更适合于短期储存，而`parquet` (需要install)更适合长期储存。
* size: csv > pickle > feather > parquet
* speed: csv > parquet > feather > pickle

## concat, merge
* concat: `ignore_index`参数可以重新生成index
* merge: `on='inner/outer'`默认为内连接，outer为外连接

## 函数形式和非函数形式
* 常见的非函数形式:
  ```python
  df2 = df.copy()
  df2['k'] = v
  ```
* 等价的函数形式
  ```python
  df2 = df.assign(k=v)
  ```
  

# Numpy

## 随机模块random
* `random.rand`: 均匀分布抽样
* `random.randint`: 参数为`(low, high, length)`
* `random.randn`: 均值为0，方差为1的正态分布抽样

## 向量扁平化
* `ndarray.ravel()`
* `ndarray.flatten()`

# Visualize (matplotlib, seaborn)

## 子图
* 
  ```python
  fig = plt.figure(..figsize=(10, 10))
  ax1 = fig.add_subplot(2, 2, 1)  # 2行2列第1个子图
  ax2 = fig.add_subplot(2, 2, 2)
  ax1.plot(...)
  ax1.scatter(...)
  ```
* 
  ```python
  fig, axes = plt.subplots(2, 3) # axes是二维数组，2行3列
  axes[0, 0].plot(...)
  ```