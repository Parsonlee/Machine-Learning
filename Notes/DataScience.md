# Pandas

- ## 对空 DataFrame 添加数据
  由于高版本`Pandas`删除了`append`方法，当解构复杂数据完添加到`DataFrame`中时，使用`loc`方法。

```python
# 创建一个空的 DataFrame
df = pd.DataFrame(columns=['A', 'B'])

# 使用 df.iloc 添加数据
# 这里会出现 IndexError，因为 df.iloc 不会自动扩展目标对象的大小
df.iloc[0] = [1, 2]

# 使用 df.loc 添加数据
# 这里不会出现错误，df.loc 会自动扩展目标对象的大小
df.loc[0] = [1, 2]

# 使用append函数
df._append([1, 2], ignore_index=True)
```

- ## 对所有 float 元素取 4 位小数

  `df.round(4)`

- ## 关于 DataFrame 的大小问题

  - 设置对应的 dtype，可以有效减少 DataFrame 的大小，但是当写入 csv 文件时，dtype 数据将会丢失。
  - `df.to_pickle(df.pickle)`pickle 文件读写比 csv 更快，且文件更小，并且会保存所有的 dtype 信息。
  - `df.to_feather(df.feather)`feather 文件读写比 csv 更快，文件更小。更适合于短期储存，而`parquet` (需要 install)更适合长期储存。
  - size: csv > pickle > feather > parquet
  - speed: csv > parquet > feather > pickle

- ## concat, merge

  - concat: `ignore_index`参数可以重新生成 index
  - merge: `on='inner/outer'`默认为内连接，outer 为外连接

- ## 函数形式和非函数形式

  - 常见的非函数形式:
    ```python
    df2 = df.copy()
    df2['k'] = v
    ```
  - 等价的函数形式
    ```python
    df2 = df.assign(k=v)
    ```

- ## 分层索引
  - `index/columns .get_level_values(level=0)`获取 level0 上的索引。
  - `droplevel(level=0)`删除 level0 上的索引。删除列索引使用`df.droplevel(level=0, axis=1)`。

# tqdm 实践

```python
# tqdm的基础使用方式，1.手动导入普通版本或者notebook版本 2.导入自动版本
from tqdm import tqdm, trange

# manually import, with style in jupyter notebook
from tqdm.notebook import tqdm

# tqdm auto, choose the suitable one
from tqdm.auto import tqdm
```

```python
tqdm( iterable obj, desc='describe', total=length, disable=False )

range(...) => trange(...)
```

```python
# tqdm和pandas的结合使用
tqdm.pandas()  # 启动tqdm对pandas的监测
df.progress_apply(func, axis)  # 使用带进度条功能的apply方法

# 当使用while循环时，手动定制进度条
pbar = tqdm(total=100)
while exp:
    # 显示数据在进度条头部
    pbar.set_description('Processing %s' % data)
    pbar.update(1)
    pass
pbar.close()
```

# Numpy

- ## 随机模块 random

  - `random.rand`: 均匀分布抽样
  - `random.randint`: 参数为`(low, high, length)`
  - `random.randn`: 均值为 0，方差为 1 的正态分布抽样

- ## 向量扁平化

  - `ndarray.ravel()`
  - `ndarray.flatten()`

- ## 用于计算向量距离的指标

  - Euclidean Distance(L2)

  ```python
  # Euclidean Distance
  L2 = [(vec1[i] - vec2[i])**2 for i in range(len(vec1))]
  L2 = np.sqrt(np.array(L2).sum())

  np.linalg.norm((vec1 - vec2), ord=2)
  ```

  - Manhattan Distance(L1)

  ```python
  # Manhattan Distance
  L1 = [(vec1[i] - vec2[i]) for i in range(len(vec1))]
  L1 = np.abs(L1).sum()

  np.linalg.norm((vec1 - vec2), ord=1)
  ```

  - Dot Product

  ```python
  # Dot Product
  np.dot(vec1, vec2)
  ```

  - Cosine Distance

  ```python
  # Cosine Distance
  cosine = 1 - (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
  ```

# Visualize (matplotlib, seaborn, matplotx)

- ## 子图

  - ```python
    fig = plt.figure(..figsize=(10, 10))
    ax1 = fig.add_subplot(2, 2, 1)  # 2行2列第1个子图
    ax2 = fig.add_subplot(2, 2, 2)
    ax1.plot(...)
    ax1.scatter(...)
    ```
  - ```python
    fig, axes = plt.subplots(2, 3) # axes是二维数组，2行3列
    axes[0, 0].plot(...)
    ```

- ## 美化插件 [matplotx](https://github.com/nschloe/matplotx)

  常用图表：

  - Line chart
    ![示例图](https://camo.githubusercontent.com/0c4926e2cd05649898badbeb36a4ae3a3db3bc424d119460ae4c596abaa97287/68747470733a2f2f6e7363686c6f652e6769746875622e696f2f6d6174706c6f74782f6578312d64756674652e737667)

  ```python
  import matplotlib.pyplot as plt
  import matplotx
  import numpy as np

  # create data
  rng = np.random.default_rng(0)
  offsets = [1.0, 1.50, 1.60]
  labels = ["no balancing", "CRV-27", "CRV-27*"]
  x0 = np.linspace(0.0, 3.0, 100)
  y = [offset * x0 / (x0 + 1) + 0.1 * rng.random(len(x0)) for offset in offsets]

  # plot
  with plt.style.context(matplotx.styles.dufte):
      for yy, label in zip(y, labels):
          plt.plot(x0, yy, label=label)
      plt.xlabel("distance [m]")
      matplotx.ylabel_top("voltage [V]")  # move ylabel to the top, rotate
      matplotx.line_labels()  # line labels to the right
      plt.show()
  ```

  - Bar chart
    ![示例图](https://camo.githubusercontent.com/d47f916f07fcca3b9b5b4990cb208cd744f382c8d0012afe7e76a1c300046cc1/68747470733a2f2f6e7363686c6f652e6769746875622e696f2f6d6174706c6f74782f626172732d6475667465322e737667)

  ```python
  import matplotlib.pyplot as plt
  import matplotx

  labels = ["Australia", "Brazil", "China", "Germany", "Mexico", "United\nStates"]
  vals = [21.65, 24.5, 6.95, 8.40, 21.00, 8.55]
  xpos = range(len(vals))

  with plt.style.context(matplotx.styles.dufte_bar):
      plt.bar(xpos, vals)
      plt.xticks(xpos, labels)
      matplotx.show_bar_values("{:.2f}")
      plt.title("average temperature [°C]")
      plt.show()
  ```

# 图像处理

- ## 一张图像的基本数据格式为`uint8`；当使用`opencv`的`cv2.imread()`方法读取后，形状为(height, width, channel:BGR)

- ## 常用的基础方法

  1. Convert to Grayscale (`cv2.cvtColor()`)
  2. Blur (`cv2.GaussianBlur()`)
  3. Edge Cascade (`cv2.Canny()`)
  4. Dilation (`cv2.dilate()`)
  5. Erosion (`cv2.erode()`)
  6. Resize (`cv2.resize()`)
  7. Crop (`img[y:y+h, x:x+w]`)

- ## 平移和旋转 `cv2.warpAffine(src, M, dsize)`

  - 平移

    ```python
    def translate(img, x, y):
     transMat = np.float32([[1, 0, x],[0, 1, y]])
     dimensions = (img.shape[1], img.shape[0])
     return cv2.warpAffine(img, transMat, dimensions)
    ```

    -x → 往左移動 ; x → 往右移動
    -y → 往上移動 ; y → 往下移動

  - 旋转

    ```python
    def rotate(img, angle, rotPoint=None):
       (height,width) = img.shape[:2]
      if rotPoint is None:
       rotPoint = (width // 2, height // 2)

       rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)
       dimensions = (width,height)
      return cv2.warpAffine(img, rotMat, dimensions)
    ```

- ## 平滑和模糊

  1. 均值模糊：
     `cv2.blur(img, (5, 5))`
     均值模糊是一种最简单的平滑滤波方法，它的原理是将图像中每个像素周围的邻域取平均值，从而实现平滑效果。均值模糊对于去除轻微的噪声效果不错，但是对于比较严重的噪声效果并不好，会使图像失去细节。此外，由于均值模糊是基于像素的简单平均值，因此它对于图像中的边缘和纹理等细节部分的保护能力比较弱。

  2. 高斯模糊：
     `cv2.GaussianBlur(img, (5, 5), 0)`
     高斯模糊是一种比较常用的平滑滤波方法，它的原理是利用高斯核对图像进行加权平均，从而实现平滑效果。相比于均值模糊，高斯模糊的效果更加柔和，可以更好地保留图像的细节。同时，高斯模糊还可以通过调整高斯核的参数来控制平滑程度，可以根据具体情况进行调整。

  3. 中值模糊：
     `cv2.medianBlur(img, 5)`
     中值模糊是一种基于排序的平滑滤波方法，它的原理是对图像中每个像素周围的邻域进行排序，然后将中间的值作为该像素的值，从而实现平滑效果。中值模糊对于去除椒盐噪声和斑点噪声效果很好，同时也可以保留图像的边缘和细节。但是，对于比较连续的噪声或纹理等复杂的情况，中值模糊的效果就不太好了。

  4. 双边模糊：
     `cv2.bilateralFilter(img, 9, 75, 75)`
     第三个参数 sigmaColor 是像素值高斯函数的标准差。它控制着像素值之间差异的权重，如果该值越大，则滤波器会考虑更远处像素值之间的差异，使得更多的图像细节能够被保留。但同时，它也会增加噪声的影响，使得输出图像过于保真，看起来过于“清晰”。

  第四个参数 sigmaSpace 是空间高斯函数的标准差。它控制着像素位置之间的差异的权重，如果该值越大，则滤波器会考虑更远处像素之间的空间距离，使得更多的图像边缘能够被保留。与 sigmaColor 不同的是，增加 sigmaSpace 不会产生太多的噪声，因为它只考虑像素之间的空间距离，而不考虑像素值之间的差异。

  对于每个像素，它会考虑在其邻域内的像素距离和像素值之间的差异，然后将权重分配给邻域中的像素，最后计算出该像素的模糊值。它在模糊图像的同时，尽可能地保留图像的边缘和细节。在某些情况下，这种方法可以提供更好的图像效果。但是，双边模糊的计算量比较大，处理速度比较慢。

  一般来说，如果需要**去除较弱**的噪声，可以选择均值模糊；如果需要**保留图像的细节**，可以选择高斯模糊；如果需要**去除比较严重**的噪声，可以选择中值模糊；如果需要**保留图像的边缘和细节**，可以选择双边模糊。

- ## Bitwise Operations
  - `cv2.bitwise_and(img1, img2)`: 两个图像的交集
  - `cv2.bitwise_or(img1, img2)`: 两个图像的并集
  - `cv2.bitwise_xor(img1, img2)`: 两个图像的异或，即非相交的区域
  - `cv2.bitwise_not(img)`: 反转图像的每个像素，比如图像反色
