# Pandas

## 关于 DataFrame 的大小问题

-   设置对应的 dtype，可以有效减少 DataFrame 的大小，但是当写入 csv 文件时，dtype 数据将会丢失。
-   `df.to_pickle(df.pickle)`pickle 文件读写比 csv 更快，且文件更小，并且会保存所有的 dtype 信息。
-   `df.to_feather(df.feather)`feather 文件读写比 csv 更快，文件更小。更适合于短期储存，而`parquet` (需要 install)更适合长期储存。
-   size: csv > pickle > feather > parquet
-   speed: csv > parquet > feather > pickle

## concat, merge

-   concat: `ignore_index`参数可以重新生成 index
-   merge: `on='inner/outer'`默认为内连接，outer 为外连接

## 函数形式和非函数形式

-   常见的非函数形式:
    ```python
    df2 = df.copy()
    df2['k'] = v
    ```
-   等价的函数形式
    ```python
    df2 = df.assign(k=v)
    ```

## 分层索引

-   `index/columns .get_level_values(level=0)`获取 level0 上的索引。
-   `droplevel(level=0)`删除 level0 上的索引。删除列索引使用`df.droplevel(level=0, axis=1)`。

# Numpy

## 随机模块 random

-   `random.rand`: 均匀分布抽样
-   `random.randint`: 参数为`(low, high, length)`
-   `random.randn`: 均值为 0，方差为 1 的正态分布抽样

## 向量扁平化

-   `ndarray.ravel()`
-   `ndarray.flatten()`

# Visualize (matplotlib, seaborn)

## 子图

-   ```python
    fig = plt.figure(..figsize=(10, 10))
    ax1 = fig.add_subplot(2, 2, 1)  # 2行2列第1个子图
    ax2 = fig.add_subplot(2, 2, 2)
    ax1.plot(...)
    ax1.scatter(...)
    ```
-   ```python
    fig, axes = plt.subplots(2, 3) # axes是二维数组，2行3列
    axes[0, 0].plot(...)
    ```

# 图像处理

## 一张图像的基本数据格式为`uint8`，当使用`opencv`的`cv2.imread()`方法读取后，形状为(height, width, channel:BGR)

## 常用的基础方法

1. Convert to Grayscale (`cv2.cvtColor()`)
2. Blur (`cv2.GaussianBlur()`)
3. Edge Cascade (`cv2.Canny()`)
4. Dilation (`cv2.dilate()`)
5. Erosion (`cv2.erode()`)
6. Resize (`cv2.resize()`)
7. Crop (`img[y:y+h, x:x+w]`)

## 平移和旋转 `cv2.warpAffine(src, M, dsize)`

-   平移

    ```python
    def translate(img, x, y):
     transMat = np.float32([[1, 0, x],[0, 1, y]])
     dimensions = (img.shape[1], img.shape[0])
     return cv2.warpAffine(img, transMat, dimensions)
    ```

    -x → 往左移動 ; x → 往右移動
    -y → 往上移動 ; y → 往下移動

-   旋转
    ```python
    def rotate(img, angle, rotPoint=None):
       (height,width) = img.shape[:2]
      if rotPoint is None:
       rotPoint = (width // 2, height // 2)

       rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)
       dimensions = (width,height)
      return cv2.warpAffine(img, rotMat, dimensions)
    ```

## 平滑和模糊
1. 均值模糊：
`cv2.blur(img, (5, 5))`
均值模糊是一种最简单的平滑滤波方法，它的原理是将图像中每个像素周围的邻域取平均值，从而实现平滑效果。均值模糊对于去除轻微的噪声效果不错，但是对于比较严重的噪声效果并不好，会使图像失去细节。此外，由于均值模糊是基于像素的简单平均值，因此它对于图像中的边缘和纹理等细节部分的保护能力比较弱。

2. 高斯模糊：
`cv2.GaussianBlur(img, (5, 5), 0)`
高斯模糊是一种比较常用的平滑滤波方法，它的原理是利用高斯核对图像进行加权平均，从而实现平滑效果。相比于均值模糊，高斯模糊的效果更加柔和，可以更好地保留图像的细节。同时，高斯模糊还可以通过调整高斯核的参数来控制平滑程度，可以根据具体情况进行调整。

3. 中值模糊：
`cv2.medianBlur(img, 5)`
中值模糊是一种基于排序的平滑滤波方法，它的原理是对图像中每个像素周围的邻域进行排序，然后将中间的值作为该像素的值，从而实现平滑效果。中值模糊对于去除椒盐噪声和斑点噪声效果很好，同时也可以保留图像的边缘和细节。但是，对于比较连续的噪声或纹理等复杂的情况，中值模糊的效果就不太好了。

一般来说，如果需要**去除较弱**的噪声，可以选择均值模糊；如果需要**保留图像的细节**，可以选择高斯模糊；如果需要**去除比较严重**的噪声，可以选择中值模糊。