## 权重衰减（WeightDecay）：

- 通过添加惩罚项 L2 正则使得模型参数不会过大，从而控制模型复杂度
- 正则项权重是控制模型复杂度的超参数

## 丢弃法（Dropout）：

- 将一些输出项随机置 0 来控制模型复杂度
- 常作用在 MLP 的隐藏层输出上
- 丢弃概率是控制模型复杂度的超参数

## 数值稳定性：

- 合理的权重初始值和激活函数的选取可以提高数值稳定性

## 微调（fine tuning）：

在下游任务的数据集上进行继续训练，区别于 linear prob，fine-tune 会更新整个模型参数，而 linear prob 只更新底层(最后线性层的参数)

- 使用已经训练过了的模型，但要使用更强的正则化
- 使用更小的学习率
- 使用更少的数据迭代次数

## 批量归一化（BatchNorm）：

特点：

- 一般用于深层网络，浅层效果不好
- 固定小批量中的均值和方差，然后学习出适合的偏移和缩放
- 可以加快收敛，一般不改变模型精度，但是一些情况下可能导致精度下降

为什么需要 BN：

- 数据预处理的方式会对最终结果产生巨大影响。
- 对于通常的深度网络，中间层的输入分布会发生变化，这会导致训练过程中的梯度消失或者爆炸。
- 更深层的网络很容易过拟合，需要正则化。

## 层归一化（LayerNorm）：

- 对每条样本的所有特征做归一化

## BN 和 LN 的对比：

- BatchNorm： 对一个 batch-size 样本内的每个特征做归一化
- LayerNorm： 针对每条样本，对每条样本的所有特征做归一化  
  对于 2D 输入来说，BatchNorm 就是对每列特征做归一化，LayerNorm 就是对每行做归一化。

差异：

- 如果你的特征依赖不同样本的统计参数，那 BatchNorm 更有效， 因为它不考虑不同特征之间的大小关系，但是保留不同样本间的大小关系
- NLP 领域适合用 LayerNorm， CV 适合 BatchNorm，
- 对于 NLP 来说，它不考虑不同样本间的大小关系，保留样本内不同特征之间的大小关系

## 关于正则化、归一化、标准化的区别和联系

**标准化**：Standardization, **归一化**：Normalization, **正则化**：Regularization  
(在 pytorch 或者 DL 中，Normalization 是指标准化)

- 标准化：减去均值，除以方差：
  $${x_{new} = (x - \mu) / \sigma}$$

- 归一化：将数据压缩到一个区间内，比如[0, 1]、[-1, 1]等。常见方法有两种：

  - Min-Max Normalization:
    $${x_{new} = \frac{x - \min}{\max - \min}}$$
  - Mean Normalization:
    $${x_{new} = \frac{x - x_{mean}}{\max - \min}}$$

- 正则化：对模型参数进行惩罚，使得模型参数不会过大，从而控制模型复杂度。
  - 加入 L2 范数正则项，称为岭回归 _ridge regression_
  - 加入 L1 范数正则项，称为 _Lasso regression_

## 多分类问题下，评价指标的不同计算方法

- **macro**:宏平均（Macro-averaging）
  把每个类别都当成二分类，分别计算出各个类别 对应的 precision，recall, f1 , 然后求所有类别的 precision，recall,f1 的平均值，得到最终的 precision recall f1. 这里假设所有分类都是一样的重要，所以 整体结果受小类别（数量比较少的 target）的影响比较大。

- **micro**:微平均（Micro-averaging）
  把各个类别当成二分类，统计各自的混淆矩阵，然后统计加和 比例 得到一个最终的 混淆矩阵，再计算 precision，recall，f1.

## Softmax

softmax 具有保序性，即 softmax 值和模型的输出值对应的排序不会发生变化。
在 Pytorch 中，`nn.CrossEntropyLoss()`会在内部先计算 softmax 再计算 loss，因而不需要在构建模型时加入 softmax 层。  
在推理时，直接使用`argmax()`操作即可得到对应的分类标号。

## 1\*1 卷积

1\*1 卷积操作相当于一个线性层，通过减少输出通道数，达到降维的效果。其次是为了控制模型的复杂度。

## 卷积计算与池化计算后的特征图维度

- `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None)`:
  $$out = \frac{H/W - kernelsize + padding * 2}{stride}  + 1$$
  当 kernel_size=3, stride=1, padding=1 时，输入和输出尺寸相同。

- `nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)`:
  $$out = \frac{H/W + 2 * padding - dilation * (kernelsize - 1)-1}{stride}  + 1$$
  kernel_size=2, stride=2, padding=0, dilation=1 时，输出尺寸变为输入的一半。如果步长为 1，输出会比输入变小。
  池化层的作用： 1. 实现 invariance(不变性)，这种不变性包括 translation(平移)，rotation(旋转)，scale(尺度) 2. （下采样）保留主要的特征同时减少参数(降维，效果类似 PCA)和计算量，防止过拟合，提高模型泛化能力

## Pytorch 矩阵乘法的函数

- `torch.mm()`: 基本的**二维**矩阵乘法，但只适用二维。
- `torch.bmm()`: **批量**矩阵乘法。
- `torch.matmul()`: 二维矩阵乘法和批量矩阵乘法的统一接口，可以适用于高维。
- `torch.mul()`: 逐元素乘法，可以适用于高维。即哈达玛积，等同于 a \* b。

## RNN 和 Transformer

- RNN 因为每个输出要依赖前一个输出当作输入，无法做到并行计算
- `self-attention`可以做并行，输出结果是同时被计算出来的
- 传统的词向量(word2vec)对于每个词预训练好后就不会改变，然而每个词在实际语境下会有不同的表达含义

## 可视化一个 batch 的 PyTorch 张量类型的图片

```python
def visualize_batch_tensor(t1):
    img = torchvision.utils.make_grid(t1.long()).cpu().numpy()
    plt.imshow(np.transpose(img, (1,2,0)))
    plt.show()
```
