# PART1

## 数字特征

1. 对于数字特征的缩放和 rank
   - 树形模型不依赖于此类预处理
   - 非树形模型高度需要此类处理
2. 常用的特征处理
   - 标准化。`MaxMinScaler`或者`StandardScaler`
   - 离群值。设置边界，剔除掉离群值，减少对模型决策边界的影响
   - 取对数或平方根。
3. 特征生成  
   依靠创造力和对数据的理解，可以生成更多的数据。例如房价问题上，已知房子总价和面积，可以得到每平米单价等。

## 类别特征与顺序特征

1. 顺序性特征中的数值是以某种有意义的方式进行排序的
2. 标签编码将类别映射为数字
3. 频率编码将类别映射到它们的频率上
4. 标签编码和频率编码经常被用于基于树的的模型
5. 独热编码通常用于非树状模型
6. 类别特征的相互作用可以帮助线性模型和 KNN

## 时间特征和位置特征

1. 日期时间
   - 周期性
   - 行相关或行不相关的事件发生以来的时间
   - 日期之间的差异
2. 坐标
   - 来自训练/测试数据或其他数据的有趣的地方
   - 集群的中心
   - 汇总的统计数据

## 缺失值

1. 填充 NaN 的方法的视情况而定
2. 处理缺失值的通常方法是用-999、平均数或中位数取代它们
3. 缺失的数值可能已经被处理或替换为特殊值
4. 二进制特征 "isnull "可能是有益的
5. 一般来说，避免在特征生成前有 NaN 值
6. XGBoost 可以处理 NaN

# PART2

## EDA

1. 建立数据的直觉
   - 获得领域知识。它有助于更深入地了解问题
   - 检查数据是否是直观的。并与领域知识一致
   - 理解数据是如何产生的。因为这对建立一个适当的验证是至关重要的
   - 常用的函数：`df.info(), df.dtypes, x.value_counts(), x.isnull(), df.describe()`
   - 关于`df.nunique()`和`df.unique()`: 前者在计算时会排除掉 NaN，后者则会将 NaN 视为一个独一无二值
2. 可视化
   探索特征的关系
   - 成对探索：绘制相关性图表：`df.corr() -> plt.matshow()`
   - 成组探索：绘制每列的均值/方差等：`df.mean().plot(style='.')`

## 验证集策略

1. HoldOut
2. KFold
3. LOO (leave-one-out)
4. 对于数据量小的和样本分类不均衡的数据集，采用`stratification`划分数据集，它将使得被划分后的每块中样本类别比例等于原始数据集

## 验证时出现的一些问题

1. 如果我们在验证阶段有很大的分数分散性，我们应该进行广泛的验证
   - 从不同的 Fold 分割中获得平均分数
   - 在一个分割上调整模型，在另一个分割上评估分数
2. 如果提交的分数与本地验证的分数不一致，我们应该
   - 检查我们在公共 LB 中是否有太少的数据
   - 检查我们是否过度拟合
   - 检查我们是否选择了正确的拆分策略
   - 检查训练/测试是否有不同的分布情况
3. 预计 LB 洗牌的原因是：随机性、数据量小、不同的公有/私有分布

# PART3

## 指标

选择竞赛指定的指标来度量模型，并在此基础上进行优化。

## 回归模型指标

- **MSE**, **RMSE**, **R-Squared**
  从优化角度上来说几乎一致
- **MAE**
  更具有鲁棒性，对离群值不那么敏感
- **(R)MSPE**, **MAPE** (percentage)
  加了权重计算的 MSE 和 MAE
- **(R)MSLE**
  对数空间上的 MSE

## 分类模型指标

**Accuracy, Logloss, AUR(ROC), (Quadratic weighted)Kappa**

## 平均数编码

Mean Encoding，正则化，衍生的情况以及优势和劣势。作用于指定的数据集或是特定情况，在此前提下，会产生显著效果

# PART4

## pipeline 指导

关于特征工程，不同的问题可以做以下方向的参考：

- 图像分类：缩放、移位、旋转、CNNs
- 声音分类：傅立叶，Mfcc，specgrams，缩放
- 文本分类：Tf-idf、svd、词干 stemming、拼写检查、删除停顿词、x-grams
- 时间序列：滞后 lags、加权平均、指数平滑
- 分类：目标编码、频率、单热、序数、标签编码
- 数值计算： 缩放、分档 binning、导数、离群点去除、降维
- 相互作用：乘法，除法，分组特征，concatenation
- 推荐：交易历史上的特征，项目流行度，购买频率

## t-SNE

- t-SNE 是一个有效的可视化工具
- 它也能被用作一项特征(feature)
- 当解释它的结果时要小心
- 强烈依赖于超参数，可以试试不同的 perplexities

## 模型融合

- 平均法：单纯的求均值聚合模型的输出；带权重计算的平均输出；根据特点情况的平均
- bagging：将**同一个**模型的多个轻微变化平均组合在一起，常见的 bagging 参数有：

  - 改变随机种子
  - 随机采样样本
  - 特征随机采样
  - 模型自身的不同参数
  - 模型的数量

- boosting：一种加权平均模型的形式，其中每个模型是通过考虑过去的模型性能而依次建立的。boosting 有两种类型：基于权重和基于残差。

  - 基于权重：对于上一个模型预测做的比较差的样本给予更大的权重，输入到下一个模型中
  - 基于残差：对于上一个模型预测计算误差，但是不求绝对值(带正负号)，并与预测值相加，然后作为下一个模型的输入

- stacking：将多个模型的输出作为特征项，输入到新的模型中，模型会在学习中给不同模型的输出分配权重。通常的流程是： 1. 将数据集分为训练集和验证集 2. 在训练集上训练几个基础模型 3. 使用训练好的基础对验证集进行预测 4. 使用上一步的预测结果作为新的特征项，训练一个新的模型

  toy example:

  ```
  ## 原始数据为 train_dataset->(X, y), test_dataset->(test)
  train_dataset, test_dataset
  train, valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

  model1 = LogisticRegression()
  model2 = RandomForestRegressor()

  model1.fit(train, y_train)
  model2.fit(train, y_train)

  pred1 = model1.predict(valid)
  pred2 = model2.predict(valid)

  test_pred1 = model1.predict(test)
  test_pred2 = model2.predict(test)

  stacked_pred = np.column_stack((pred1, pred2))
  test_stacked_pred = np.column_stack((test_pred1, test_pred2))

  meta_model = LinearRegression()
  meta_model.fit(stacked_pred, y_valid)

  final_pred = meta_model.predict(test_stacked_pred)
  ```

  注意事项：

  - 时序数据需要特别注意
  - 多样性可以带来更好的结果。多样性来自于：不同的模型和不同的模型训练数据
  - 元模型不需要过于复杂
