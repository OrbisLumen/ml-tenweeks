# Logistic Regression

> Logistic Regression 虽然名字中包含 Regression，但主要用于分类问题。本节介绍二分类逻辑回归、交叉熵、生成式与判别式模型、多分类 Softmax，以及逻辑回归的限制。

# 逻辑回归

在生成式分类器中，当两个类别共享协方差矩阵时，可以得到：

$$
P(C_1\mid\mathbf{x})
=
\sigma(\mathbf{w}^{T}\mathbf{x}+b)
$$

生成式模型先估计：

$$
\boldsymbol{\mu}_1,
\boldsymbol{\mu}_2,
\boldsymbol{\Sigma},
P(C_1),
P(C_2)
$$

再由这些参数计算 $\mathbf{w}$ 和 $b$。

逻辑回归则直接从训练数据中学习：

$$
\mathbf{w},b
$$

这是一种判别式模型。

# Step 1：选择函数集合

首先计算输入特征的线性组合：

$$
z
=
\mathbf{w}^{T}\mathbf{x}+b
=
\sum_iw_ix_i+b
$$

然后通过 Sigmoid 函数将其转换为 $0$ 到 $1$ 之间的数：

$$
\sigma(z)
=
\frac{1}{1+\exp(-z)}
$$

逻辑回归模型为：

$$
f_{\mathbf{w},b}(\mathbf{x})
=
P_{\mathbf{w},b}(C_1\mid\mathbf{x})
=
\sigma(\mathbf{w}^{T}\mathbf{x}+b)
$$

函数集合包含所有不同的 $\mathbf{w}$ 和 $b$。

## 分类规则

如果：

$$
P_{\mathbf{w},b}(C_1\mid\mathbf{x})\geq 0.5
$$

则输出 $C_1$，否则输出 $C_2$。

因为 Sigmoid 函数单调递增，并且：

$$
\sigma(0)=0.5
$$

所以分类规则也可以写成：

$$
f(\mathbf{x})
=
\begin{cases}
C_1,&\mathbf{w}^{T}\mathbf{x}+b\geq0\\
C_2,&\mathbf{w}^{T}\mathbf{x}+b<0
\end{cases}
$$

分类边界为：

$$
\boxed{
\mathbf{w}^{T}\mathbf{x}+b=0
}
$$

因此，一个逻辑回归模型只能产生线性分类边界。

# 逻辑回归与线性回归

线性回归模型为：

$$
f_{\mathbf{w},b}(\mathbf{x})
=
\mathbf{w}^{T}\mathbf{x}+b
$$

输出范围为：

$$
(-\infty,+\infty)
$$

逻辑回归模型为：

$$
f_{\mathbf{w},b}(\mathbf{x})
=
\sigma(\mathbf{w}^{T}\mathbf{x}+b)
$$

输出范围为：

$$
(0,1)
$$

因此，可以将逻辑回归的输出解释为类别概率。

# Step 2：定义函数的好坏

假设训练数据为：

$$
\left\{
\left(\mathbf{x}^{(1)},y^{(1)}\right),
\left(\mathbf{x}^{(2)},y^{(2)}\right),
\cdots,
\left(\mathbf{x}^{(N)},y^{(N)}\right)
\right\}
$$

标签定义为：

$$
y^{(n)}
=
\begin{cases}
1,&\mathbf{x}^{(n)}\in C_1\\
0,&\mathbf{x}^{(n)}\in C_2
\end{cases}
$$

模型输出：

$$
f_{\mathbf{w},b}(\mathbf{x}^{(n)})
=
P(C_1\mid\mathbf{x}^{(n)})
$$

因此：

$$
P(C_2\mid\mathbf{x}^{(n)})
=
1-f_{\mathbf{w},b}(\mathbf{x}^{(n)})
$$

## 似然函数

对于一个属于 $C_1$ 的样本，模型产生正确标签的概率为：

$$
f_{\mathbf{w},b}(\mathbf{x})
$$

对于一个属于 $C_2$ 的样本，模型产生正确标签的概率为：

$$
1-f_{\mathbf{w},b}(\mathbf{x})
$$

所有训练数据同时出现的概率为：

$$
L(\mathbf{w},b)
=
\prod_{n=1}^{N}
f_{\mathbf{w},b}(\mathbf{x}^{(n)})^{y^{(n)}}
\left(
1-f_{\mathbf{w},b}(\mathbf{x}^{(n)})
\right)^{1-y^{(n)}}
$$

目标是寻找使训练数据出现概率最大的参数：

$$
(\mathbf{w}^*,b^*)
=
\arg\max_{\mathbf{w},b}
L(\mathbf{w},b)
$$

# 从最大似然到交叉熵

由于连乘不方便计算，对似然函数取对数：

$$
\ln L(\mathbf{w},b)
=
\sum_{n=1}^{N}
\left[
y^{(n)}
\ln f_{\mathbf{w},b}(\mathbf{x}^{(n)})
+
(1-y^{(n)})
\ln
\left(
1-f_{\mathbf{w},b}(\mathbf{x}^{(n)})
\right)
\right]
$$

最大化对数似然等价于最小化负对数似然：

$$
\mathcal{L}(\mathbf{w},b)
=
-\sum_{n=1}^{N}
\left[
y^{(n)}
\ln f_{\mathbf{w},b}(\mathbf{x}^{(n)})
+
(1-y^{(n)})
\ln
\left(
1-f_{\mathbf{w},b}(\mathbf{x}^{(n)})
\right)
\right]
$$

单个样本的损失为：

$$
C(f(\mathbf{x}),y)
=
-\left[
y\ln f(\mathbf{x})
+
(1-y)\ln(1-f(\mathbf{x}))
\right]
$$

这就是二元交叉熵：

$$
\boxed{
\text{Binary Cross Entropy}
}
$$

## 标签为 1

当：

$$
y=1
$$

损失变成：

$$
C=-\ln f(\mathbf{x})
$$

为了减小损失，模型需要让：

$$
f(\mathbf{x})\rightarrow1
$$

## 标签为 0

当：

$$
y=0
$$

损失变成：

$$
C=-\ln(1-f(\mathbf{x}))
$$

为了减小损失，模型需要让：

$$
f(\mathbf{x})\rightarrow0
$$

因此，交叉熵会推动模型将正确类别的预测概率提高。

# 交叉熵的含义

真实标签 $y$ 可以看作一个伯努利概率分布：

$$
\hat{\mathbf{y}}
=
\begin{cases}
(1,0),&y=1\\
(0,1),&y=0
\end{cases}
$$

模型预测的概率分布为：

$$
\mathbf{p}
=
\left(
f(\mathbf{x}),
1-f(\mathbf{x})
\right)
$$

交叉熵衡量真实分布与预测分布之间的差异。

交叉熵越小，说明模型预测的概率分布越接近真实标签。

# Step 3：寻找最好的函数

逻辑回归的损失函数为：

$$
\mathcal{L}(\mathbf{w},b)
=
-\sum_{n=1}^{N}
\left[
y^{(n)}
\ln f_{\mathbf{w},b}(\mathbf{x}^{(n)})
+
(1-y^{(n)})
\ln
\left(
1-f_{\mathbf{w},b}(\mathbf{x}^{(n)})
\right)
\right]
$$

其中：

$$
f_{\mathbf{w},b}(\mathbf{x})
=
\sigma(z)
$$

$$
z=\mathbf{w}^{T}\mathbf{x}+b
$$

Sigmoid 的导数为：

$$
\sigma'(z)
=
\sigma(z)(1-\sigma(z))
$$

对权重 $w_i$ 求偏导，最终得到：

$$
\frac{\partial\mathcal{L}}{\partial w_i}
=
\sum_{n=1}^{N}
\left(
f_{\mathbf{w},b}(\mathbf{x}^{(n)})
-y^{(n)}
\right)x_i^{(n)}
$$

对偏置求偏导：

$$
\frac{\partial\mathcal{L}}{\partial b}
=
\sum_{n=1}^{N}
\left(
f_{\mathbf{w},b}(\mathbf{x}^{(n)})
-y^{(n)}
\right)
$$

梯度下降更新为：

$$
w_i
\leftarrow
w_i
-
\eta
\sum_{n=1}^{N}
\left(
f_{\mathbf{w},b}(\mathbf{x}^{(n)})
-y^{(n)}
\right)x_i^{(n)}
$$

$$
b
\leftarrow
b
-
\eta
\sum_{n=1}^{N}
\left(
f_{\mathbf{w},b}(\mathbf{x}^{(n)})
-y^{(n)}
\right)
$$

也可以写成：

$$
w_i
\leftarrow
w_i
+
\eta
\sum_{n=1}^{N}
\left(
y^{(n)}
-f_{\mathbf{w},b}(\mathbf{x}^{(n)})
\right)x_i^{(n)}
$$

其中：

$$
y^{(n)}
-f_{\mathbf{w},b}(\mathbf{x}^{(n)})
$$

表示真实标签与预测概率之间的误差。

# 与线性回归梯度的比较

线性回归使用平方误差：

$$
\mathcal{L}_{\mathrm{linear}}
=
\frac{1}{2}
\sum_n
\left(
f(\mathbf{x}^{(n)})-y^{(n)}
\right)^2
$$

它对权重的梯度为：

$$
\frac{\partial\mathcal{L}_{\mathrm{linear}}}{\partial w_i}
=
\sum_n
\left(
f(\mathbf{x}^{(n)})-y^{(n)}
\right)x_i^{(n)}
$$

逻辑回归配合交叉熵得到的梯度也是：

$$
\frac{\partial\mathcal{L}_{\mathrm{logistic}}}{\partial w_i}
=
\sum_n
\left(
f(\mathbf{x}^{(n)})-y^{(n)}
\right)x_i^{(n)}
$$

两者更新形式相同，但模型不同：

线性回归：

$$
f(\mathbf{x})
=
\mathbf{w}^{T}\mathbf{x}+b
$$

逻辑回归：

$$
f(\mathbf{x})
=
\sigma(\mathbf{w}^{T}\mathbf{x}+b)
$$

# 为什么不用平方误差

假设逻辑回归使用平方误差：

$$
\mathcal{L}
=
\frac{1}{2}
\sum_n
\left(
f(\mathbf{x}^{(n)})-y^{(n)}
\right)^2
$$

单个样本对权重的梯度为：

$$
\frac{\partial\mathcal{L}}{\partial w_i}
=
\left(
f(\mathbf{x})-y
\right)
f(\mathbf{x})
\left(
1-f(\mathbf{x})
\right)x_i
$$

梯度中多出了：

$$
f(\mathbf{x})(1-f(\mathbf{x}))
$$

当模型输出接近 $0$ 或 $1$ 时：

$$
f(\mathbf{x})(1-f(\mathbf{x}))
\approx0
$$

即使模型预测完全错误，梯度也可能非常小。

例如真实标签为：

$$
y=1
$$

但模型预测：

$$
f(\mathbf{x})\approx0
$$

此时模型错得非常严重，但平方误差的梯度仍可能接近 $0$，导致参数几乎不更新。

因此，平方误差配合 Sigmoid 容易产生平坦区域，使优化变得困难。

交叉熵求导后不会保留这个额外的 Sigmoid 导数项：

$$
\frac{\partial\mathcal{L}}{\partial w_i}
=
(f(\mathbf{x})-y)x_i
$$

预测越错误，通常会产生越明显的更新。

所以分类问题中一般使用：

$$
\boxed{
\text{Sigmoid}+\text{Binary Cross Entropy}
}
$$

而不是：

$$
\text{Sigmoid}+\text{Square Error}
$$

# 生成式模型与判别式模型

## 生成式模型

生成式模型先估计：

$$
P(\mathbf{x}\mid C_i)
$$

和：

$$
P(C_i)
$$

再通过贝叶斯公式得到：

$$
P(C_i\mid\mathbf{x})
$$

在共享高斯协方差的情况下，需要估计：

$$
\boldsymbol{\mu}_1,
\boldsymbol{\mu}_2,
\boldsymbol{\Sigma},
P(C_1),
P(C_2)
$$

然后计算：

$$
\mathbf{w}
=
\boldsymbol{\Sigma}^{-1}
(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)
$$

以及：

$$
b
=
-\frac{1}{2}
\boldsymbol{\mu}_1^T
\boldsymbol{\Sigma}^{-1}
\boldsymbol{\mu}_1
+
\frac{1}{2}
\boldsymbol{\mu}_2^T
\boldsymbol{\Sigma}^{-1}
\boldsymbol{\mu}_2
+
\ln\frac{P(C_1)}{P(C_2)}
$$

## 判别式模型

逻辑回归不建立完整的数据生成过程，而是直接学习：

$$
P(C_1\mid\mathbf{x})
=
\sigma(\mathbf{w}^{T}\mathbf{x}+b)
$$

也就是直接寻找：

$$
\mathbf{w},b
$$

## 两者的关系

在共享高斯协方差的假设下，生成式模型和逻辑回归具有相同的函数形式：

$$
P(C_1\mid\mathbf{x})
=
\sigma(\mathbf{w}^{T}\mathbf{x}+b)
$$

因此，它们的函数集合相同。

但是它们选择参数的方法不同：

- 生成式模型通过估计数据分布间接得到 $\mathbf{w},b$
- 判别式模型直接优化分类目标得到 $\mathbf{w},b$

所以，即使训练数据相同，最终得到的参数也不一定相同。

视频中的宝可梦分类实验结果约为：

- 生成式模型：$73\%$
- 判别式逻辑回归：$79\%$

这并不代表判别式模型永远优于生成式模型，而是说明两者使用了不同的假设和参数估计方式。

# 生成式模型的优点

生成式模型对数据分布作出了额外假设。

这些假设可能带来以下优点：

- 训练数据较少时，可以利用分布假设降低估计难度
- 对训练数据中的噪声可能更加稳健
- 先验概率和类别条件概率可以从不同来源估计
- 可以描述数据是如何产生的
- 可以处理部分缺失信息或结合领域知识

但如果分布假设与真实数据差异很大，也可能限制模型效果。

# 多分类

对于 $K$ 个类别，每个类别都有一组参数：

$$
\mathbf{w}_1,b_1,
\mathbf{w}_2,b_2,
\cdots,
\mathbf{w}_K,b_K
$$

首先计算每个类别的分数：

$$
z_i
=
\mathbf{w}_i^T\mathbf{x}+b_i
$$

将所有分数写成向量：

$$
\mathbf{z}
=
\begin{bmatrix}
z_1\\
z_2\\
\vdots\\
z_K
\end{bmatrix}
$$

然后使用 Softmax 将分数转换为概率：

$$
p_i
=
\frac{\exp(z_i)}
{\sum_{j=1}^{K}\exp(z_j)}
$$

其中：

$$
p_i=P(C_i\mid\mathbf{x})
$$

Softmax 输出满足：

$$
0<p_i<1
$$

以及：

$$
\sum_{i=1}^{K}p_i=1
$$

因此，Softmax 的输出可以解释为各类别的概率分布。

预测类别为：

$$
\hat C
=
\arg\max_i p_i
$$

# One-Hot 标签

多分类任务通常使用 One-Hot 向量表示标签。

三个类别的标签可以表示为：

$$
C_1
\rightarrow
\begin{bmatrix}
1\\0\\0
\end{bmatrix}
$$

$$
C_2
\rightarrow
\begin{bmatrix}
0\\1\\0
\end{bmatrix}
$$

$$
C_3
\rightarrow
\begin{bmatrix}
0\\0\\1
\end{bmatrix}
$$

设真实标签为：

$$
\mathbf{y}
=
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_K
\end{bmatrix}
$$

Softmax 预测为：

$$
\mathbf{p}
=
\begin{bmatrix}
p_1\\
p_2\\
\vdots\\
p_K
\end{bmatrix}
$$

多分类交叉熵为：

$$
\mathcal{L}
=
-\sum_{i=1}^{K}
y_i\ln p_i
$$

由于 One-Hot 标签中只有正确类别对应的元素为 $1$，因此：

$$
\mathcal{L}
=
-\ln p_{\mathrm{correct}}
$$

最小化交叉熵就是提高正确类别的预测概率。

常见组合为：

$$
\boxed{
\text{Binary Classification}
\rightarrow
\text{Sigmoid}+\text{Binary Cross Entropy}
}
$$

$$
\boxed{
\text{Multi-class Classification}
\rightarrow
\text{Softmax}+\text{Cross Entropy}
}
$$

# 逻辑回归的限制

单个逻辑回归模型的分类边界为：

$$
\mathbf{w}^{T}\mathbf{x}+b=0
$$

所以它只能解决线性可分的问题。

例如 XOR 问题：

| $x_1$ | $x_2$ | 类别 |
| ---: | ---: | --- |
| 0 | 0 | $C_2$ |
| 0 | 1 | $C_1$ |
| 1 | 0 | $C_1$ |
| 1 | 1 | $C_2$ |

属于同一类别的点分别位于对角线位置，无法用一条直线将两个类别完全分开。

因此，单个逻辑回归模型无法解决 XOR 问题。

# 特征变换

一种解决方法是先将原始特征转换为新的特征：

$$
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
\longrightarrow
\begin{bmatrix}
x_1'\\
x_2'
\end{bmatrix}
$$

例如，可以定义：

$$
x_1'
=
\left\|
\mathbf{x}
-
\begin{bmatrix}
0\\0
\end{bmatrix}
\right\|
$$

$$
x_2'
=
\left\|
\mathbf{x}
-
\begin{bmatrix}
1\\1
\end{bmatrix}
\right\|
$$

也就是将新特征定义为输入点到两个参考点的距离。

原本在特征空间中线性不可分的数据，经过适当变换后可能变得线性可分。

但是人工寻找合适的特征变换通常并不容易。

# 串联逻辑回归

可以使用多个逻辑回归模型自动完成特征变换。

第一层逻辑回归：

$$
x_1'
=
\sigma(\mathbf{w}_1^T\mathbf{x}+b_1)
$$

$$
x_2'
=
\sigma(\mathbf{w}_2^T\mathbf{x}+b_2)
$$

将第一层的输出作为新的特征：

$$
\mathbf{x}'
=
\begin{bmatrix}
x_1'\\
x_2'
\end{bmatrix}
$$

再输入下一层逻辑回归：

$$
y
=
\sigma(\mathbf{w}^T\mathbf{x}'+b)
$$

第一层负责将原始特征转换到新的表示空间，第二层在新的空间中进行分类。

不断串联这种结构：

$$
\boxed{
\text{Linear Combination}
+
\text{Activation Function}
}
$$

就形成了神经网络。

其中，每个计算单元可以看作一个神经元：

$$
a
=
\sigma
\left(
\sum_iw_ix_i+b
\right)
$$

多个神经元组成一层，多层神经元串联形成深度神经网络。

# 总结

1. 逻辑回归用于分类问题。
2. 模型先计算线性组合：

$$
z=\mathbf{w}^{T}\mathbf{x}+b
$$

3. 再通过 Sigmoid 输出类别概率：

$$
P(C_1\mid\mathbf{x})=\sigma(z)
$$

4. 二分类标签通常表示为 $0$ 和 $1$。
5. 最大化训练数据的似然等价于最小化二元交叉熵。
6. 逻辑回归的梯度为：

$$
\frac{\partial\mathcal{L}}{\partial w_i}
=
\sum_n
(f(\mathbf{x}^{(n)})-y^{(n)})x_i^{(n)}
$$

7. Sigmoid 配合平方误差容易产生梯度接近 $0$ 的平坦区域。
8. 生成式模型学习 $P(\mathbf{x}\mid C)$ 和 $P(C)$。
9. 判别式模型直接学习 $P(C\mid\mathbf{x})$。
10. 多分类使用 Softmax 将各类别分数转换为概率。
11. Softmax 通常配合 One-Hot 标签和交叉熵。
12. 单个逻辑回归只能产生线性分类边界。
13. 逻辑回归无法直接解决 XOR 等线性不可分问题。
14. 特征变换可以将线性不可分问题转换为线性可分问题。
15. 将多个逻辑回归单元层层串联，就得到神经网络的基本结构。