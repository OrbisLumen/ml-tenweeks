# Classification：Probabilistic Generative Model

> 通过宝可梦属性分类案例，介绍分类问题、贝叶斯分类器、概率生成模型与高斯分布。


# 分类问题

分类要寻找一个函数：

$$
f(\mathbf{x})=C_i
$$

其中：

- $\mathbf{x}$：输入特征
- $C_i$：预测的类别
- 分类模型的输出是离散类别，而不是连续数值

分类问题的例子：

- 输入收入、年龄、职业等信息，判断是否批准贷款
- 输入症状和病史，判断疾病种类
- 输入手写图片，判断对应字符
- 输入人脸图片，判断人物身份
- 输入宝可梦属性，判断宝可梦种类

本节使用宝可梦分类作为案例，根据宝可梦的属性判断它属于水系还是一般系。

宝可梦的特征可以包括：

$$
\mathbf{x}
=
(x_{\mathrm{total}},x_{\mathrm{hp}},x_{\mathrm{attack}},
x_{\mathrm{defense}},x_{\mathrm{sp.attack}},
x_{\mathrm{sp.defense}},x_{\mathrm{speed}})
$$

# 能否将分类当作回归

## 二分类

可以暂时将两个类别表示成数值：

$$
\begin{aligned}
C_1&\rightarrow 1\\
C_2&\rightarrow -1
\end{aligned}
$$

训练一个线性回归模型：

$$
y=b+\mathbf{w}^{T}\mathbf{x}
$$

预测时：

- 输出接近 $1$，判断为 $C_1$
- 输出接近 $-1$，判断为 $C_2$

但是这种方法存在问题。

假设某个属于 $C_1$ 的样本被模型预测为：

$$
y\gg 1
$$

从分类角度看，这个样本已经被正确分类。

但是平方误差仍然会认为预测值与目标值 $1$ 相差很大，从而惩罚这个“过于正确”的样本，并可能影响分类边界。

## 多分类

假设直接将类别编号为：

$$
\begin{aligned}
C_1&\rightarrow 1\\
C_2&\rightarrow 2\\
C_3&\rightarrow 3
\end{aligned}
$$

这种表示会错误地暗示：

- 类别之间存在大小顺序
- $C_2$ 与 $C_1$、$C_3$ 更接近
- $C_1$ 与 $C_3$ 的差异更大

但类别编号通常只是名称，不代表类别之间的距离。

因此，不能简单地使用普通线性回归处理分类问题。

# 理想的分类模型

对于二分类，可以定义：

$$
f(\mathbf{x})
=
\begin{cases}
C_1,&g(\mathbf{x})>0\\
C_2,&g(\mathbf{x})\leq 0
\end{cases}
$$

损失函数可以直接统计错误分类的数量：

$$
L(f)
=
\sum_{n=1}^{N}
\delta\left(f(\mathbf{x}^{(n)})\neq y^{(n)}\right)
$$

其中：

$$
\delta(\text{condition})
=
\begin{cases}
1,&\text{condition 成立}\\
0,&\text{condition 不成立}
\end{cases}
$$

但是分类错误数量是离散的，无法直接使用普通梯度下降进行优化。

其他分类方法包括：

- Perceptron
- Support Vector Machine

本节使用概率生成模型解决分类问题。

# 从盒子问题理解贝叶斯公式

假设有两个盒子：

$$
P(B_1)=\frac{2}{3},
\qquad
P(B_2)=\frac{1}{3}
$$

盒子中的球满足：

$$
\begin{aligned}
P(\mathrm{Blue}\mid B_1)&=\frac{4}{5}\\
P(\mathrm{Blue}\mid B_2)&=\frac{2}{5}
\end{aligned}
$$

现在随机选择一个盒子，再从盒子中拿出一个蓝球。

这个蓝球来自盒子 $B_1$ 的概率为：

$$
P(B_1\mid \mathrm{Blue})
=
\frac{
P(\mathrm{Blue}\mid B_1)P(B_1)
}{
P(\mathrm{Blue}\mid B_1)P(B_1)
+
P(\mathrm{Blue}\mid B_2)P(B_2)
}
$$

这就是贝叶斯公式。

# 二分类的贝叶斯公式

给定一个新的输入 $\mathbf{x}$，它属于类别 $C_1$ 的概率为：

$$
P(C_1\mid\mathbf{x})
=
\frac{
P(\mathbf{x}\mid C_1)P(C_1)
}{
P(\mathbf{x}\mid C_1)P(C_1)
+
P(\mathbf{x}\mid C_2)P(C_2)
}
$$

其中：

- $P(C_i)$：类别 $C_i$ 的先验概率
- $P(\mathbf{x}\mid C_i)$：类别 $C_i$ 产生特征 $\mathbf{x}$ 的概率
- $P(C_i\mid\mathbf{x})$：观察到 $\mathbf{x}$ 后属于 $C_i$ 的后验概率

分类规则为：

$$
f(\mathbf{x})
=
\begin{cases}
C_1,&P(C_1\mid\mathbf{x})>0.5\\
C_2,&P(C_1\mid\mathbf{x})\leq 0.5
\end{cases}
$$

# 先验概率

假设训练集中有：

- $79$ 个水系宝可梦
- $61$ 个一般系宝可梦

令：

$$
C_1=\text{水系},
\qquad
C_2=\text{一般系}
$$

则：

$$
P(C_1)
=
\frac{79}{79+61}
=
0.56
$$

$$
P(C_2)
=
\frac{61}{79+61}
=
0.44
$$

一般来说：

$$
P(C_i)=\frac{N_i}{N}
$$

其中 $N_i$ 是类别 $C_i$ 的样本数。

# 类别条件概率

接下来需要估计：

$$
P(\mathbf{x}\mid C_i)
$$

它表示：

> 已知样本属于类别 $C_i$，观察到特征 $\mathbf{x}$ 的概率。

每个宝可梦由一个特征向量表示：

$$
\mathbf{x}
=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_D
\end{bmatrix}
$$

为了估计 $P(\mathbf{x}\mid C_i)$，假设同一类别的特征由一个多元高斯分布产生。

# 多元高斯分布

高斯分布的概率密度函数为：

$$
f_{\boldsymbol{\mu},\boldsymbol{\Sigma}}(\mathbf{x})
=
\frac{1}
{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}}
\exp
\left(
-\frac{1}{2}
(\mathbf{x}-\boldsymbol{\mu})^{T}
\boldsymbol{\Sigma}^{-1}
(\mathbf{x}-\boldsymbol{\mu})
\right)
$$

其中：

- $D$：特征维度
- $\boldsymbol{\mu}$：均值向量
- $\boldsymbol{\Sigma}$：协方差矩阵
- $|\boldsymbol{\Sigma}|$：协方差矩阵的行列式

均值决定分布的中心位置：

$$
\boldsymbol{\mu}
=
\mathbb{E}[\mathbf{x}]
$$

协方差矩阵描述：

- 每个特征的变化范围
- 不同特征之间的相关关系
- 高斯分布的形状和方向

因此，可以假设：

$$
P(\mathbf{x}\mid C_1)
=
f_{\boldsymbol{\mu}_1,\boldsymbol{\Sigma}_1}(\mathbf{x})
$$

$$
P(\mathbf{x}\mid C_2)
=
f_{\boldsymbol{\mu}_2,\boldsymbol{\Sigma}_2}(\mathbf{x})
$$

# 最大似然估计

假设类别 $C_i$ 中共有 $N_i$ 个样本：

$$
\mathbf{x}^{(1)},
\mathbf{x}^{(2)},
\cdots,
\mathbf{x}^{(N_i)}
$$

这些样本同时由高斯分布产生的概率为：

$$
L(\boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i)
=
\prod_{n=1}^{N_i}
f_{\boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i}
\left(\mathbf{x}^{(n)}\right)
$$

这个函数称为似然函数。

最大似然估计要寻找最有可能产生训练数据的参数：

$$
\left(
\boldsymbol{\mu}_i^*,
\boldsymbol{\Sigma}_i^*
\right)
=
\arg\max_{\boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i}
L(\boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i)
$$

最大似然估计得到的均值为：

$$
\boldsymbol{\mu}_i^*
=
\frac{1}{N_i}
\sum_{n=1}^{N_i}
\mathbf{x}^{(n)}
$$

也就是该类别中所有样本的平均值。

协方差矩阵为：

$$
\boldsymbol{\Sigma}_i^*
=
\frac{1}{N_i}
\sum_{n=1}^{N_i}
\left(
\mathbf{x}^{(n)}-\boldsymbol{\mu}_i^*
\right)
\left(
\mathbf{x}^{(n)}-\boldsymbol{\mu}_i^*
\right)^T
$$

# 使用生成模型分类

训练阶段需要估计：

$$
P(C_1),\quad P(C_2)
$$

以及：

$$
\boldsymbol{\mu}_1,\quad
\boldsymbol{\mu}_2,\quad
\boldsymbol{\Sigma}_1,\quad
\boldsymbol{\Sigma}_2
$$

对于一个新的输入 $\mathbf{x}$，计算：

$$
P(C_1\mid\mathbf{x})
=
\frac{
f_{\boldsymbol{\mu}_1,\boldsymbol{\Sigma}_1}(\mathbf{x})P(C_1)
}{
f_{\boldsymbol{\mu}_1,\boldsymbol{\Sigma}_1}(\mathbf{x})P(C_1)
+
f_{\boldsymbol{\mu}_2,\boldsymbol{\Sigma}_2}(\mathbf{x})P(C_2)
}
$$

如果：

$$
P(C_1\mid\mathbf{x})>0.5
$$

则判断为 $C_1$，否则判断为 $C_2$。

之所以称为生成模型，是因为模型描述了每个类别产生特征 $\mathbf{x}$ 的过程：

$$
P(\mathbf{x},C_i)
=
P(\mathbf{x}\mid C_i)P(C_i)
$$

# 独立协方差矩阵的问题

最开始可以让每个类别拥有自己的协方差矩阵：

$$
\boldsymbol{\Sigma}_1\neq\boldsymbol{\Sigma}_2
$$

但是协方差矩阵包含大量参数。

当特征维度为 $D$ 时，一个协方差矩阵包含大约：

$$
D^2
$$

个元素。

参数过多而训练数据较少时，协方差估计可能不准确，导致模型泛化能力较差。

视频实验中：

- 使用 Defense 和 SP Defense 两个特征，测试准确率约为 $47\%$
- 使用全部 $7$ 个特征，测试准确率约为 $54\%$

增加特征后参数数量也随之增加，因此效果没有明显改善。

# 共享协方差矩阵

为了减少参数，可以假设两个类别使用相同的协方差矩阵：

$$
\boldsymbol{\Sigma}_1
=
\boldsymbol{\Sigma}_2
=
\boldsymbol{\Sigma}
$$

共享协方差矩阵的最大似然估计为：

$$
\boldsymbol{\Sigma}
=
\frac{N_1}{N_1+N_2}\boldsymbol{\Sigma}_1
+
\frac{N_2}{N_1+N_2}\boldsymbol{\Sigma}_2
$$

也就是两个类别协方差矩阵的加权平均。

共享协方差矩阵后：

- 模型需要估计的参数更少
- 在训练数据较少时通常更加稳定
- 视频案例的准确率由约 $54\%$ 提升到约 $73\%$
- 最终得到线性分类边界

这体现了模型复杂度与泛化能力之间的权衡。

# 其他概率分布

生成模型不一定必须使用高斯分布，可以根据数据特征选择其他分布。

如果假设各维特征在给定类别后彼此独立：

$$
P(\mathbf{x}\mid C_i)
=
\prod_{k=1}^{D}
P(x_k\mid C_i)
$$

得到的模型称为朴素贝叶斯分类器：

$$
\boxed{\text{Naive Bayes Classifier}}
$$

例如：

- 连续特征可以使用一维高斯分布
- 二元特征可以使用伯努利分布

朴素贝叶斯中的“朴素”指的是假设不同特征条件独立。

# 后验概率与 Sigmoid

二分类的后验概率为：

$$
P(C_1\mid\mathbf{x})
=
\frac{
P(\mathbf{x}\mid C_1)P(C_1)
}{
P(\mathbf{x}\mid C_1)P(C_1)
+
P(\mathbf{x}\mid C_2)P(C_2)
}
$$

上下同时除以 $P(\mathbf{x}\mid C_1)P(C_1)$：

$$
P(C_1\mid\mathbf{x})
=
\frac{
1
}{
1+
\frac{
P(\mathbf{x}\mid C_2)P(C_2)
}{
P(\mathbf{x}\mid C_1)P(C_1)
}
}
$$

定义：

$$
z
=
\ln
\frac{
P(\mathbf{x}\mid C_1)P(C_1)
}{
P(\mathbf{x}\mid C_2)P(C_2)
}
$$

则：

$$
P(C_1\mid\mathbf{x})
=
\frac{1}{1+\exp(-z)}
$$

定义 Sigmoid 函数：

$$
\sigma(z)
=
\frac{1}{1+\exp(-z)}
$$

因此：

$$
P(C_1\mid\mathbf{x})=\sigma(z)
$$

$z$ 表示两个类别概率的对数比值：

$$
z
=
\ln\frac{P(\mathbf{x}\mid C_1)}{P(\mathbf{x}\mid C_2)}
+
\ln\frac{P(C_1)}{P(C_2)}
$$

当：

$$
z>0
$$

有：

$$
P(C_1\mid\mathbf{x})>0.5
$$

因此判断为 $C_1$。

# 共享协方差下的线性形式

当两个类别使用相同协方差矩阵时：

$$
\boldsymbol{\Sigma}_1
=
\boldsymbol{\Sigma}_2
=
\boldsymbol{\Sigma}
$$

将高斯分布代入 $z$ 后，关于 $\mathbf{x}$ 的二次项会相互抵消，最终得到：

$$
z=\mathbf{w}^{T}\mathbf{x}+b
$$

其中：

$$
\mathbf{w}
=
\boldsymbol{\Sigma}^{-1}
(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)
$$

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

根据样本数量估计先验概率时：

$$
\ln\frac{P(C_1)}{P(C_2)}
=
\ln\frac{N_1}{N_2}
$$

因此：

$$
\boxed{
P(C_1\mid\mathbf{x})
=
\sigma(\mathbf{w}^{T}\mathbf{x}+b)
}
$$

分类边界满足：

$$
P(C_1\mid\mathbf{x})=0.5
$$

因为：

$$
\sigma(0)=0.5
$$

所以分类边界为：

$$
\boxed{
\mathbf{w}^{T}\mathbf{x}+b=0
}
$$

这是一个线性分类边界。

如果两个类别的协方差矩阵不同，关于 $\mathbf{x}$ 的二次项不会抵消，分类边界一般是非线性的。

# 生成式分类器的三个步骤

## Step 1：选择模型

假设：

$$
P(\mathbf{x}\mid C_i)
$$

服从某种概率分布，例如高斯分布。

后验概率为：

$$
P(C_1\mid\mathbf{x})
=
\frac{
P(\mathbf{x}\mid C_1)P(C_1)
}{
P(\mathbf{x}\mid C_1)P(C_1)
+
P(\mathbf{x}\mid C_2)P(C_2)
}
$$

## Step 2：定义函数的好坏

使用似然函数衡量某组概率分布参数产生训练数据的可能性：

$$
L(\boldsymbol{\mu},\boldsymbol{\Sigma})
=
\prod_n
f_{\boldsymbol{\mu},\boldsymbol{\Sigma}}
(\mathbf{x}^{(n)})
$$

似然越大，参数越好。

## Step 3：寻找最好的函数

通过最大似然估计寻找：

$$
(\boldsymbol{\mu}^*,\boldsymbol{\Sigma}^*)
=
\arg\max_{\boldsymbol{\mu},\boldsymbol{\Sigma}}
L(\boldsymbol{\mu},\boldsymbol{\Sigma})
$$

# 总结

1. 分类模型输出的是离散类别。
2. 不能简单地使用类别编号进行普通线性回归。
3. 贝叶斯公式可以根据先验概率和类别条件概率计算后验概率。
4. 生成模型学习 $P(\mathbf{x}\mid C_i)$ 和 $P(C_i)$。
5. 可以使用高斯分布描述每个类别中的特征分布。
6. 高斯分布的均值和协方差可以通过最大似然估计得到。
7. 分别估计每个类别的协方差矩阵需要较多参数。
8. 共享协方差矩阵可以减少参数并提升泛化能力。
9. 共享协方差时，后验概率可以写成：

$$
P(C_1\mid\mathbf{x})
=
\sigma(\mathbf{w}^{T}\mathbf{x}+b)
$$

10. 共享协方差产生线性分类边界。
11. 朴素贝叶斯假设给定类别后，各个特征彼此独立。