# Regression：Case Study

> 通过预测宝可梦进化后的 CP 值，介绍机器学习的基本流程。

## 回归问题

回归要寻找一个输出为连续数值的函数：

$$
f(\mathbf{x})=y
$$

其中：

- $\mathbf{x}$：输入特征
- $y$：需要预测的连续数值

本节案例中：

$$
\mathbf{x}
=
(x_{\mathrm{cp}},x_{\mathrm{hp}},x_{\mathrm{weight}},
x_{\mathrm{height}},x_{\mathrm{species}})
$$

$$
y=\text{进化后的 CP}
$$

# 机器学习的三个步骤

## Step 1：选择模型

模型是一个函数集合，而不是某一个确定函数。

首先假设进化后的 CP 只与进化前的 CP 有关：

$$
\hat y=b+w x_{\mathrm{cp}}
$$

其中：

- $x_{\mathrm{cp}}$：进化前的 CP，是一个特征
- $w$：权重
- $b$：偏置
- $\hat y$：模型预测值

不同的 $w,b$ 对应不同的函数，它们共同构成一个线性模型：

$$
\hat y=b+\sum_i w_i x_i
$$

之所以称为线性模型，是因为输出是各个特征的线性组合。

## Step 2：定义函数的好坏

训练数据可以写成：

$$
\left\{
\left(\mathbf{x}^{(1)},y^{(1)}\right),
\left(\mathbf{x}^{(2)},y^{(2)}\right),
\cdots,
\left(\mathbf{x}^{(N)},y^{(N)}\right)
\right\}
$$

定义损失函数衡量模型有多差：

$$
L(w,b)
=
\sum_{n=1}^{N}
\left(
y^{(n)}-
\left(b+w x_{\mathrm{cp}}^{(n)}\right)
\right)^2
$$

也就是：

$$
L(w,b)=\sum_{n=1}^{N}
\left(y^{(n)}-\hat y^{(n)}\right)^2
$$

每个样本都计算：

$$
\text{误差}=\text{真实值}-\text{预测值}
$$

然后将误差平方并求和。

- $L$ 越小，函数越好
- $L$ 越大，函数越差
- 参数空间中的每一个点 $(w,b)$ 都代表一个函数

## Step 3：寻找最好的函数

目标是寻找使损失最小的参数：

$$
(w^*,b^*)
=
\arg\min_{w,b}L(w,b)
$$

也可以写成：

$$
f^*=\arg\min_f L(f)
$$

这里使用梯度下降寻找参数。

# 梯度下降

## 一个参数

假设损失只有一个参数：

$$
L(w)
$$

从一个初始值 $w_0$ 开始，计算当前点的导数：

$$
\left.\frac{dL}{dw}\right|_{w=w_0}
$$

更新参数：

$$
w_1
=
w_0-\eta
\left.\frac{dL}{dw}\right|_{w=w_0}
$$

一般形式为：

$$
w_{t+1}
=
w_t-\eta\frac{dL}{dw}\bigg|_{w=w_t}
$$

其中 $\eta$ 为学习率。

- 导数为正：减小 $w$
- 导数为负：增大 $w$
- 导数绝对值越大：当前位置越陡，移动距离越大

反复更新，直到参数基本不再变化。

## 多个参数

当模型同时具有 $w,b$ 时：

$$
\begin{aligned}
w_{t+1}
&=
w_t-\eta\frac{\partial L}{\partial w}\\
b_{t+1}
&=
b_t-\eta\frac{\partial L}{\partial b}
\end{aligned}
$$

梯度为：

$$
\nabla L=
\begin{bmatrix}
\frac{\partial L}{\partial w}\\[4pt]
\frac{\partial L}{\partial b}
\end{bmatrix}
$$

因此参数整体沿负梯度方向更新：

$$
\begin{bmatrix}
w\\
b
\end{bmatrix}_{t+1}
=
\begin{bmatrix}
w\\
b
\end{bmatrix}_t
-\eta\nabla L
$$

平方损失对应的偏导数为：

$$
\frac{\partial L}{\partial w}
=
-2\sum_{n=1}^{N}
\left(
y^{(n)}-b-wx_{\mathrm{cp}}^{(n)}
\right)x_{\mathrm{cp}}^{(n)}
$$

$$
\frac{\partial L}{\partial b}
=
-2\sum_{n=1}^{N}
\left(
y^{(n)}-b-wx_{\mathrm{cp}}^{(n)}
\right)
$$

在线性回归中，平方损失关于参数是凸函数，因此不存在较差的局部最小值问题。

# 训练数据与测试数据

训练得到的直线为：

$$
\hat y=-188.4+2.7x_{\mathrm{cp}}
$$

结果：

- 训练误差：$31.9$
- 测试误差：$35.0$

真正关心的是模型在没有见过的新数据上的表现，也就是测试误差，这种能力称为泛化能力。

# 模型选择与过拟合

只使用一次函数效果不够好，可以尝试更复杂的多项式模型：

$$
\hat y=b+w_1x_{\mathrm{cp}}
+w_2x_{\mathrm{cp}}^2+\cdots+w_kx_{\mathrm{cp}}^k
$$

实验结果：

| 多项式次数 | 训练误差 | 测试误差 |
| ---: | ---: | ---: |
| 1 | 31.9 | 35.0 |
| 2 | 15.4 | 18.4 |
| 3 | 15.3 | 18.1 |
| 4 | 14.9 | 28.2 |
| 5 | 12.8 | 232.1 |

随着模型复杂度增加：

- 训练误差持续下降
- 测试误差先下降，随后急剧上升

模型在训练数据上表现很好，但在新数据上表现很差，这称为：

$$
\boxed{\text{Overfitting（过拟合）}}
$$

因此，并不是模型越复杂越好，而是需要选择复杂度适中的模型。

# 重新设计特征

只使用原始 CP 无法解释所有数据，因为还存在隐藏因素。

不同种类的宝可梦具有不同的进化规律：

$$
\hat y=
\begin{cases}
b_1+w_1x_{\mathrm{cp}},&x_s=\text{Pidgey}\\
b_2+w_2x_{\mathrm{cp}},&x_s=\text{Weedle}\\
b_3+w_3x_{\mathrm{cp}},&x_s=\text{Caterpie}\\
b_4+w_4x_{\mathrm{cp}},&x_s=\text{Eevee}
\end{cases}
$$

可以用指示函数表示种类：

$$
\delta(x_s=\text{Pidgey})
=
\begin{cases}
1,&x_s=\text{Pidgey}\\
0,&\text{otherwise}
\end{cases}
$$

例如：

$$
\begin{aligned}
\hat y={}&
b_1\delta(x_s=\text{Pidgey})
+b_2\delta(x_s=\text{Weedle})+\cdots\\
&+
w_1\delta(x_s=\text{Pidgey})x_{\mathrm{cp}}
+w_2\delta(x_s=\text{Weedle})x_{\mathrm{cp}}+\cdots
\end{aligned}
$$

虽然表达式变长，但它仍然是参数的线性组合，因此仍属于线性模型。

加入种类信息后：

- 训练误差：$14.3$
- 测试误差：$3.8$

说明选择正确的特征往往比盲目增加多项式次数更加有效。

# 加入更多特征

还可以加入：

- HP
- 身高
- 体重
- CP 的平方项
- 其他特征的平方项

但一次加入太多特征后：

- 训练误差降至 $1.9$
- 测试误差升至 $102.3$

模型再次发生过拟合。

这说明：

$$
\text{训练误差低}\not\Rightarrow\text{泛化能力强}
$$

# 正则化

为了限制模型过于复杂，在原损失后加入权重惩罚：

$$
L=
\sum_n
\left(
y^{(n)}-
\left(b+\sum_iw_ix_i^{(n)}\right)
\right)^2
+
\lambda\sum_iw_i^2
$$

第一部分：

$$
\sum_n(y^{(n)}-\hat y^{(n)})^2
$$

要求模型拟合训练数据。

第二部分：

$$
\lambda\sum_iw_i^2
$$

要求权重不要太大。

这里通常不惩罚 $b$，因为改变偏置只会让整个函数上下移动，不会改变函数的平滑程度。

## 为什么希望权重较小

模型为：

$$
\hat y=b+\sum_iw_ix_i
$$

输入发生微小变化 $\Delta x_i$ 时：

$$
\Delta\hat y=\sum_iw_i\Delta x_i
$$

当 $w_i$ 较小时：

- 输入中出现少量噪声
- 输出不会发生剧烈变化
- 函数更加平滑
- 模型对新数据更加稳定

因此，正则化是在拟合训练数据和保持模型简单平滑之间进行权衡。

## 正则化强度

$\lambda$ 控制正则化程度：

- $\lambda=0$：完全不限制权重，容易过拟合
- $\lambda$ 较大：权重更小，模型更平滑
- $\lambda$ 过大：模型过于简单，产生欠拟合

实验结果：

| $\lambda$ | 训练误差 | 测试误差 |
| ---: | ---: | ---: |
| 0 | 1.9 | 102.3 |
| 10 | 3.5 | 25.7 |
| 100 | 4.1 | 11.1 |
| 1000 | 5.6 | 12.8 |
| 100000 | 8.5 | 26.8 |

训练误差随着 $\lambda$ 增大而上升，但测试误差先下降再上升。

因此，正则化也不能越强越好，需要选择合适的 $\lambda$。

# 总结

机器学习的基本流程：

$$
\boxed{
\text{选择模型}
\rightarrow
\text{定义损失函数}
\rightarrow
\text{优化参数}
}
$$

在实际应用中，还要不断重复这一流程：

$$
\boxed{
\text{观察结果}
\rightarrow
\text{重新设计特征或模型}
\rightarrow
\text{重新训练}
}
$$