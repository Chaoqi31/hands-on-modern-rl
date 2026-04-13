# E.3 微积分与优化

> 相关章节：[第5章 策略梯度](/chapter05_policy_gradient/policy-gradient)、[第6章 PPO 数学推导](/chapter06_ppo/ppo-math)、[第8章 GRPO](/chapter08_grpo_rlvr/grpo-mechanism)

优化是强化学习的核心操作。训练 RL 模型的过程，本质上是对参数 $\theta$ 反复调整，使得某个目标函数最大化或最小化。微积分提供了"朝哪个方向调整"和"调整多少"的数学工具。

本节从单变量导数出发，推广到多变量的梯度与 Hessian 矩阵，介绍梯度下降的各类变体。在开始之前，先看看本书正文中用到了哪些微积分公式——你现在不需要完全理解它们，每学完一个概念，我们就回头解锁一个。

## 本书中你将遇到的微积分公式

以下是本书正文中依赖微积分的三个核心公式。如果你现在看不懂，不用担心——下面每一节都会帮你解锁其中的一部分。

**公式① 策略梯度定理**（[第5章 策略梯度](/chapter05_policy_gradient/policy-gradient)）

$$\nabla_\theta J(\theta) = \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a \mid s) \cdot Q^\pi(s, a)$$

涉及概念：**导数**、**链式法则**、**梯度**。推导过程用到了乘积法则和对数导数技巧 $\nabla \pi = \pi \cdot \nabla \log \pi$。

**公式② PPO 裁剪的 Taylor 分析**（[第6章 PPO](/chapter06_ppo/ppo-math)）

概率比 $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$ 在 $\theta = \theta_{old}$ 处的二阶 Taylor 展开：

$$r_t(\theta) \approx 1 + \nabla_\theta r_t \cdot (\theta - \theta_{old}) + \frac{1}{2}(\theta - \theta_{old})^\top \nabla^2 r_t (\theta - \theta_{old})$$

涉及概念：**Taylor 展开**。PPO 用裁剪替代显式的二阶约束——效果上等价于限制每步更新的二阶变化量。

**公式③ GRPO 的组归一化**（[第8章 GRPO](/chapter08_grpo_rlvr/grpo-mechanism)）

$$\hat{A}_i = \frac{r_i - \mu}{\sigma}$$

涉及概念：**标准化**。减去均值使数据中心化，除以标准差使数据归一化——这是微积分中"标准化"的直接应用。

---

接下来，我们从最基本的导数概念开始。

## 导数

考虑一个具体的场景。假设我们有一个神经网络，参数为 $\theta$，损失函数为 $\mathcal{L}(\theta)$。这个函数极其复杂——它编码了给定架构的所有可能模型在该数据集上的表现。我们几乎不可能直接找到使 $\mathcal{L}$ 最小的 $\theta$。因此实践中，我们从随机初始化出发，然后沿使损失下降最快的方向走一小步。

要回答"往哪个方向走"，首先需要理解函数在一个点附近的行为。

**定义（导数）.** 函数 $f: \mathbb{R} \to \mathbb{R}$ 在点 $x$ 处的导数定义为

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

其中 $h$ 是自变量的微小增量，分母是函数值的平均变化率，取极限 $h \to 0$ 后得到瞬时变化率。

![导数的几何含义：切线斜率等于该点的瞬时变化率](./images/tangent-line.svg)

导数的几何含义是函数图像在该点处切线的斜率。它回答了一个基本问题：若自变量沿正方向移动一个无穷小量，函数值将如何变化？导数为正时函数局部递增，导数为负时函数局部递减，导数为零时函数在该点取极值或为鞍点。

函数在 $x$ 附近的局部行为可以用一阶 Taylor 展开来近似：

$$f(x + h) \approx f(x) + f'(x) \cdot h$$

要让 $f$ 下降，只需选 $h$ 使 $f'(x) \cdot h < 0$。最简单的选择是 $h = -\alpha f'(x)$（$\alpha > 0$），此时 $f(x + h) \approx f(x) - \alpha [f'(x)]^2 < f(x)$。这就是梯度下降的基本想法。

## 链式法则

**定理（链式法则）.** 设 $y = f(g(x))$，其中 $g$ 在 $x$ 处可导，$f$ 在 $g(x)$ 处可导，则

$$\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$$

链式法则的直观含义是：变化沿复合结构逐层传递。$x$ 的微小变化引起 $g(x)$ 的变化，$g(x)$ 的变化又引起 $f(g(x))$ 的变化，总变化率等于各层变化率的乘积。

在 RL 中，策略梯度 $\nabla_\theta \log \pi_\theta(a \mid s)$ 的计算就是链式法则的应用。$\pi_\theta$ 通常是一个多层神经网络，$\log \pi$ 对第 $l$ 层权重 $\mathbf{W}_l$ 的梯度为

$$\frac{\partial \log \pi}{\partial \mathbf{W}_l} = \frac{\partial \log \pi}{\partial \mathbf{h}_L} \cdot \frac{\partial \mathbf{h}_L}{\partial \mathbf{h}_{L-1}} \cdots \frac{\partial \mathbf{h}_{l+1}}{\partial \mathbf{h}_l} \cdot \frac{\partial \mathbf{h}_l}{\partial \mathbf{W}_l}$$

每一项 $\frac{\partial \mathbf{h}_{k+1}}{\partial \mathbf{h}_k}$ 是一层变换的 Jacobian 矩阵。整个梯度是各层 Jacobian 的连乘——这正是反向传播（backpropagation）的数学本质。

> **回头看公式①（部分解锁）.** 现在你知道了导数和链式法则，再来看开头的策略梯度推导中的关键步骤：
>
> 对数导数技巧 $\nabla_\theta \pi_\theta(a \mid s) = \pi_\theta(a \mid s) \cdot \nabla_\theta \log \pi_\theta(a \mid s)$ 其实就是链式法则。设 $u = \pi_\theta$，$f(u) = \log u$，则 $\nabla \log \pi = \frac{1}{\pi} \cdot \nabla \pi$，两边乘以 $\pi$ 即得 $\nabla \pi = \pi \cdot \nabla \log \pi$。这个技巧将难以计算的 $\nabla \pi$ 转化为容易计算的 $\pi \cdot \nabla \log \pi$。
>
> 还不能完全理解公式①——还需要知道**梯度**的概念，我们继续往下学。

## 梯度

当函数有多个自变量时，导数的概念推广为**梯度**。

**定义（梯度）.** 设 $f: \mathbb{R}^n \to \mathbb{R}$，则 $f$ 在 $\mathbf{x}$ 处的梯度定义为

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

其中 $\frac{\partial f}{\partial x_i}$ 是 $f$ 对第 $i$ 个变量的偏导数（其他变量视为常数）。

梯度的核心性质是：**梯度方向是函数局部增长最快的方向**，因此**负梯度方向是函数局部下降最快的方向**。这一结论可由 Cauchy-Schwarz 不等式严格证明。

![梯度下降沿负梯度方向移动，逐步逼近最小值](./images/gradient-descent.svg)

由此直接得到梯度下降的基本形式：

$$\theta \leftarrow \theta - \alpha \nabla f(\theta)$$

其中 $\alpha > 0$ 为学习率（步长）。

> **回头看公式①（进一步解锁）.** 现在你知道了梯度，再来看策略梯度定理的完整推导：
>
> 目标 $J(\theta) = \sum_s d^\pi(s) \sum_a \pi_\theta(a \mid s) \cdot Q^\pi(s, a)$ 对 $\theta$ 求梯度时，先用乘积法则拆开 $\nabla_\theta[\pi \cdot Q]$，再用对数导数技巧 $\nabla \pi = \pi \cdot \nabla \log \pi$ 重组，最后利用期望的定义合并，得到 $\nabla_\theta J = \mathbb{E}_\pi[\nabla \log \pi \cdot Q]$。这里的 $\nabla_\theta$ 就是梯度——对所有参数的偏导数组成的向量。
>
> 参数更新 $\theta \leftarrow \theta + \alpha \nabla_\theta J$ 就是沿梯度方向走一步。公式①已完全解锁。

## 多变量微积分的扩展

### Jacobian 矩阵

当函数的输出也是多维的（$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$），偏导数排列成 Jacobian 矩阵：

$$\mathbf{J} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

Jacobian 描述了多输入多输出函数的局部线性近似。梯度是 Jacobian 在 $m = 1$ 时的特例（转置）。

### Hessian 矩阵

二阶偏导数排列成 Hessian 矩阵：

$$\mathbf{H} = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$

Hessian 描述了函数的**曲率**——梯度的变化率。在优化中它用于判断驻点的性质：

- Hessian 正定（所有特征值 $> 0$）→ 局部极小值
- Hessian 负定（所有特征值 $< 0$）→ 局部极大值
- 特征值既有正又有负 → 鞍点

二阶优化方法（如牛顿法）利用 Hessian 的逆矩阵调整步长。在 RL 中，TRPO 的 KL 约束可以用 Fisher 信息矩阵（Hessian 的期望形式）来近似。

## 梯度下降的演化

### 梯度下降

$$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)$$

每步沿负梯度方向移动 $\alpha$ 步。学习率 $\alpha$ 过大则训练发散，过小则收敛极慢。

### 随机梯度下降（SGD）

$$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta; \mathbf{x}_i)$$

SGD 用单个样本（或小批量）估计梯度，而非全量数据。代价是梯度估计含有噪声，但优势在于：(1) 每步计算代价低；(2) 噪声具有隐式正则化效果，有助于逃离局部极小值。

强化学习天然就是随机的——每条轨迹仅提供真实梯度的一个有噪估计。REINFORCE 算法（[第5章](/chapter05_policy_gradient/policy-gradient)）即是用单条轨迹的样本来估计策略梯度。

### Adam

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\theta \leftarrow \theta - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

其中 $g_t = \nabla_\theta \mathcal{L}(\theta_t)$ 是当前步的梯度，$m_t$ 是一阶矩（指数移动平均），$v_t$ 是二阶矩（梯度的平方的指数移动平均），$\hat{m}_t = m_t / (1 - \beta_1^t)$ 和 $\hat{v}_t = v_t / (1 - \beta_2^t)$ 是偏差校正项，$\epsilon$（通常取 $10^{-8}$）防止分母为零。

Adam 是 RL 训练中最常用的优化器。原因在于 RL 梯度的两个特点：(1) 梯度噪声极大；(2) 不同参数的梯度尺度差异显著。Adam 的自适应学习率为每个参数单独调整步长，自动适应这种异质性。默认超参数 $\beta_1 = 0.9$、$\beta_2 = 0.999$、$\epsilon = 10^{-8}$ 在大多数 RL 任务中无需调整。

## 凸优化基础

**定义（凸函数）.** 函数 $f$ 称为凸函数，当且仅当对任意 $\mathbf{x}, \mathbf{y}$ 和 $\lambda \in [0, 1]$，有

$$f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \le \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y})$$

凸函数的基本性质是：**局部最优即全局最优**。从任何初始点出发，梯度下降都能收敛到全局最优解。

RL 的目标函数几乎都是非凸的——策略参数 $\theta$ 与目标函数 $J(\theta)$ 之间经过多层非线性变换，存在大量局部极小值和鞍点。这是 RL 训练困难的根本原因之一。尽管如此，正则化项（如权重衰减 $\lambda \|\theta\|_2^2$）是凸的，附加到非凸目标上可以使优化景观更平滑。

## Taylor 展开

Taylor 展开用多项式近似一个函数在给定点的局部行为。一阶 Taylor 展开为

$$f(x + h) \approx f(x) + f'(x) \cdot h$$

多变量情形下为

$$f(\mathbf{x} + \boldsymbol{h}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \boldsymbol{h}$$

二阶 Taylor 展开引入 Hessian：

$$f(\mathbf{x} + \boldsymbol{h}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \boldsymbol{h} + \frac{1}{2}\boldsymbol{h}^\top \mathbf{H} \boldsymbol{h}$$

Taylor 展开为梯度下降提供了严格的理论依据：在 $\mathbf{x}$ 的邻域内，函数可被线性近似。使 $f$ 下降最多的 $\boldsymbol{h}$ 即为 $\boldsymbol{h} = -\alpha \nabla f(\mathbf{x})$。

> **回头看公式②.** 现在你知道了 Taylor 展开，再来看开头的 PPO 裁剪分析：
>
> 概率比 $r_t(\theta) = \pi_\theta / \pi_{\theta_{old}}$ 在 $\theta = \theta_{old}$ 处展开：零阶项是 1（新旧策略相同时概率比为 1），一阶项反映策略变化的方向，二阶项反映变化的速度。KL 约束的本质就是限制二阶项的大小。PPO 用裁剪 $\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)$ 替代显式的二阶约束——当 $r_t$ 超出范围时直接截断，效果上等价于限制每步更新的二阶变化量。
>
> 公式②已完全解锁。还剩公式③涉及标准化，其实你已经会了——$\hat{A}_i = (r_i - \mu)/\sigma$ 就是减去均值再除以标准差。公式③也解锁了。
>
> 至此，开头的三个公式你都已经能够理解了。

## 公式速查

| 概念        | 公式                                                                                           | 应用场景                |
| ----------- | ---------------------------------------------------------------------------------------------- | ----------------------- |
| 导数        | $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$                                              | 优化基础                |
| 链式法则    | $\frac{d}{dx}f(g(x)) = f'(g) \cdot g'(x)$                                                     | 反向传播、策略梯度计算  |
| 梯度        | $\nabla f = [\partial f/\partial x_1, \ldots, \partial f/\partial x_n]^\top$                   | 参数更新方向            |
| 梯度下降    | $\theta \leftarrow \theta - \alpha \nabla \mathcal{L}$                                         | RL 优化的基本形式      |
| Adam        | $\theta \leftarrow \theta - \alpha \hat{m}/(\sqrt{\hat{v}} + \epsilon)$                        | RL 最常用的优化器      |
| 一阶 Taylor | $f(\mathbf{x}+\boldsymbol{h}) \approx f(\mathbf{x}) + \nabla f^\top \boldsymbol{h}$           | PPO/GRPO 的局部近似分析 |
