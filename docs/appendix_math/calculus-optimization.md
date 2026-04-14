# E.3 微积分与优化

> 相关章节：[第5章 策略梯度](/chapter05_policy_gradient/policy-gradient)、[第6章 PPO 数学推导](/chapter06_ppo/ppo-math)、[第8章 GRPO](/chapter08_grpo_rlvr/grpo-mechanism)

优化是强化学习的核心操作。训练 RL 模型的过程，本质上是对参数 $\theta$ 反复调整，使得某个目标函数最大化或最小化。微积分提供了"朝哪个方向调整"和"调整多少"的数学工具。

本节从单变量导数出发，推广到多变量的梯度与 Hessian 矩阵，介绍梯度下降的各类变体。在开始之前，先看看本书正文中用到了哪些微积分公式。你不需要一次看懂——每学完下面一个概念，我们就回来拆解这些公式中对应的部分，直到全部解锁。

## 本书中你将遇到的微积分公式

以下是本书正文中依赖微积分的三个核心公式。每个公式先给出整体目标（它要算什么、为什么重要），再逐符号拆解。详细的符号表默认折叠，点击展开即可查看。后面每学完一个知识点，我们会回来解锁对应的部分。

### 公式① 策略梯度定理

告诉智能体"参数往哪个方向调，平均回报会变高"——就像调收音机旋钮时，系统告诉你每个旋钮微调后信号会变好还是变差。出自[第5章 策略梯度](/chapter05_policy_gradient/policy-gradient)。

$$\nabla_\theta J(\theta) = \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a \mid s) \cdot Q^\pi(s, a)$$

<details>
<summary>点击展开：逐符号拆解</summary>

| 符号                                 | 读作     | 含义                                                                 |
| ------------------------------------ | -------- | -------------------------------------------------------------------- |
| $\theta$                             | theta    | 神经网络的全部参数（权重和偏置），是一个很长的向量                   |
| $J(\theta)$                          | 目标函数 | 智能体的**平均回报**，我们希望它越大越好                             |
| $\nabla_\theta$                      | 梯度     | 对 $\theta$ 的每个分量分别求偏导，组成一个向量——下文学完"梯度"你就懂 |
| $\pi_\theta(a \mid s)$               | 策略     | 给定状态 $s$ 时，参数为 $\theta$ 的网络选择动作 $a$ 的概率           |
| $\nabla_\theta \pi_\theta(a \mid s)$ | 策略梯度 | "参数微调一点，选择动作 $a$ 的概率变化多少"                          |
| $Q^\pi(s, a)$                        | Q 值     | 在状态 $s$ 执行动作 $a$ 后，**未来预期能拿多少总回报**               |
| $d^\pi(s)$                           | 状态分布 | 在策略 $\pi$ 下，**状态 $s$ 被访问的频率**（越常出现的状态权重越大） |
| $\sum_s \sum_a$                      | 双重求和 | 遍历所有可能的状态和动作，加权求和——本质是求期望                     |

**用一句话串起来**：把"每个状态下、每个动作"的策略变化量 $\nabla_\theta \pi$，乘以该动作的好坏程度 $Q$，再按状态出现频率 $d^\pi$ 加权求和——就得到"调整参数后平均回报怎么变"。

**搭建路径**（你将如何一步步理解它）：

1. 学完 **导数** → 理解"变化率"的含义
2. 学完 **链式法则** → 理解对数导数技巧 $\nabla \pi = \pi \cdot \nabla \log \pi$（本公式推导的关键步骤）
3. 学完 **梯度** → 理解 $\nabla_\theta$ 是对多维参数同时求导，公式①完全解锁

</details>

### 公式② PPO 裁剪的 Taylor 分析

把"新策略和旧策略差了多少"拆成基准值 + 线性变化 + 曲率修正三项，帮助理解 PPO 的裁剪机制为什么能限制策略变化幅度。就像预测走几步后海拔变多少：先猜不变（零阶），再按坡度修正（一阶），再加曲率修正（二阶）。出自[第6章 PPO](/chapter06_ppo/ppo-math)。

先定义概率比：

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$$

然后用二阶 Taylor 展开分析：

$$r_t(\theta) \approx \underbrace{1}_{\text{零阶}} + \underbrace{\nabla_\theta r_t \cdot (\theta - \theta_{old})}_{\text{一阶：线性变化}} + \underbrace{\frac{1}{2}(\theta - \theta_{old})^\top \nabla^2 r_t (\theta - \theta_{old})}_{\text{二阶：曲率}}$$

<details>
<summary>点击展开：逐符号拆解</summary>

**概率比 $r_t$ 的符号拆解：**

| 符号                               | 读作       | 含义                                                                                     |
| ---------------------------------- | ---------- | ---------------------------------------------------------------------------------------- |
| $\pi_\theta(a_t \mid s_t)$         | 新策略概率 | 当前参数下，在状态 $s_t$ 选动作 $a_t$ 的概率                                             |
| $\pi_{\theta_{old}}(a_t \mid s_t)$ | 旧策略概率 | 上一轮参数下，同一状态选同一动作的概率                                                   |
| $r_t(\theta)$                      | 概率比     | 新旧概率的比值。$r_t = 1$ 表示没变；$> 1$ 表示新策略更倾向选这个动作；$< 1$ 表示更不倾向 |

**Taylor 展开各项的作用：**

| 项                                                                                       | 含义                                                     | 直觉                               |
| ---------------------------------------------------------------------------------------- | -------------------------------------------------------- | ---------------------------------- |
| $1$（零阶项）                                                                            | $\theta = \theta_{old}$ 时 $r_t = 1$（新旧策略完全相同） | 起点处的基准值                     |
| $\nabla_\theta r_t \cdot (\theta - \theta_{old})$（一阶项）                              | 参数偏移引起的线性变化                                   | "走一小步，概率比大约变化这么多"   |
| $\frac{1}{2}(\theta - \theta_{old})^\top \nabla^2 r_t (\theta - \theta_{old})$（二阶项） | 线性近似不够准确时，加上曲率修正                         | "拐弯的程度"——变化越剧烈，这项越大 |

PPO 的裁剪操作 $\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)$ 本质上就是把 $r_t$ 限制在一个范围内，不让二阶项变得太大——相当于隐式地约束了策略的变化速度。

**搭建路径**：

1. 学完 **导数** → 理解一阶近似 $f(x+h) \approx f(x) + f'(x) \cdot h$
2. 学完 **梯度** → 理解多变量版本 $f(\mathbf{x}+\mathbf{h}) \approx f(\mathbf{x}) + \nabla f^\top \mathbf{h}$
3. 学完 **Taylor 展开** → 理解二阶项的加入与 Hessian 矩阵，公式②完全解锁

</details>

### 公式③ GRPO 的组归一化

把一组原始奖励分数标准化为"偏离平均值多少个标准差"（z-score），让不同尺度的奖励可以公平比较——就像两个班级考不同试卷，标准化后才能判断谁更优秀。出自[第8章 GRPO](/chapter08_grpo_rlvr/grpo-mechanism)。

$$\hat{A}_i = \frac{r_i - \mu}{\sigma}$$

<details>
<summary>点击展开：逐符号拆解</summary>

| 符号        | 读作          | 含义                                                                        |
| ----------- | ------------- | --------------------------------------------------------------------------- |
| $r_i$       | 第 $i$ 个奖励 | 同一个问题的第 $i$ 个回答获得的原始奖励分数                                 |
| $\mu$       | 均值          | 这组回答的奖励平均值，$\mu = \frac{1}{n}\sum_{i=1}^n r_i$                   |
| $\sigma$    | 标准差        | 这组奖励的离散程度，$\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^n (r_i - \mu)^2}$ |
| $\hat{A}_i$ | 标准化优势    | 第 $i$ 个回答相对于平均水平的标准化得分                                     |

**两步操作的含义**：

1. **减均值** $r_i - \mu$：把数据中心化。大于 0 表示比平均好，小于 0 表示比平均差。
2. **除标准差** $(r_i - \mu) / \sigma$：把数据缩放到统一尺度。结果的绝对值表示"偏离平均值多少个标准差"——这正是统计学中的 **z-score**。

**为什么需要归一化？** 不同问题的原始奖励尺度可能差异巨大（问题 A 的奖励在 0\~1，问题 B 的在 100\~200）。标准化后所有问题的优势都在同一尺度上，训练更稳定。

**搭建路径**：这个公式不依赖高深概念，理解"均值"和"标准差"即可。公式③已解锁。

</details>

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

> **回头看公式①（部分解锁——你已掌握导数 + 链式法则）**
>
> <details>
> <summary>点击展开：对数导数技巧的完整推导</summary>
>
> 开头给出的公式①是：
>
> $$\nabla_\theta J(\theta) = \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a \mid s) \cdot Q^\pi(s, a)$$
>
> 其中有一项 $\nabla_\theta \pi_\theta(a \mid s)$——"策略概率对参数的变化率"。直接算它很困难，但有一个巧妙的改写叫**对数导数技巧**：
>
> $$\nabla_\theta \pi = \pi \cdot \nabla_\theta \log \pi$$
>
> 它来自链式法则，我们逐步拆解：
>
> **第一步：设中间变量**
>
> 设 $u = \pi_\theta(a \mid s)$，即策略概率本身。那么 $\log \pi_\theta(a \mid s) = \log u$。我们想求的是 $\frac{d \log u}{d \theta}$。
>
> **第二步：识别出复合结构**
>
> $\log u$ 是一个复合函数：外层是 $\log(\cdot)$，内层是 $u = \pi_\theta$。这正是链式法则的适用场景！回顾刚才学的链式法则：
>
> $$\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$$
>
> 对应过来：外层函数 $f = \log$，内层函数 $g = \pi_\theta$，自变量 $x = \theta$。
>
> **第三步：应用链式法则**
>
> $$\frac{d \log u}{d \theta} = \underbrace{\frac{d \log u}{du}}_{= 1/u} \cdot \underbrace{\frac{du}{d\theta}}_{= \nabla_\theta \pi} = \frac{1}{\pi} \cdot \nabla_\theta \pi$$
>
> 其中 $\frac{d \log u}{du} = 1/u$ 是对数函数的导数（这是一个基本求导公式，和链式法则无关，属于预备知识）。
>
> **第四步：两边乘以 $\pi$，得到最终形式**
>
> $$\pi \cdot \nabla_\theta \log \pi = \pi \cdot \frac{1}{\pi} \cdot \nabla_\theta \pi = \nabla_\theta \pi$$
>
> **为什么这个改写有用？** 因为 $\log \pi$ 的梯度可以通过**反向传播**自动计算（神经网络框架帮你做了链式法则的逐层展开），而直接算 $\nabla \pi$ 反而难算。这个技巧把"难算的"变成了"好算的"。
>
> **已解锁的部分**：你现在理解了公式①中 $\nabla_\theta \pi_\theta(a \mid s)$ 这一项是怎么算出来的。
>
> **还没解锁的部分**：$\nabla_\theta$ 下标的意思是"对所有参数同时求导"——这是下节"梯度"要讲的内容。继续往下学。
>
> </details>

## 梯度

当函数有多个自变量时，导数的概念推广为**梯度**。

**定义（梯度）.** 设 $f: \mathbb{R}^n \to \mathbb{R}$，则 $f$ 在 $\mathbf{x}$ 处的梯度定义为

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

其中 $\frac{\partial f}{\partial x_i}$ 是 $f$ 对第 $i$ 个变量的偏导数（其他变量视为常数）。

梯度的核心性质是：**梯度方向是函数局部增长最快的方向**，因此**负梯度方向是函数局部下降最快的方向**。这一结论可由 Cauchy-Schwarz 不等式严格证明。

![梯度下降沿负梯度方向移动，逐步逼近最小值](./images/gradient-descent.svg)

图中的**红色箭头** $\nabla f$ 指向函数值增长最快的方向，因此会朝着更高的等高线"向外走"；**绿色箭头** $-\nabla f$ 与之方向相反，表示函数值下降最快的方向，因此会一步步朝最小值移动。也就是说，若我们的目标是**最小化** $f(\theta)$，更新时就必须取负号；若直接沿 $\nabla f$ 更新，参数会往"上坡"方向走，函数值反而增大。

这也解释了为什么有时你会看到"梯度上升"。如果目标是**最大化**某个目标函数 $J(\theta)$（例如强化学习里的期望回报），那就沿 $+\nabla J(\theta)$ 更新；在工程实现中，人们常把 $-J(\theta)$ 写成 loss，再继续使用统一的梯度下降框架。

由此直接得到梯度下降的基本形式：

$$\theta \leftarrow \theta - \alpha \nabla f(\theta)$$

其中 $\alpha > 0$ 为学习率（步长）。

> **回头看公式①（完全解锁——你已掌握导数 + 链式法则 + 梯度）**
>
> <details>
> <summary>点击展开：策略梯度定理的完整推导</summary>
>
> 开头给出的公式①是：
>
> $$\nabla_\theta J(\theta) = \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a \mid s) \cdot Q^\pi(s, a)$$
>
> 现在所有零件都齐了，让我们从"目标是什么"出发，逐步推导出这个公式。
>
> ---
>
> **目标**：我们要最大化智能体的平均回报 $J(\theta)$。方法是算出 $J(\theta)$ 对参数 $\theta$ 的梯度 $\nabla_\theta J$，然后沿梯度方向更新参数。
>
> ---
>
> **第一步：写出目标函数的展开形式**
>
> 平均回报 $J(\theta)$ 可以展开为"所有状态下、所有动作"的加权和：
>
> $$J(\theta) = \sum_s d^\pi(s) \sum_a \pi_\theta(a \mid s) \cdot Q^\pi(s, a)$$
>
> 回顾每个符号：
>
> - $d^\pi(s)$：状态 $s$ 被访问的频率（权重）
> - $\pi_\theta(a \mid s)$：在状态 $s$ 选动作 $a$ 的概率
> - $Q^\pi(s, a)$：执行动作 $a$ 后的预期总回报
>
> 整个式子的含义：对每个状态，按出现频率加权；在每个状态下，对每个动作，按"选择概率 × 回报"求和。
>
> ---
>
> **第二步：对 $\theta$ 求梯度**
>
> 现在我们要算 $\nabla_\theta J(\theta)$——"参数 $\theta$ 的每个分量各变一点点，$J$ 会怎么变"。
>
> 回顾刚学的**梯度**概念：$\nabla_\theta$ 就是对 $\theta$ 的每个分量求偏导，组成向量。
>
> 关键观察：$Q^\pi(s,a)$ 在这里是一个**固定值**（它评估的是当前策略下的回报，不是参数的函数）。而 $d^\pi(s)$ 也近似不变（短期内状态分布不变）。所以梯度只作用在 $\pi_\theta(a \mid s)$ 上：
>
> $$\nabla_\theta J = \sum_s d^\pi(s) \sum_a \underbrace{\nabla_\theta \pi_\theta(a \mid s)}_{\text{只对这一项求导}} \cdot Q^\pi(s, a)$$
>
> 这就是开头的公式①！但 $\nabla_\theta \pi$ 直接算不好算。
>
> ---
>
> **第三步：用对数导数技巧改写**
>
> 用上节学的对数导数技巧 $\nabla_\theta \pi = \pi \cdot \nabla_\theta \log \pi$ 替换：
>
> $$\nabla_\theta J = \sum_s d^\pi(s) \sum_a \underbrace{\pi_\theta(a \mid s)}_{\text{策略概率}} \cdot \underbrace{\nabla_\theta \log \pi_\theta(a \mid s)}_{\text{用反向传播好算}} \cdot \underbrace{Q^\pi(s, a)}_{\text{动作的好坏}}$$
>
> 现在 $\nabla_\theta \log \pi$ 可以通过反向传播高效计算（链式法则逐层展开，框架自动完成）。
>
> ---
>
> **第四步：重写为期望形式**
>
> 注意到 $\sum_a \pi_\theta(a \mid s) \cdot [\text{某个关于 } a \text{ 的量}]$ 正好是"在策略 $\pi$ 下关于动作 $a$ 的期望"。同样，$\sum_s d^\pi(s) \cdot [\cdots]$ 是关于状态的期望。所以整个双重求和可以简写为：
>
> $$\nabla_\theta J = \mathbb{E}_\pi \left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot Q^\pi(s, a)\right]$$
>
> 这就是策略梯度定理最常用的形式。
>
> ---
>
> **最终含义：怎么用它更新参数？**
>
> 我们要**最大化** $J$（平均回报越大越好），所以沿梯度**正方向**更新：
>
> $$\theta \leftarrow \theta + \alpha \cdot \nabla_\theta \log \pi_\theta(a \mid s) \cdot Q^\pi(s, a)$$
>
> 注意这里是**加号**（梯度上升），因为目标是最大化。回顾梯度节讲过的："如果目标是最大化某个目标函数，那就沿 $+\nabla$ 更新。"
>
> 直觉解读：
>
> - 如果 $Q > 0$（动作好），梯度更新会让 $\log \pi(a|s)$ 增大 → 选这个动作的概率变大
> - 如果 $Q < 0$（动作差），梯度更新会让 $\log \pi(a|s)$ 减小 → 选这个动作的概率变小
>
> **公式①已完全解锁。**
>
> </details>

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

> **回头看公式②③（完全解锁——你已掌握 Taylor 展开）**
>
> <details>
> <summary>点击展开：PPO Taylor 分析与 GRPO 归一化的完整拆解</summary>
>
> ---
>
> **公式②：PPO 裁剪的 Taylor 分析**
>
> 开头给出的公式②是：
>
> $$r_t(\theta) \approx \underbrace{1}_{\text{零阶}} + \underbrace{\nabla_\theta r_t \cdot (\theta - \theta_{old})}_{\text{一阶：线性变化}} + \underbrace{\frac{1}{2}(\theta - \theta_{old})^\top \nabla^2 r_t (\theta - \theta_{old})}_{\text{二阶：曲率}}$$
>
> 其中 $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$ 是概率比。
>
> 现在我们逐项拆解，每一项都对应刚学的 Taylor 展开知识：
>
> **零阶项 = 1**：为什么？因为展开点选在 $\theta = \theta_{old}$，此时新旧策略完全相同，$\pi_\theta = \pi_{\theta_{old}}$，所以 $r_t = \pi/\pi = 1$。这对应 Taylor 展开中的 $f(\mathbf{x})$——函数在展开点的值。
>
> **一阶项 $\nabla_\theta r_t \cdot (\theta - \theta_{old})$**：这对应刚学的多变量一阶 Taylor 展开中的 $\nabla f(\mathbf{x})^\top \boldsymbol{h}$。这里的 $f$ 就是 $r_t$，$\mathbf{x}$ 就是 $\theta_{old}$，$\boldsymbol{h}$ 就是 $(\theta - \theta_{old})$（参数的偏移量）。含义：参数偏移一小步，概率比大约线性变化这么多。
>
> **二阶项 $\frac{1}{2}(\theta - \theta_{old})^\top \nabla^2 r_t (\theta - \theta_{old})$**：这对应二阶 Taylor 展开中的 $\frac{1}{2}\boldsymbol{h}^\top \mathbf{H} \boldsymbol{h}$。$\nabla^2 r_t$ 就是 Hessian 矩阵（二阶偏导数组成的矩阵）。含义：线性近似不够准确时，用曲率来修正——"拐弯的程度"。
>
> **PPO 裁剪怎么用这个？** PPO 的裁剪操作 $\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)$ 把 $r_t$ 限制在 $[1-\varepsilon, 1+\varepsilon]$ 范围内。当参数更新太剧烈（一阶项或二阶项过大导致 $r_t$ 超出范围）时直接截断。效果上等价于：**限制了每步更新的二阶变化量，不让策略变化太快。**
>
> **公式②已完全解锁。**
>
> ---
>
> **公式③：GRPO 的组归一化**
>
> 开头给出的公式③是：
>
> $$\hat{A}_i = \frac{r_i - \mu}{\sigma}$$
>
> 这个公式不依赖 Taylor 展开等高级概念，我们已经拆解清楚：
>
> - **减均值** $r_i - \mu$：中心化，大于 0 表示比平均好
> - **除标准差** $/\sigma$：缩放到统一尺度，结果表示"偏离多少个标准差"（z-score）
> - **为什么需要**：不同问题的原始奖励尺度不同，标准化后才能公平比较
>
> **公式③也已解锁。**
>
> ---
>
> 至此，开头的三个公式你都已经能够完全理解了。
>
> </details>

## 公式速查

| 概念        | 公式                                                                                | 应用场景                |
| ----------- | ----------------------------------------------------------------------------------- | ----------------------- |
| 导数        | $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$                                    | 优化基础                |
| 链式法则    | $\frac{d}{dx}f(g(x)) = f'(g) \cdot g'(x)$                                           | 反向传播、策略梯度计算  |
| 梯度        | $\nabla f = [\partial f/\partial x_1, \ldots, \partial f/\partial x_n]^\top$        | 参数更新方向            |
| 梯度下降    | $\theta \leftarrow \theta - \alpha \nabla \mathcal{L}$                              | RL 优化的基本形式       |
| Adam        | $\theta \leftarrow \theta - \alpha \hat{m}/(\sqrt{\hat{v}} + \epsilon)$             | RL 最常用的优化器       |
| 一阶 Taylor | $f(\mathbf{x}+\boldsymbol{h}) \approx f(\mathbf{x}) + \nabla f^\top \boldsymbol{h}$ | PPO/GRPO 的局部近似分析 |
