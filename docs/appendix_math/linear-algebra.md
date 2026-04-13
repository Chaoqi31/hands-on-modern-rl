# E.1 线性代数

> 相关章节：[第3章 MDP 形式化](/chapter03_mdp/formalism)、[第4章 DQN](/chapter04_dqn/from-q-to-dqn)、[第9章 连续控制](/chapter09_continuous_control/intro)

线性代数是强化学习的数学语言。RL 中的状态用向量表示，策略参数存储在矩阵中，价值函数的收敛性与特征值有关。理解这些对象的几何意义，能让我们在面对"为什么这个网络架构有效""为什么梯度会爆炸"等问题时，有直观的判断依据。

本节从向量的几何解释出发，逐步建立矩阵运算、特征分解和范数的概念体系。在开始之前，先看看本书正文中用到了哪些线性代数公式——你现在不需要完全理解它们，每学完一个概念，我们就回头解锁一个。

## 本书中你将遇到的线性代数公式

以下是本书正文中依赖线性代数的四个核心公式。如果你现在看不懂，不用担心——下面每一节都会帮你解锁其中的一部分。

**公式① Q 值的线性近似**（[第4章 DQN](/chapter04_dqn/from-q-to-dqn)）

$$Q(s, a) = \mathbf{w}^\top \phi(s, a)$$

涉及概念：**向量**、**点积**。$\mathbf{w}$ 是权重向量，$\phi(s,a)$ 是特征向量，两者做点积得到 Q 值。

**公式② 策略梯度的方向**（[第5章 策略梯度](/chapter05_policy_gradient/policy-gradient)）

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot G_t\right]$$

涉及概念：**向量作为方向**。$\nabla_\theta J$ 是一个向量，指向策略性能增长最快的方向。

**公式③ Bellman 算子的谱半径**（[第3章 MDP](/chapter03_mdp/formalism)）

$$\mathcal{T}\mathbf{V} = \mathbf{R} + \gamma \mathbf{P}\mathbf{V}$$

涉及概念：**矩阵乘法**、**特征值**。$\mathbf{P}$ 是状态转移矩阵，谱半径恰好等于折扣因子 $\gamma$，保证了价值迭代的收敛。

**公式④ TRPO 的信任域约束**（[第6章 PPO](/chapter06_ppo/ppo-math)）

$$(\theta - \theta_{old})^\top \mathbf{F}(\theta - \theta_{old}) \le \delta$$

涉及概念：**范数**、**Hessian 矩阵**。$\mathbf{F}$ 是 Fisher 信息矩阵，在参数空间中定义了一个椭球形的"安全区域"。

---

接下来，我们从最基本的向量概念开始。

## 向量的几何解释

一个向量 $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$ 可以写成列向量

$$
\mathbf{x} = \begin{bmatrix}x_1\\x_2\\\vdots\\x_n\end{bmatrix},
$$

或行向量 $\mathbf{x}^\top = [x_1, x_2, \ldots, x_n]$。在 RL 中，状态通常表示为列向量，而策略参数中的权重矩阵按行存储。

对向量的理解有两种等价的几何视角，二者可以随时切换。

![向量既可以理解为空间中的点，也可以理解为空间中的方向](./images/vec-point-direction.svg)

**作为空间中的点。** 在二维情形下，$[3, 2]^\top$ 表示平面上横坐标为 3、纵坐标为 2 的点。在 RL 中，状态向量 $\mathbf{s} = [s_1, s_2, \ldots, s_n]^\top$ 就是状态空间中的一个点——每个分量编码了环境的一个特征（位置、速度、角度等）。这种视角使我们能够讨论"两个状态之间的距离"。

这种几何视角也让我们能够更抽象地思考问题。与其面对"如何让智能体学会玩 Atari 游戏"这样具体的问题，不如将问题抽象为"如何在状态空间中找到一条从初始点到高回报区域的路径"。

**作为空间中的方向。** 同一个 $[3, 2]^\top$ 也可以理解为"向右走 3 步、向上走 2 步"这一方向。这一视角在理解梯度时尤为关键。策略梯度 $\nabla_\theta J$ 本身就是一个方向向量——它指示参数应在参数空间中朝哪个方向移动才能提升期望回报。同方向的任意长度的向量代表同一个方向，只是走的远近不同。

## 向量加法与标量乘法

向量加法 $\mathbf{u} + \mathbf{v}$ 的几何含义遵循平行四边形法则：先沿 $\mathbf{u}$ 的方向移动，再沿 $\mathbf{v}$ 的方向移动，最终到达的位置即为 $\mathbf{u} + \mathbf{v}$。

![向量加法的平行四边形法则](./images/vec-addition.svg)

向量减法 $\mathbf{u} - \mathbf{v}$ 也有直观的几何含义：它表示从 $\mathbf{v}$ 的终点指向 $\mathbf{u}$ 的终点的方向。由恒等式 $\mathbf{u} = \mathbf{v} + (\mathbf{u}-\mathbf{v})$ 可以看出，$\mathbf{u}-\mathbf{v}$ 正是从 $\mathbf{v}$ 走到 $\mathbf{u}$ 需要的方向。

在理解残差连接（ResNet 中的 skip connection）时，向量加法很直观——网络学习的增量 $\Delta \mathbf{y}$ 被叠加到原始输入 $\mathbf{x}$ 上：$\mathbf{y} = \mathbf{x} + \Delta \mathbf{y}$。这使得梯度可以"跳过"网络层直接回传。

标量乘法 $c \cdot \mathbf{v}$ 的几何含义是：将向量 $\mathbf{v}$ 沿其方向拉伸（$c > 1$）或压缩（$0 < c < 1$），$c < 0$ 时方向反转。对于任意向量 $\mathbf{v}$，$\mathbf{v}$ 与 $c \cdot \mathbf{v}$ 的夹角始终为零——缩放不改变方向。在 SAC 算法中，熵系数 $\alpha$ 的作用正是对探索方向的力度进行标量缩放。

## 点积与夹角

给定两个列向量 $\mathbf{u}$ 和 $\mathbf{v}$，它们的点积定义为

$$\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^\top \mathbf{v} = \sum_{i=1}^{n} u_i v_i$$

由于求和顺序不影响结果，点积满足交换律：

$$
\mathbf{u}\cdot\mathbf{v} = \mathbf{u}^\top\mathbf{v} = \mathbf{v}^\top\mathbf{u}
$$

点积的几何含义是什么？它与两个向量之间的夹角 $\theta$ 有密切关系。考虑两个特定向量 $\mathbf{v} = (r, 0)$ 和 $\mathbf{w} = (s\cos\theta, s\sin\theta)$，即 $\mathbf{v}$ 平行于 $x$ 轴、长度为 $r$，$\mathbf{w}$ 与 $x$ 轴夹角为 $\theta$、长度为 $s$。计算它们的点积：

$$
\mathbf{v}\cdot\mathbf{w} = r \cdot s\cos\theta = rs\cos\theta = \|\mathbf{v}\|\|\mathbf{w}\|\cos\theta
$$

这一关系在一般情形下也成立。对任意两个向量 $\mathbf{u}$ 和 $\mathbf{v}$，夹角 $\theta$ 由下式给出：

$$\theta = \arccos\left(\frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}\right)$$

其中 $\|\mathbf{u}\| = \sqrt{\sum u_i^2}$ 是 L2 范数（向量长度）。注意这个公式没有限制维度——在三维或三千维空间中同样适用。

![点积与夹角的关系](./images/vec-angle.svg)

由此可得三个重要的特殊情况：

- **方向相同**（$\theta = 0$）：点积取最大值 $\|\mathbf{u}\| \|\mathbf{v}\|$，$\cos\theta = 1$
- **相互正交**（$\theta = 90°$）：$\cos\theta = 0$，因此点积为零
- **方向相反**（$\theta = 180°$）：点积取最小值 $-\|\mathbf{u}\| \|\mathbf{v}\|$

### 正交性

两个向量正交（$\mathbf{u}\cdot\mathbf{v} = 0$）意味着它们"完全不相关"——一个向量在另一个向量方向上的投影为零。

![两个向量正交：$\mathbf{u} \cdot \mathbf{v} = 0$](./images/orthogonal-vectors.svg)

一个合理的疑问是：计算角度为什么有用？答案在于角度具有**尺度不变性**。考虑一幅图像和它亮度降为 10% 的版本——像素值相差很大，但内容完全相同。角度 $\theta$ 不受向量缩放影响：对任意 $\mathbf{v}$，$\mathbf{v}$ 与 $0.1 \cdot \mathbf{v}$ 的夹角始终为零。

### 余弦相似度

在 ML 中，用角度衡量两个向量的接近程度时，通常只用 $\cos\theta$ 部分，称为**余弦相似度**：

$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \cdot \|\mathbf{v}\|}$$

取值为 $[-1, 1]$：$1$ 表示方向完全相同，$-1$ 表示方向完全相反，$0$ 表示正交。

> **回头看公式①②.** 现在你知道了向量、点积和方向的含义，再来看开头的两个公式：
>
> - 公式① $Q(s, a) = \mathbf{w}^\top \phi(s, a)$ 中的 $\mathbf{w}^\top \phi(s,a)$ 就是两个向量的**点积**。$\mathbf{w}$ 是"重要方向"，$\phi(s,a)$ 是状态-动作特征。点积越大，说明该特征与重要方向越一致，Q 值越高。
> - 公式② $\nabla_\theta J(\theta)$ 中的 $\nabla_\theta$ 表示对参数 $\theta$ 的梯度——一个**向量**，方向指向策略性能增长最快的方向。$G_t$ 是标量，决定沿这个方向走多远。
>
> 还剩公式③④涉及矩阵、特征值和范数，我们继续往下学。

## 矩阵乘法与线性变换

矩阵与向量的乘法 $\mathbf{y} = \mathbf{A}\mathbf{x}$ 的几何含义是对向量 $\mathbf{x}$ 施加一次线性变换。一种直观的理解方式是将矩阵的每一列视为基向量变换后的位置。

对于 $2 \times 2$ 矩阵 $\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$，第一列 $[a, c]^\top$ 是原 $[1, 0]^\top$ 变换后的位置，第二列 $[b, d]^\top$ 是原 $[0, 1]^\top$ 变换后的位置。

![矩阵 $\mathbf{A}$ 将正方形的基向量变换到新的位置](./images/matrix-transform.svg)

矩阵乘法对空间执行拉伸、旋转或剪切。在 RL 中，神经网络的每一层（忽略非线性激活函数）都是一次线性变换 $\mathbf{h} = \mathbf{W}\mathbf{x} + \mathbf{b}$。策略网络 $\pi_\theta$ 的参数 $\theta = \{\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2, \ldots\}$ 由一组矩阵和向量构成，定义了从状态空间到动作概率空间的映射。

## 特征值与特征向量

设 $\mathbf{A}$ 为 $n \times n$ 方阵。若存在非零向量 $\mathbf{v}$ 和标量 $\lambda$ 使得

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

则 $\mathbf{v}$ 称为 $\mathbf{A}$ 的**特征向量**（eigenvector），$\lambda$ 称为对应的**特征值**（eigenvalue）。

几何上，特征向量是矩阵变换中方向不变的向量——矩阵对它的唯一效应是缩放。若 $\lambda > 1$，该方向被拉伸；若 $|\lambda| < 1$，该方向被压缩。理解特征值的一个好处是：它揭示了矩阵在各个方向上的"放大倍数"。

在 RL 理论中，特征值有几个重要的应用：价值迭代的收敛性由 Bellman 算子的谱半径（最大特征值的绝对值）等于折扣因子 $\gamma < 1$ 保证；TRPO 的信任域约束中的 Fisher 信息矩阵是 KL 散度的 Hessian，其特征值反映参数空间各方向的曲率。

> **回头看公式③.** 现在你知道了矩阵乘法和特征值，再来看开头的公式③：
>
> $$\mathcal{T}\mathbf{V} = \mathbf{R} + \gamma \mathbf{P}\mathbf{V}$$
>
> Bellman 算子 $\mathcal{T}$ 是一个矩阵运算——把状态转移矩阵 $\mathbf{P}$ 乘到价值向量 $\mathbf{V}$ 上，再加奖励向量 $\mathbf{R}$。关键在于 $\mathbf{P}$ 的特征值：最大特征值的绝对值（谱半径）恰好等于 $\gamma < 1$，这意味着每次迭代都将误差压缩 $\gamma$ 倍——这就是价值迭代一定收敛的数学保证。

## 范数

范数度量向量或矩阵的"大小"。最常用的向量范数：

| 范数             | 定义                                       | 性质           |
| ---------------- | ------------------------------------------ | -------------- |
| L1 范数          | $\|\mathbf{x}\|_1 = \sum \|x_i\|$          | 促进稀疏性     |
| L2 范数          | $\|\mathbf{x}\|_2 = \sqrt{\sum x_i^2}$     | 对应欧氏距离   |
| L$\infty$ 范数   | $\|\mathbf{x}\|_\infty = \max_i \|x_i\|$   | 度量最大分量   |

矩阵的 Frobenius 范数 $\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} a_{ij}^2}$ 是 L2 范数在矩阵上的推广。

在 RL 中，梯度裁剪使用 L2 范数防止梯度爆炸：若 $\|\nabla_\theta \mathcal{L}\|_2 > c$，则将梯度缩放到 $c \cdot \nabla_\theta \mathcal{L} / \|\nabla_\theta \mathcal{L}\|_2$。权重衰减的惩罚项 $\lambda \|\theta\|_2^2$ 使用 L2 范数约束参数规模。

> **回头看公式④.** 现在你知道了范数和特征值，再来看开头的公式④：
>
> $$(\theta - \theta_{old})^\top \mathbf{F}(\theta - \theta_{old}) \le \delta$$
>
> 其中 $\mathbf{F}$ 是 Fisher 信息矩阵。这个不等式说的是：更新量 $(\theta - \theta_{old})$ 的"加权范数"不能超过 $\delta$。$\mathbf{F}$ 的特征值决定了权重的分配——特征值大的方向（曲率陡峭），步长被收紧；特征值小的方向（曲率平缓），步长可以放大。整个约束在参数空间中画出了一个椭球形的"安全区域"。
>
> 至此，开头的四个公式你都已经能够理解了。

## 公式速查

| 概念         | 公式                                                                | 应用场景             |
| ------------ | ------------------------------------------------------------------- | -------------------- |
| 点积         | $\mathbf{u} \cdot \mathbf{v} = \sum u_i v_i$                        | Q 值计算、相似度度量 |
| 余弦相似度   | $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$ | embedding 比较       |
| 矩阵-向量乘法 | $\mathbf{y} = \mathbf{A}\mathbf{x}$                                 | 神经网络层变换       |
| 特征值分解   | $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$                         | 收敛性分析、自然梯度 |
| L2 范数      | $\|\mathbf{x}\|_2 = \sqrt{\sum x_i^2}$                              | 梯度裁剪、权重衰减   |
| Frobenius 范数 | $\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} a_{ij}^2}$                     | 参数正则化           |
