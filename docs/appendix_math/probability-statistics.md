# E.2 概率与统计

> 相关章节：[第3章 MDP 形式化](/chapter03_mdp/formalism)、[第5章 策略梯度](/chapter05_policy_gradient/policy-gradient)、[第6章 GAE](/chapter06_ppo/gae-reward-model)

概率论是强化学习的理论基础。MDP 在形式上是一个概率框架：状态转移由分布 $P(s' \mid s, a)$ 描述，策略输出的是动作的概率分布 $\pi(a \mid s)$，目标函数是回报的期望。策略梯度定理、GAE、重要性采样——这些核心算法全是概率推导的产物。

本节从条件概率出发，逐步建立期望、方差、常用分布和重要性采样等概念。在开始之前，先看看本书正文中用到了哪些概率公式——你现在不需要完全理解它们，每学完一个概念，我们就回头解锁一个。

## 本书中你将遇到的概率公式

以下是本书正文中依赖概率论的四个核心公式。如果你现在看不懂，不用担心——下面每一节都会帮你解锁其中的一部分。

**公式① 价值函数**（[第3章 MDP](/chapter03_mdp/formalism)）

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s\right]$$

涉及概念：**条件概率**、**期望**。$\mathbb{E}_\pi$ 表示在策略 $\pi$ 下对所有轨迹取期望，$s_0 = s$ 是条件概率。

**公式② 策略梯度定理**（[第5章 策略梯度](/chapter05_policy_gradient/policy-gradient)）

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot G_t\right]$$

涉及概念：**期望**、**方差**。梯度是期望的形式，而 $G_t$ 的高方差是引入 baseline 的动机。

**公式③ GAE 广义优势估计**（[第6章 GAE](/chapter06_ppo/gae-reward-model)）

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{k=0}^{T-t-1} (\gamma\lambda)^k \delta_{t+k}$$

涉及概念：**方差与偏差的权衡**。$\lambda$ 在高方差的蒙特卡洛估计和高偏差的 TD 估计之间插值。

**公式④ PPO 裁剪目标**（[第6章 PPO](/chapter06_ppo/ppo-math)）

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t\right)\right]$$

涉及概念：**重要性采样**。概率比 $r_t(\theta) = \pi_\theta / \pi_{\theta_{old}}$ 就是重要性权重，裁剪防止权重出现极端值。

---

接下来，我们从最基本的条件概率开始。

## 条件概率

考虑一个简单的概率实验。掷两枚骰子，样本空间包含 $36$ 个等概率的结果。已知第一枚骰子是 $3$，那么两枚之和为 $7$ 的概率是多少？

在没有额外信息时，和为 $7$ 的概率是 $6/36 = 1/6$。但已知第一枚是 $3$ 之后，我们只需要第二枚是 $4$，概率变成了 $1/6$。这个例子说明：额外信息改变了我们对事件发生概率的判断。

**定义（条件概率）.** 设 $P(B) > 0$，事件 $A$ 在给定事件 $B$ 下的条件概率定义为

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

其中 $A \cap B$ 是事件 $A$ 和事件 $B$ 同时发生的事件，$P(A \cap B)$ 是两者同时发生的概率，分母 $P(B)$ 将样本空间从"所有可能结果"缩小到"使得 $B$ 成立的子集"。

![条件概率的韦恩图表示：$P(A \mid B)$ 是交集占 $B$ 的比例](./images/conditional-prob.svg)

在 MDP 中，状态转移概率 $P(s' \mid s, a)$ 表示"在当前状态为 $s$ 且动作为 $a$ 的条件下，下一状态为 $s'$ 的概率"。策略 $\pi(a \mid s)$ 表示"在状态为 $s$ 的条件下选择动作 $a$ 的概率"。整个 MDP 的轨迹可以看作条件概率的链式乘积（[第3章](/chapter03_mdp/formalism)）：

$$P(\tau \mid \pi) = P(s_0) \prod_{t=0}^{T} \pi(a_t \mid s_t) \cdot P(s_{t+1} \mid s_t, a_t)$$

三个因子都是条件概率。策略梯度定理的推导正是从这个链式展开出发的。

> **回头看公式①.** 现在你知道了条件概率，再来看开头的价值函数：
>
> $$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s\right]$$
>
> 竖线 $s_0 = s$ 就是条件概率——"在起始状态为 $s$ 的条件下"。而 MDP 轨迹 $P(\tau \mid \pi) = P(s_0) \prod_t \pi(a_t \mid s_t) \cdot P(s_{t+1} \mid s_t, a_t)$ 本身就是条件概率的链式乘积。
>
> 还不能理解的是 $\mathbb{E}_\pi[\cdot]$ 的含义——这需要期望的概念，我们继续往下学。

## 贝叶斯定理

**定理（贝叶斯）.** 设 $P(B) > 0$，则

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

贝叶斯定理建立了正向概率 $P(B \mid A)$ 与逆向概率 $P(A \mid B)$ 之间的桥梁。各部分的名称如下：

- $P(A)$ 称为**先验**（prior），表示在观测到 $B$ 之前对 $A$ 的信念
- $P(B \mid A)$ 称为**似然**（likelihood），表示若 $A$ 为真时观测到 $B$ 的概率
- $P(A \mid B)$ 称为**后验**（posterior），表示观测到 $B$ 之后对 $A$ 的更新信念

在强化学习中，贝叶斯强化学习利用贝叶斯定理维护对环境参数（转移概率、奖励函数）的后验分布。偏好学习中的 Bradley-Terry 模型（[第7章](/chapter07_alignment/dpo-math)）同样基于贝叶斯思想：通过观测到的偏好 $(y_w \succ y_l)$ 推断奖励函数的后验。

## 随机变量与期望

随机变量是一个取值具有不确定性的量，其取值由某个概率分布决定。掷骰子的结果、股票的收盘价、RL 智能体在某状态获得的回报——都是随机变量的实例。

**定义（期望）.** 离散随机变量 $X$ 的期望（均值）定义为

$$\mathbb{E}[X] = \sum_x x \cdot P(x)$$

其中求和遍历 $X$ 的所有可能取值，每个值 $x$ 乘以其概率 $P(x)$ 后累加。对于连续随机变量，求和替换为积分：

$$\mathbb{E}[X] = \int x \cdot p(x) \, dx$$

期望具有**线性性**，这是策略梯度推导中反复使用的性质：

$$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$$

期望号可以"分配"到各个加项上——这一性质使得我们可以将 RL 目标函数中的复杂期望拆解为简单部分的组合。

> **回头看公式①②.** 现在你知道了期望，再来看开头两个公式中的 $\mathbb{E}_\pi[\cdot]$：
>
> - 公式① $V^\pi(s) = \mathbb{E}_\pi[\sum \gamma^t r_t \mid s_0=s]$：$\mathbb{E}_\pi$ 表示"在策略 $\pi$ 产生的所有可能轨迹上取平均"。从状态 $s$ 出发，跑无数条轨迹，把每条的折扣奖励加起来，再取平均值——这就是"状态 $s$ 有多好"的数学定义。
> - 公式② $\nabla_\theta J = \mathbb{E}_\pi[\nabla \log \pi \cdot G_t]$：同样是期望——在所有 $(s, a)$ 对上取平均，$\nabla \log \pi$ 是方向，$G_t$ 是大小。
>
> 至此公式①已经完全解锁。公式②还差一个问题：$G_t$ 的**方差**太大，导致梯度估计不稳定——这正是下一节要讲的。

## 方差与标准差

**定义（方差）.** 随机变量 $X$ 的方差定义为

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

标准差是方差的平方根 $\sigma = \sqrt{\text{Var}(X)}$。

方差度量随机变量的"波动程度"。两个分布可以有相同的均值，但方差截然不同——方差大的分布散布得更广，结果更不稳定。

![两个分布均值相同但方差不同：窄而高的分布方差小，宽而矮的分布方差大](./images/variance-comparison.svg)

在 RL 中，策略梯度方法面临的核心困难之一就是梯度估计的方差过高。$\nabla_\theta J = \mathbb{E}[\nabla \log \pi(a \mid s) \cdot G_t]$ 中的 $G_t$ 是多步累积回报，其方差通常很大。引入 baseline $b(s)$（通常取 $V(s)$）可以在不改变期望值的前提下降低方差：

$$\nabla_\theta J = \mathbb{E}_\pi\left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot (G_t - b(s))\right]$$

关键性质：$\mathbb{E}[\nabla \log \pi(a \mid s) \cdot b(s)] = 0$，因此减去 baseline 不改变梯度的期望值。但由于 $G_t - V(s)$ 的绝对值通常远小于 $G_t$，方差显著降低。这正是从 REINFORCE 到 Actor-Critic 演化的关键动机。

> **回头看公式②③.** 现在你知道了方差和偏差-方差权衡，再来看开头的两个公式：
>
> - 公式②中，$G_t$ 是多步累积回报，方差极大。减去 baseline $V(s)$ 后得到 $G_t - V(s)$（即优势函数），期望不变但方差显著降低——这就是 Actor-Critic 的动机。
> - 公式③ GAE 进一步解决了同一个问题：$\hat{A}_t = \sum (\gamma\lambda)^k \delta_{t+k}$ 用指数衰减的权重组合多步 TD error。$\lambda = 0$ 时只有一步（低方差高偏差），$\lambda = 1$ 时退化为蒙特卡洛（高方差低偏差）。$\lambda$ 在两者之间插值——这正是"偏差与方差的权衡"。
>
> 公式②③已完全解锁。还剩公式④涉及重要性采样，我们继续往下学。

## 常用概率分布

### 伯努利分布

最简单的离散分布，仅有两种可能的取值：

$$P(X = 1) = p, \quad P(X = 0) = 1 - p$$

期望 $\mathbb{E}[X] = p$，方差 $\text{Var}(X) = p(1-p)$。二值奖励（任务成功或失败）服从伯努利分布。

### 分类分布（Categorical Distribution）

伯努利分布的 $K$ 维推广：

$$P(X = k) = p_k, \quad \sum_{k=1}^{K} p_k = 1$$

分类分布是**离散动作空间 RL 中最基本的分布**。策略网络 $\pi_\theta(a \mid s)$ 输出的即为分类分布——每个动作对应一个概率 $p_k$。

### 高斯分布（正态分布）

$$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

高斯分布由均值 $\mu$（中心位置）和方差 $\sigma^2$（散布程度）两个参数确定，呈钟形曲线。改变 $\mu$ 平移曲线，改变 $\sigma$ 改变宽窄。

![不同参数的高斯分布](./images/gaussian.svg)

高斯分布是**连续动作空间 RL 中最基本的分布**。连续策略通常参数化为 $\pi_\theta(a \mid s) = \mathcal{N}(a \mid \mu_\theta(s), \sigma_\theta(s)^2)$，即网络输出均值和标准差，动作从该高斯分布中采样。SAC、PPO 在连续控制任务中均采用高斯策略。

多元高斯分布的形式为

$$\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{n/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

其中 $\boldsymbol{\Sigma}$ 为协方差矩阵。实际应用中最常见的简化是取对角协方差矩阵（假设各维度独立）。

> **回头看：策略使用了哪些分布？** 本书中的策略 $\pi(a \mid s)$ 本质上就是一个概率分布：
>
> - 离散动作空间（Atari 游戏、文本生成中的 token 选择）→ **分类分布**：网络输出每个动作的概率 $p_k$
> - 连续动作空间（机器人控制）→ **高斯分布**：网络输出均值 $\mu$ 和标准差 $\sigma$，动作从 $\mathcal{N}(\mu, \sigma^2)$ 中采样

## 大数定律与蒙特卡洛估计

**大数定律**表明，当样本量 $N \to \infty$ 时，样本均值收敛于总体期望：

$$\frac{1}{N}\sum_{i=1}^{N} x_i \xrightarrow{N \to \infty} \mathbb{E}[X]$$

大数定律是蒙特卡洛方法的数学基础。在 RL 中，蒙特卡洛回报估计 $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$ 即是对价值函数的蒙特卡洛估计。若从状态 $s$ 出发跑 $N$ 条完整轨迹，则

$$V(s) \approx \frac{1}{N}\sum_{i=1}^{N} G_t^{(i)}$$

轨迹数越多，估计越准确。但单条轨迹的方差往往很高——这正是 TD 学习和 GAE 引入偏差以换取方差降低的动机。

## 重要性采样

重要性采样是一种利用来自分布 $Q$ 的样本来估计分布 $P$ 下期望值的技术。

**定理（重要性采样）.** 设 $P$ 和 $Q$ 为同一空间上的两个概率分布，且当 $P(x) > 0$ 时 $Q(x) > 0$，则

$$\mathbb{E}_{x \sim P}[f(x)] = \mathbb{E}_{x \sim Q}\left[f(x) \cdot \frac{P(x)}{Q(x)}\right]$$

其中 $\frac{P(x)}{Q(x)}$ 称为**重要性权重**。我们希望估计 $P$ 分布下的期望，但手中仅有 $Q$ 分布下收集的样本。对每个样本乘以重要性权重——若 $P$ 比 $Q$ 更倾向选择该样本，权重大于 1；反之权重小于 1。

需要注意的是，当 $P$ 和 $Q$ 差异较大时，重要性权重可能出现极端值，导致估计方差急剧增大。PPO 的裁剪机制 $\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)$ 正是为了缓解这一问题。

> **回头看公式④.** 现在你知道了重要性采样，再来看开头的 PPO 裁剪目标：
>
> $$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t\right)\right]$$
>
> 概率比 $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{old}}(a_t \mid s_t)$ 就是重要性权重 $P(x)/Q(x)$。PPO 用旧策略 $\pi_{\theta_{old}}$ 收集数据，但要用新策略 $\pi_\theta$ 的梯度来更新——这正是重要性采样的场景。裁剪 $\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)$ 防止权重过大导致方差爆炸。
>
> 至此，开头的四个公式你都已经能够理解了。

## 协方差与相关系数

协方差度量两个随机变量之间的线性关系：

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]$$

将其标准化后得到相关系数 $\rho \in [-1, 1]$：

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

## 公式速查

| 概念       | 公式                                                   | 应用场景                    |
| ---------- | ------------------------------------------------------ | --------------------------- |
| 条件概率   | $P(A \mid B) = P(A \cap B) / P(B)$                     | MDP 状态转移、策略分布      |
| 贝叶斯定理 | $P(A \mid B) = P(B \mid A) P(A) / P(B)$                | 贝叶斯 RL、偏好学习         |
| 期望       | $\mathbb{E}[X] = \sum x \cdot P(x)$                    | 价值函数、策略梯度          |
| 方差       | $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ | baseline 的动机、GAE       |
| 分类分布   | $P(X=k) = p_k$                                         | 离散动作策略                |
| 高斯分布   | $\mathcal{N}(\mu, \sigma^2)$                           | 连续动作策略                |
| 重要性采样 | $\mathbb{E}_P[f] = \mathbb{E}_Q[f \cdot P/Q]$          | PPO/GRPO 的 off-policy 修正 |
