# 3.6 从价值到策略

## 本节导读

**核心内容**

- **Value-Based 的局限**：$\arg\max_a Q(s,a)$ 在连续/高维动作空间无法直接求解。
- **参数化策略**：用 $\pi_\theta(a \mid s)$ 直接建模行为分布，跳过 $Q$ 表。
- **策略目标与梯度**：目标函数 $J(\theta)$ 评价策略平均回报，策略梯度定理给出优化方向。
- **REINFORCE 与方差**：用轨迹回报直接更新策略，但波动大，需要基线稳定。

上一节学习的是**基于价值的强化学习**（Value-Based RL）：给每个”状态-动作对”存储一个价值估计 $Q(s,a)$，策略通过 $\arg\max_a Q(s,a)$ 导出。这个路线在动作空间有限时工作得很好——CartPole 只有左、右两个动作，比较 $Q$ 值就能选出最优。

但它的核心操作 $\arg\max_a Q(s,a)$ 隐含一个前提：**动作必须能够枚举。**

一旦动作空间连续或过大，这个前提就失效了。机械臂要给 6 个关节同时输出力矩：

$$
a=(\tau_1,\tau_2,\ldots,\tau_6)\in\mathbb{R}^6.
$$

你当然可以训练一个 $Q(s,a)$ 网络来估计每个动作的分数，但真正的困难在于下一步：怎样在无穷多个 $a$ 中求解

$$
\arg\max_a Q(s,a)?
$$

自动驾驶的油门、方向盘角度连续变化；大语言模型的词表有几万项。在这些场景下，”先打分再取最大”的计算代价和搜索难度，使得价值路线不再直接可行。

那么，如果最终需要的是一个会行动的智能体，为什么一定要先学一张打分表？

**核心思路是：直接对策略本身建模。** 不通过价值函数间接推导行为，而是让策略网络直接输出动作分布或连续动作，并用环境反馈的回报来优化策略参数。这就是**基于策略的强化学习**（Policy-Based RL）。

::: info 核心概念
强化学习常见有两条路线。

**Value-Based** 方法先学习动作价值 $Q(s,a)$，再通过 $\arg\max_a Q(s,a)$ 选出当前最好的动作。Q-Learning 和 DQN 属于这条路线。它适合动作数量不多、可以逐个比较的任务，比如 CartPole 只有“向左推”和“向右推”两个动作；它也更容易复用旧经验，常见做法是把历史数据放进 replay buffer 反复学习。

**Policy-Based** 方法不先学动作打分表，而是直接学习策略 $\pi_\theta(a\mid s)$：在状态 $s$ 下，以多大概率选择动作 $a$。REINFORCE 属于这条路线。它优化的是整套策略的平均回报

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G_0].
$$

直觉上，高回报轨迹里的动作会被提高概率，低回报轨迹里的动作会被降低概率。因为它直接输出动作分布或连续动作，所以更适合机械臂、自动驾驶、LLM 生成这类动作空间很大或连续的任务。不过，像 REINFORCE 这样的纯策略梯度方法通常依赖当前策略采样的新数据，也就是更偏 **On-policy**[^1][^5][^6]。
:::

## 从动作打分到行为分布

在上一节中，我们用**动作价值函数** $Q(s,a)$ 描述状态 $s$ 下动作 $a$ 的长期价值。假设 CartPole 的当前状态为 $s$，可选动作只有”向左推”和”向右推”。**基于价值的方法**首先估计 $Q(s,\text{左})$ 和 $Q(s,\text{右})$，然后选择价值较大的动作。

现在我们考虑另一种表示方式。与其先为每个动作估计价值、再从中选出价值最高的动作，不如直接描述智能体在状态 $s$ 下的**行为分布**。前者的核心问题是"哪个动作分数最高"；后者的核心问题是"该以什么概率行动"。例如，**策略**可以规定：在状态 $s$ 下，以 $70\%$ 的概率向左推，以 $30\%$ 的概率向右推。如果动作数量更多，这个分布也可以写成 $70\%$ 向左、$20\%$ 向前、$10\%$ 向右。

这种行为分布在训练初期通常并不准确。智能体按照当前分布与环境交互，并根据得到的回报调整分布。如果某些动作经常出现在**高回报轨迹**中，它们在相应状态下的概率应当**增大**；如果某些动作经常出现在**低回报轨迹**中，它们的概率应当**减小**。

因此，**策略路线**的思路发生了转变：不再通过迭代更新一张 $Q$ 表来回答"哪个动作最好"，而是直接学习从状态到动作分布的映射，来回答"该怎么行动"。下面，我们将把这个分布写成可训练的函数。

## 参数化策略

为了让策略能够从数据中学习，我们需要给策略引入**参数**。原来的策略 $\pi(a\mid s)$ 表示在状态 $s$ 下选择动作 $a$ 的概率。加入参数后，我们将其写为

$$
\pi_\theta(a\mid s)=P_\theta(A_t=a\mid S_t=s).
$$

其中 $\theta$ 是策略的参数，通常对应**神经网络的权重**；$S_t$ 是时刻 $t$ 的状态，$A_t$ 是时刻 $t$ 选择的动作。于是，$\pi_\theta(a\mid s)$ 表示在当前参数 $\theta$ 下，智能体在状态 $s$ 中选择动作 $a$ 的概率。

对于**离散动作空间**，神经网络通常先为每个动作输出一个**偏好分** $z_\theta(s,a)$。偏好分本身不要求非负，也不要求和为 $1$。因此，我们需要将这些分数转换为合法的概率分布。常见做法是使用 **softmax** 函数：

$$
\pi_\theta(a\mid s)
=
\frac{\exp(z_\theta(s,a))}
{\sum_{a'}\exp(z_\theta(s,a'))}.
$$

分母对所有候选动作 $a'$ 求和，作用是对偏好分进行归一化。这样得到的每个概率都非负，并且所有动作的概率之和等于 $1$。在 CartPole 中，网络只需要输出“向左推”和“向右推”的两个偏好分，再由 softmax 得到两个动作概率。

对于**连续控制任务**，动作不能通过枚举得到。此时，策略常被表示为一个**连续分布**。例如，可以令动作从**高斯分布**中采样：

$$
a\sim\mathcal{N}\left(\mu_\theta(s),\sigma_\theta(s)^2\right).
$$

其中，网络根据状态 $s$ 输出均值 $\mu_\theta(s)$ 和标准差 $\sigma_\theta(s)$。均值表示当前最可能采用的动作，标准差控制采样的波动范围。这样，策略既可以表达主要的动作倾向，也可以在训练中保留随机探索。

由此可以看出，$Q(s,a)$ 和 $\pi_\theta(a\mid s)$ 描述的是两类不同对象：

$$
Q(s,a)\quad\text{和}\quad \pi_\theta(a\mid s).
$$

$Q(s,a)$ 估计的是在状态 $s$ 下采取动作 $a$ 的长期价值；$\pi_\theta(a\mid s)$ 给出的是在状态 $s$ 下选择动作 $a$ 的概率。前者用于评价动作，后者直接定义行为。

## 策略目标 J(θ)

有了参数化策略以后，下一步要问：这组参数 $\theta$ 好不好？

上一节用 $Q(s,a)$ 比较某个状态下的不同动作。现在视角再往外拉一层：我们要评价的不是某个动作，而是整套策略参数。CartPole 每次 reset 后，小车位置和杆子角度会从一个小范围随机初始化；语言模型每次也会遇到不同 prompt。一个策略好不好，应该看它从这些可能起点出发，平均能拿到多少长期回报。

如果初始状态来自分布 $\rho_0$，策略目标可以写成：

$$
J(\theta)
=
\mathbb{E}_{s_0\sim\rho_0}
\left[
V^{\pi_\theta}(s_0)
\right].
$$

这里 $s_0$ 是初始状态，$\rho_0$ 描述起点如何被采样；$V^{\pi_\theta}(s_0)$ 表示从 $s_0$ 出发，之后一直按照策略 $\pi_\theta$ 行动时的期望折扣回报。因此这行公式的意思是：对所有可能起点取平均，看看当前策略整体能拿多少分。

同一件事也可以用轨迹来写。令

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots)
$$

表示一条轨迹，$G_0$ 表示从初始时刻开始的折扣总回报：

$$
G_0=\sum_{t=0}^{\infty}\gamma^t r_t.
$$

其中 $r_t$ 是第 $t$ 步得到的奖励，$\gamma\in[0,1]$ 是折扣因子。于是

$$
J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}[G_0]
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{\infty}\gamma^t r_t
\right].
$$

$\tau\sim\pi_\theta$ 的意思是：用当前策略和环境交互，采样出一条轨迹。这个期望不是对一个固定数据集求平均，而是对当前策略可能产生的所有轨迹求平均。

所以 $J(\theta)$ 可以直读为：

$$
J(\theta)=\text{当前这套策略参数的平均长期回报。}
$$

学习目标也就变成

$$
\theta^*=\arg\max_\theta J(\theta).
$$

注意这里最大化的是参数 $\theta$，不是动作 $a$。上一节的

$$
\arg\max_a Q(s,a)
$$

是在某个状态里挑动作；这里的

$$
\arg\max_\theta J(\theta)
$$

是在所有策略参数里找一套平均表现最好的行为规则。强化学习在这里变成了一个直接的优化问题：调整策略网络参数，让平均回报上升。

## 轨迹概率

要优化 $J(\theta)$，自然会想到求梯度：

$$
\nabla_\theta J(\theta).
$$

但它和监督学习里的 loss 有一个重要区别。监督学习常常面对固定数据：图片已经在那里，标签也已经在那里。策略优化的数据不是固定的。参数 $\theta$ 一变，动作概率会变，智能体走到的状态会变，采样出来的轨迹也会变。

因此先把轨迹发生的概率写清楚。一条轨迹

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots)
$$

在参数 $\theta$ 下发生的概率可以写成

$$
P_\theta(\tau)
=
\rho_0(s_0)
\prod_t
\pi_\theta(a_t\mid s_t)
P(s_{t+1}\mid s_t,a_t).
$$

这行公式从左到右读就可以。$\rho_0(s_0)$ 是起点 $s_0$ 出现的概率；$\pi_\theta(a_t\mid s_t)$ 是策略在状态 $s_t$ 选择动作 $a_t$ 的概率；$P(s_{t+1}\mid s_t,a_t)$ 是环境在看到状态和动作后转移到下一状态的概率。

这里最关键的是：三类因素里，只有策略概率 $\pi_\theta(a_t\mid s_t)$ 直接含有 $\theta$。初始状态分布和环境转移规律是环境给的，不是策略网络的参数。

把目标函数写成对所有轨迹的求和，就是

$$
J(\theta)
=
\sum_\tau P_\theta(\tau)G(\tau).
$$

$G(\tau)$ 是轨迹 $\tau$ 的总回报。这个式子只是期望的展开：每条轨迹用“发生概率”乘以“这条轨迹的回报”，再把所有可能轨迹加起来。高概率轨迹和高回报轨迹都会更影响平均值。

## 对数求导技巧

现在对 $J(\theta)$ 求梯度：

$$
\nabla_\theta J(\theta)
=
\sum_\tau \nabla_\theta P_\theta(\tau)G(\tau).
$$

一旦轨迹 $\tau$ 已经确定，它里面的状态、动作、奖励也就确定了，所以 $G(\tau)$ 可以先看成这条轨迹上的常数。参数 $\theta$ 改变的是这条轨迹被采到的概率，而不是已经写下来的这串奖励本身。

上式还不方便采样估计，因为它包含的是 $\nabla_\theta P_\theta(\tau)$。我们希望把它改写成“概率乘某个量”的形式，也就是

$$
\sum_\tau P_\theta(\tau)(\cdots),
$$

这样就可以用当前策略采样几条轨迹，取平均来近似。关键工具是对数求导技巧：

$$
\nabla_\theta P_\theta(\tau)
=
P_\theta(\tau)\nabla_\theta\log P_\theta(\tau).
$$

它来自恒等式

$$
\nabla_\theta\log f(\theta)
=
\frac{\nabla_\theta f(\theta)}{f(\theta)}.
$$

把分母乘到左边，就得到 $\nabla_\theta f(\theta)=f(\theta)\nabla_\theta\log f(\theta)$。把 $f(\theta)$ 换成 $P_\theta(\tau)$，就得到上面的式子。

代回目标梯度：

$$
\nabla_\theta J(\theta)
=
\sum_\tau
P_\theta(\tau)
\nabla_\theta\log P_\theta(\tau)
G(\tau).
$$

这已经是期望形式：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\nabla_\theta\log P_\theta(\tau)G(\tau)
\right].
$$

这一步的含义是：不需要枚举所有轨迹。只要能按照当前策略采样轨迹，就能用样本平均估计策略梯度。

还可以继续化简 $\log P_\theta(\tau)$。对轨迹概率取对数，连乘变成连加：

$$
\log P_\theta(\tau)
=
\log\rho_0(s_0)
+
\sum_t\log\pi_\theta(a_t\mid s_t)
+
\sum_t\log P(s_{t+1}\mid s_t,a_t).
$$

对 $\theta$ 求梯度时，$\rho_0$ 和环境转移 $P$ 都不含策略参数，所以它们的梯度为 0。留下来的只有策略项：

$$
\nabla_\theta\log P_\theta(\tau)
=
\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

于是得到最基础的策略梯度形式：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
G(\tau)
\right].
$$

进一步地，第 $t$ 步动作不可能影响已经发生过的过去奖励，所以通常用从第 $t$ 步开始的后续回报 $G_t$ 来评价这一步动作：

$$
G_t=\sum_{k=t}^{\infty}\gamma^{k-t}r_k.
$$

代入后得到更常用的写法：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
G_t
\right].
$$

公式中的 $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ 给出一个方向：怎样调整参数，才能让“在状态 $s_t$ 下选中动作 $a_t$”这件事更容易发生。$G_t$ 决定这个方向应该被推多大。如果后续回报高，这个动作会被加强；如果后续回报低，更新方向会削弱它。

这就是策略梯度最朴素的读法：

> 实际做过一个动作。后来结果好，就提高以后再这样做的概率；后来结果差，就降低以后再这样做的概率。

## 一个两动作例子

为了看清公式不是一句口号，考虑只有一个状态和两个动作 A、B 的小问题。令策略用一个参数 $\theta$ 控制选择 A 的概率：

$$
p=\pi_\theta(A)=\sigma(\theta),\qquad
\pi_\theta(B)=1-p.
$$

$\sigma(\theta)$ 是 sigmoid 函数。这里不需要记它的完整形式，只要知道 $\theta$ 变大时，$p$ 也会变大，也就是 A 更容易被选中。

可以算出两个动作的对数概率梯度：

$$
\nabla_\theta\log\pi_\theta(A)=1-p,
$$

$$
\nabla_\theta\log\pi_\theta(B)=-p.
$$

假设当前 $p=0.7$。如果这次选了 A，并且后续回报是 $G=10$，更新方向与

$$
(1-p)G=0.3\times 10=3
$$

同号。梯度为正，$\theta$ 会增大，A 的概率 $p$ 也会增大。

如果这次选了 B，并且后续回报也是 $G=10$，更新方向与

$$
(-p)G=-0.7\times 10=-7
$$

同号。梯度为负，$\theta$ 会减小，A 的概率下降，因此 B 的概率 $1-p$ 上升。

所以策略梯度并不是粗暴地”所有动作一起加强”。它加强的是实际被选中并带来高回报的动作。如果高回报来自 B，概率就会从 A 挪向 B。如果回报为负，方向会反过来：选 A 后得到坏结果，就降低 A 的概率；选 B 后得到坏结果，就降低 B 的概率。

### CartPole 上的 REINFORCE

上面的两动作例子虽然简单，但完整展示了策略梯度的更新逻辑。把它搬到 CartPole-v1 上，就得到最经典的策略梯度算法 **REINFORCE**。下图是 Gymnasium CartPole-v1 的运行示例——一根杆立在小车上，智能体每步选择向左或向右推车，目标是让杆尽量不倒：

![CartPole-v1 运行示例（Gymnasium）](./images/cart-pole.gif)

<p align="center">
  <em>图源：<a href="https://gymnasium.farama.org/environments/classic_control/cart_pole/" target="_blank" rel="noopener noreferrer">Gymnasium Documentation - Cart Pole</a></em>
</p>

REINFORCE 的更新步骤：

1. 用当前策略 $\pi_\theta$ 在 CartPole 中跑完一局（episode），记录每步的状态 $s_t$、动作 $a_t$ 和奖励 $r_t$；
2. 计算每步的后续回报 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$；
3. 对每步更新参数：$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t | s_t) \, G_t$。

CartPole 的状态是 4 维向量（小车位置、速度、杆角度、角速度），动作只有”向左”和”向右”两个。策略网络只需一个小的 MLP：输入 4 维状态，输出 2 个偏好分，经 softmax 得到两个动作概率。

训练初期，策略接近随机，杆很快就倒了（平均存活 10-20 步）。随着更新迭代，策略逐步学会”杆向左倾斜就向左推，向右倾斜就向右推”，平均存活步数持续上升。Gymnasium 中 CartPole-v1 的通关标准是平均 500 步内杆不倒，REINFORCE 通常在几百个 episode 后可以达到。

不过 REINFORCE 也有明显的缺点：因为用完整轨迹的 $G_t$ 作更新信号，回报波动会导致训练曲线抖动很大。这正是下一节要讨论的基线（baseline）和 Actor-Critic 方法要解决的问题。关于 CartPole 环境的详细规则可参考 [Gymnasium 文档](https://gymnasium.farama.org/environments/classic_control/cart_pole/)。

### REINFORCE 代码实现

以下代码在 CartPole-v1 上实现 REINFORCE，不依赖外部 RL 库：

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 策略网络：状态 → 动作概率
class PolicyNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

# 训练循环
env = gym.make("CartPole-v1")
policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
gamma = 0.99

for episode in range(500):
    state, _ = env.reset()
    log_probs = []
    rewards = []

    # 采样一条完整轨迹
    done = False
    while not done:
        action, log_prob = policy.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state

    # 计算每步的后续回报 G_t
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 归一化

    # 策略梯度更新
    loss = -sum(lp * Gt for lp, Gt in zip(log_probs, returns))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (episode + 1) % 50 == 0:
        avg_len = sum(len(rewards) for _ in range(10))  # 简化
        print(f"Episode {episode+1}, length: {len(rewards)}")

env.close()
```

代码的核心逻辑在第 40-48 行：先反向遍历奖励序列，用递推 $G_t = r_t + \gamma G_{t+1}$ 算出每步的后续回报；然后对回报做标准化（减均值除标准差），这等价于一个简单的基线，能显著降低方差；最后用 $-\sum_t \log \pi_\theta(a_t|s_t) \cdot G_t$ 作为 loss 反向传播更新参数。负号是因为 PyTorch 默认做梯度下降，而策略梯度要做梯度上升。

### MountainCar：当奖励稀疏时

CartPole 每步都有奖励（$+1$），策略梯度很容易判断动作好坏。但很多任务的奖励信号很稀疏。Gymnasium 中的 **MountainCar-v0** 就是一个典型例子：

![MountainCar-v0 运行示例（Gymnasium）](./images/mountain-car.gif)

<p align="center">
  <em>图源：<a href="https://gymnasium.farama.org/environments/classic_control/mountain_car/" target="_blank" rel="noopener noreferrer">Gymnasium Documentation - Mountain Car</a></em>
</p>

一辆车停在两座山之间的谷底，目标是开到右侧山顶。车的引擎力不够直接冲上去，必须先向左后退再向右加速，借助动量攀顶。

| 规则     | 说明                                                       |
| -------- | ---------------------------------------------------------- |
| 状态     | 2 维：位置 $x \in [-1.2, 0.6]$，速度 $v \in [-0.07, 0.07]$ |
| 动作     | 3 个：向左推（0）、不推（1）、向右推（2）                  |
| 奖励     | 每步 $-1$，到达山顶 $x \geq 0.5$ 后 episode 结束           |
| 最大步数 | 200 步                                                     |

关键难点：**随机策略几乎不可能到达山顶**。200 步内随机推车，几乎每次都超时。这意味着 REINFORCE 采样的每条轨迹回报都差不多（都是 $-200$），策略梯度信号几乎为零——所有动作的 $G_t$ 相同，无法区分好坏。

这就是**稀疏奖励**问题。Q-Learning 在 MountainCar 上表现更好：因为 TD 更新不需要等轨迹结束，即使当前 episode 没到达山顶，Q 值仍然会从终点附近逐步反向传播。而 REINFORCE 必须等到 episode 结束才能拿到 $G_t$，如果所有 episode 都失败，就没有学习信号。

### 与 Q-Learning 的对比

现在可以把本节的策略梯度方法和上一节的 Q-Learning 放在一起比较：

| 比较维度             | **Q-Learning** (上一节)                                                      | **REINFORCE** (本节)                                                                       |
| :------------------- | :--------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------- |
| **学习对象**         | 动作价值 $Q(s,a)$                                                            | 策略参数 $\theta$                                                                          |
| **更新公式**         | $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ | $\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t$ |
| **更新时机**         | 每走一步                                                                     | 每完成一局                                                                                 |
| **数据来源**         | Off-policy: 可复用旧经验                                                     | On-policy: 只能用当前策略的数据                                                            |
| **适合动作空间**     | 离散、有限                                                                   | 离散或连续                                                                                 |
| **CartPole 表现**    | 收敛快，曲线平滑                                                             | 收敛较慢，曲线抖动                                                                         |
| **MountainCar 表现** | 稀疏奖励下仍能学习（TD 传播）                                                | 稀疏奖励下几乎无法学习                                                                     |
| **连续控制**         | 需在无穷动作中求 $\arg\max$，困难                                            | 直接输出连续动作，天然适合                                                                 |

两种方法不是替代关系，而是互补。Q-Learning 擅长离散动作下的高效学习，策略梯度擅长连续动作和需要随机策略的场景。后续的 Actor-Critic 和 PPO 正是把两者结合起来：用 Critic（价值网络）提供稳定的评价信号，用 Actor（策略网络）直接输出动作。

## 高方差与基线

策略梯度通过提高高回报动作的概率来优化策略，但它也带来一个实际困难：方差很大。

同一个策略在同一个环境里跑两局，结果可能差很多。CartPole 一局可能撑 190 步，下一局可能只撑 40 步；语言模型同样的提示下生成两条回答，一条被奖励模型打高分，另一条因为细节错误被扣分。用这些波动很大的 $G_t$ 直接乘上 log probability 梯度，更新方向就会很抖。

还有一个更细的问题。在很多任务里，回报几乎总是正的。CartPole 每多活一步就是 $+1$。如果只看 $G_t$，许多动作都会被加强；但我们真正想知道的不是“这一步之后有没有拿到正分”，而是：

$$
\text{这个动作比当前状态下的平均选择更好吗？}
$$

这就把我们带回前面学过的价值函数。状态价值 $V^\pi(s)$ 可以作为基线，表示“在状态 $s$ 按当前策略正常行动，平均能拿多少分”。动作价值 $Q^\pi(s,a)$ 表示“在状态 $s$ 先做动作 $a$，之后再按当前策略行动，平均能拿多少分”。两者相减得到优势函数：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

如果 $A^\pi(s,a)>0$，说明动作 $a$ 比当前状态下的平均水平更好，应该提高概率；如果 $A^\pi(s,a)<0$，说明它比平均水平差，应该降低概率。

这也是 Actor-Critic 的基本动机。Actor 是策略网络 $\pi_\theta$，负责行动；Critic 是价值网络 $V_\phi(s)$ 或 $Q_\phi(s,a)$，负责给 Actor 一个更稳定的参照。路线一和路线二因此不是互相排斥的。路线一擅长评价，路线二擅长直接行动；现代算法经常把二者结合起来，用 Critic 降低策略梯度方差，用 Actor 保留直接优化策略的能力。

REINFORCE 可以看成最直接的策略梯度方法：采样完整轨迹，用真实回报 $G_t$ 更新策略[^1]。Actor-Critic 用价值估计或优势估计替代原始回报，减少波动[^2]。PPO 进一步限制新旧策略差距，避免一次更新把策略改得太猛[^3]。这些算法的细节不同，但共同起点都是本节的目标函数 $J(\theta)$ 和策略梯度思想。

## 和前后两节的关系

现在可以把价值路线和策略路线放在一起看。

上一节的 value-based 路线学习 $Q(s,a)$，核心问题是“在状态 $s$ 下，哪个动作分数最高”。它适合动作有限、可枚举的情形，也直接导向 Q-Learning 和深度 Q 网络。

本节的 policy-based 路线学习 $\pi_\theta(a\mid s)$，核心问题是“怎样调整策略参数，让采样出来的轨迹平均回报更高”。它适合连续动作、巨大离散动作，以及需要保留随机性的任务，也直接导向 REINFORCE、Actor-Critic、PPO、GRPO 等方法。

但两条路线还有一个共同前提：训练需要数据。无论是更新 $Q(s,a)$，还是估计 $\nabla_\theta J(\theta)$，都要有交互得到的状态、动作、奖励。真正的问题随之出现：这些数据必须来自当前策略吗？为什么 DQN 可以把旧经验放进 replay buffer 反复用，而策略梯度通常更依赖当前策略采样的数据？

下一节就专门回答这个问题：训练数据从哪里来，以及数据和当前策略之间是什么关系。

## 小结

本节讨论了第二条强化学习路线：直接优化策略。

1. 价值路线先学习 $Q(s,a)$，再用 $\arg\max_a Q(s,a)$ 选动作；策略路线直接学习 $\pi_\theta(a\mid s)$，让策略输出动作分布或连续动作。
2. 参数化策略用 $\theta$ 表示可学习参数。离散动作中常用 softmax 输出概率，连续控制中常用高斯分布输出动作。
3. 策略目标 $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G_0]$ 评价的是整套策略参数的平均长期回报，而不是某个状态或某个动作。
4. 轨迹概率 $P_\theta(\tau)$ 由初始状态、策略动作概率和环境转移概率共同决定；对 $\theta$ 求导时，只有策略概率项留下来。
5. 对数求导技巧把策略目标梯度改写成可采样估计的形式：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
G_t
\right].
$$

6. 基础策略梯度直观上是在做一件事：高回报轨迹中的动作被提高概率，低回报轨迹中的动作被降低概率。
7. 原始回报方差很大，因此常引入基线、优势函数和 Critic，让策略更新更稳定。

上一节：[动作价值函数](./value-q)

下一节：[数据从哪里来](./algorithm-taxonomy)

## 参考文献

[^1]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_, 8, 229-256.

[^2]: Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.

[^3]: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.

[^5]: Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.), Chapter 13: Policy Gradient Methods. MIT Press.

[^6]: OpenAI Spinning Up. "[Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html)." 该文档将 VPG 说明为 on-policy 策略梯度算法。
