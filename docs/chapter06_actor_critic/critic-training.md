# Critic 训练方法

上一节定义了优势函数 $A(s,a) \approx \delta = r + \gamma V(s') - V(s)$，并引出了 Critic 网络作为 $V(s)$ 的估计器。本节展开第 3 章速览过的 DP、MC、TD 三种方法在 Critic 训练中的具体实现。

## DP：理论基准

如果完全知道环境的转移概率 $P$ 和奖励函数 $R$，可以直接用贝尔曼方程迭代 Critic：

$$V_\phi(s) \leftarrow \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_\phi(s') \right]$$

反复对所有状态执行这个更新，$V_\phi$ 会收敛到 $V^\pi$ 的精确值。在这个基础上，还可以进行**策略改进**——在状态 $s$ 选择让 $Q(s,a)$ 最大的动作。"评估策略 → 改进策略 → 再评估"的循环就是**策略迭代（Policy Iteration）**，理论上保证收敛到最优策略。

但在真实问题中，几乎不可能知道完整的 $P$ 和 $R$。DP 在 Actor-Critic 中的角色更多是理论基准——它告诉你"知道一切时 Critic 的最优答案"。

## MC：用完整轨迹更新 Critic

跑完一个完整的 episode，用实际回报 $G_t$ 来更新 Critic。Critic 的损失函数是均方误差：

$$L_{\text{Critic}} = \left( G_t - V_\phi(s) \right)^2 \tag{6.3}$$

$G_t - V_\phi(s)$ 是 Critic 的预测误差——实际拿了 $G_t$ 分，但之前预测是 $V_\phi(s)$ 分。MC 方法给出**无偏估计**（用的是真实回报），但有两个限制：

1. **必须等 episode 结束**才能计算 $G_t$，不能边走边学
2. **方差大**——不同 episode 的 $G_t$ 波动剧烈

在神经网络实现中，MC 方法等价于：跑完一个 episode，收集所有 $(s_t, G_t)$ 对，然后用这些数据做一次梯度下降更新 Critic 的参数 $\phi$。

## TD：走一步就更新（实际首选）

用 TD Error 来更新 Critic。Critic 的损失函数是：

$$L_{\text{Critic}} = \left( r + \gamma V_\phi(s') - V_\phi(s) \right)^2 = \delta^2 \tag{6.4}$$

最小化 $\delta^2$ 就是让 Critic 的预测越来越准确。TD 方法的优势：

1. **不需要等 episode 结束**——每走一步就能更新
2. **方差低**——$V_\phi(s')$ 作为"锚点"稳定了估计
3. **与 Actor 的更新节奏一致**——两者都是走一步更新一次

代价是引入了**偏差**：$V_\phi(s')$ 本身也是一个估计值，不是真实的价值。但实际中，这个偏差远小于方差降低带来的好处。

## 三种方法的对比

| | **DP** | **MC** | **TD** |
| --- | --- | --- | --- |
| **用于 Critic 训练？** | 理论基准 | 可以用 | **实际首选** |
| **需要 episode 结束？** | 不需要 | 需要 | 不需要 |
| **无偏？** | 是 | 是 | 否（有偏但方差低） |
| **方差** | 低 | 高 | 中 |
| **自举** | 是 | 否 | 是 |

实际中，Actor-Critic 几乎都用 TD 方法来训练 Critic。在更高级的实现中（如第 8 章的 GAE），MC 和 TD 会被组合使用——通过参数 $\lambda$ 在两者之间插值，获得偏差和方差的最佳平衡。

## Critic 训练的完整流程

将以上内容整合，Actor-Critic 的单步训练流程如下：

1. 在状态 $s$ 下，Actor 选择动作 $a$，环境返回 $r$ 和 $s'$
2. Critic 计算当前预测 $V_\phi(s)$ 和下一步预测 $V_\phi(s')$
3. 计算 TD Error：$\delta = r + \gamma V_\phi(s') - V_\phi(s)$
4. 用 $\delta^2$ 作为损失更新 Critic 的参数 $\phi$
5. 用 $\delta$ 作为优势估计更新 Actor 的参数 $\theta$

Critic 的参数 $\phi$ 沿着"让 $\delta^2$ 更小"的方向更新——预测越来越准。Actor 的参数 $\theta$ 沿着"让正 $\delta$ 的动作概率更高"的方向更新——选择越来越好。两者形成良性循环：Critic 的评分越准，Actor 的进步就越快；Actor 尝试的新动作越多，Critic 看到的数据就越丰富，评分也越准。

## 参考文献

[^1]: Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. _Machine Learning_, 3(1), 9-44.

[^2]: Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. _ICML_. [arXiv:1602.01783](https://arxiv.org/abs/1602.01783)
