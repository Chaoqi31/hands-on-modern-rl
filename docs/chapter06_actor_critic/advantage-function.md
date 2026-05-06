# 6.1 优势函数

第 5 章末尾我们发现：减掉基线 $V(s)$ 可以降低策略梯度的方差，而不改变梯度的方向。本节将深入这个关键洞察，引出优势函数——它是连接 Actor 和 Critic 的桥梁。

::: tip 本节会用到的前置知识

- [REINFORCE 策略梯度](../chapter05_policy_gradient/policy-gradient) $\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot G_t$——基线要加在哪里
- [状态价值 $V(s)$](../chapter03_mdp/bellman-equation)——最好的基线是什么
- [动作价值 $Q(s,a)$](../chapter03_mdp/value-q)——优势函数的定义依赖 $Q$ 和 $V$ 的差
- [TD Error](../chapter03_mdp/dp-mc-td) $\delta = r + \gamma V(s') - V(s)$——优势函数的实用估计方法
  :::

## 从基线到优势函数

回忆第 5 章 REINFORCE 的[策略梯度](../chapter05_policy_gradient/policy-gradient)：

$$\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot G_t$$

$G_t$ 是从当前步到 episode 结束的总回报（回顾：[折扣累积回报](../chapter03_mdp/mdp)）。问题在于 $G_t$ 波动巨大——同一个策略、同一个状态，跑两次可能拿到完全不同的 $G_t$。

减掉基线 $V(s)$ 后：

$$\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot (G_t - V(s))$$

括号里的 $G_t - V(s)$ 就是**优势函数（Advantage Function）** 的一种估计。优势函数的正式定义是：

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s) \tag{6.1}$$

其中 $Q^\pi(s,a)$ 是[动作价值函数](../chapter03_mdp/value-q)（"在状态 $s$ 先做动作 $a$，之后按策略行动的期望回报"），$V^\pi(s)$ 是[状态价值函数](../chapter03_mdp/bellman-equation)（"在状态 $s$ 按策略行动的期望回报"）。两者的差恰好是"因为做了动作 $a$，多拿了多少分"。

优势函数的含义是：**做了这个动作，比"平均能拿多少分"好了多少。**

- $A > 0$：这个动作比预期好，应该多选
- $A < 0$：这个动作比预期差，应该少选
- $A \approx 0$：这个动作和预期差不多

用下棋类比：$V(s)$ 是"这个棋局整体胜率 60%"，$Q(s, \text{出车})$ 是"走车之后胜率 75%"。优势 $A = 75\% - 60\% = 15\%$，说明走车比平均水平好了 15%，是个好选择。

## 为什么优势函数比 G_t 更好

优势函数之所以能降低方差，核心在于它**减掉了"本来就能拿到的分"**，只保留"因为做了这个动作多拿的分"。

一个具体的数值例子：假设某状态 $s$ 下，策略的平均回报是 $V(s) = 10$。两个 episode 的回报分别是 $G_t^{(1)} = 15$ 和 $G_t^{(2)} = 5$。

|           | 用 $G_t$              | 用 $A = G_t - V(s)$     |
| --------- | --------------------- | ----------------------- |
| Episode 1 | 梯度信号 = 15（很大） | 梯度信号 = 5（适中）    |
| Episode 2 | 梯度信号 = 5（中等）  | 梯度信号 = -5（负方向） |

用 $G_t$ 时，两个 episode 都给出正的梯度信号（只是大小不同），看起来动作都是"好的"。用 $A$ 时，Episode 1 给出正信号（"比平均好了 5"），Episode 2 给出负信号（"比平均差了 5"）。优势函数把"绝对回报"变成了"相对回报"，让信号更精确、方差更低。

## 用 TD Error 估计优势

优势函数的理论定义是 $A = Q - V$，但实际中我们通常不直接计算 $Q$。利用[TD Error](../chapter03_mdp/dp-mc-td)，可以用一种更高效的方式来估计 $A$：

$$A(s,a) \approx r + \gamma V(s') - V(s) = \delta \tag{6.2}$$

这就是第 3 章介绍的 TD Error——它衡量"走了一步之后，实际结果比预测好了多少"。用 TD Error 替代 $G_t$ 作为策略梯度的信号，有两个好处：

1. **不需要等 episode 结束**——每走一步就能更新（$G_t$ 需要跑完一整局，这是[MC 方法](../chapter03_mdp/dp-mc-td)的限制）
2. **方差更低**——$\delta$ 只涉及一步的随机性（$G_t$ 涉及整条轨迹的随机性）

这是 MC → TD 的演进在策略空间的再现：REINFORCE 用 $G_t$（MC），Actor-Critic 用 $\delta$（TD）。

|          | **REINFORCE (MC)**           | **Actor-Critic (TD)**                              |
| -------- | ---------------------------- | -------------------------------------------------- |
| 优势估计 | $G_t - V(s)$（需要完整轨迹） | $r + \gamma V(s') - V(s) = \delta$（走一步就更新） |
| 更新时机 | episode 结束后               | 每走一步                                           |
| 方差     | 高                           | 低                                                 |
| 代价     | 无                           | 需要训练 Critic                                    |

## Critic：V(s) 的神经网络实现

要计算 $\delta = r + \gamma V(s') - V(s)$，你需要知道 $V(s)$ 和 $V(s')$。但在真实问题中，$V$ 是未知的——需要一个网络来估计它。这个网络就是 **Critic**。

```
Actor（策略网络）           Critic（价值网络）
  输入：状态 s                 输入：状态 s
  输出：π_θ(a|s) 概率分布     输出：V_φ(s) 标量
  作用：选动作                作用：评估状态价值
  参数：θ                     参数：φ
```

Actor 和 Critic 共享输入（状态 $s$），但输出不同：Actor 输出动作概率分布，Critic 输出价值标量。它们通过优势函数 $A \approx \delta$ 协作：Critic 给出评估，Actor 根据评估调整行为。

但 Critic 怎么训练？它怎么学会准确估计 $V(s)$？下一节将展开第 3 章速览过的 [DP、MC、TD](../chapter03_mdp/dp-mc-td) 三种方法在 Critic 训练中的具体应用。[Critic 训练方法](./critic-training)
