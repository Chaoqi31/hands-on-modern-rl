# Actor-Critic 架构

上一节我们介绍了优势函数 $A(s,a)$ 和 Critic 的训练方法。现在让我们把所有零件组装起来，看看 Actor 和 Critic 是如何协作的。

## 从 REINFORCE 到 Actor-Critic：关键的一步

REINFORCE 的梯度公式是：

$$\nabla_\theta J \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

$G_t$ 是完整轨迹的累积回报——这就是 REINFORCE 方差大的根源。

上一节我们发现，减掉基线 $V(s)$ 可以降方差。而更进一步，**不需要等 episode 结束**——用 TD Error $\delta = r + \gamma V(s') - V(s)$ 就能替代 $G_t - V(s)$ 作为优势估计：

$$\nabla_\theta J \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta$$

这一替换带来的改变是根本性的：

| | REINFORCE | Actor-Critic |
|---|---|---|
| 优势估计 | $G_t$（MC，需要完整轨迹） | $\delta = r + \gamma V(s') - V(s)$（TD，走一步就更新） |
| 更新时机 | episode 结束后 | 每走一步 |
| 方差 | 高 | 低 |
| 代价 | 无 | 需要训练 Critic |

## 第二步：最好的基线是 $V(s)$

基线可以是任何不依赖动作的东西——常数、移动平均、随便什么函数。但最好的基线是什么？

如果你回头看第 3 章对状态价值函数的定义——$V^\pi(s)$ 是"从状态 $s$ 出发，遵循策略 $\pi$，期望能拿多少分"——你会发现这不正是基线想要的"正常情况下能跑多少分"吗？

把基线替换为 $V(s)$：

$$\nabla_\theta J \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - V(s_t))$$

这里 $G_t - V(s_t)$ 有一个专门的名字——**优势函数**（Advantage Function）$A(s, a)$：

$$A(s, a) = G_t - V(s)$$

优势函数回答的是一个极其实际的问题："在这个状态下，选择动作 $a$ 比随便按策略执行**好了多少**？"

- $A > 0$：这个动作比平均水平好 → 增加它的概率
- $A < 0$：这个动作比平均水平差 → 降低它的概率
- $A \approx 0$：这个动作中规中矩 → 概率基本不变

## 第三步：用 TD Error 替代 $G_t$——不用等 episode 结束

还有最后一步优化。$G_t$ 有一个工程上的麻烦：它需要跑完整个 episode 才能计算。在赌博机这种一步结束的场景无所谓，但在 CartPole（可能撑几百步）或者围棋（几百步）中，你必须等到游戏结束才能更新策略。

还记得第 3 章的 TD Error 吗？

$$\delta = r + \gamma V(s') - V(s)$$

TD Error 只依赖一步转移——当前状态 $s$、动作 $a$、即时奖励 $r$、下一状态 $s'$。不需要等到 episode 结束。而且，TD Error 本身就是优势函数的一种估计：

$$A(s,a) \approx \delta = r + \gamma V(s') - V(s)$$

直觉上很清楚：TD Error 说的是"我刚才拿到奖励 $r$，然后到了状态 $s'$——这和 Critic 对当前状态的评估 $V(s)$ 相比，好了多少？"如果 $r + \gamma V(s')$ 比 $V(s)$ 大，说明这一步走得好；如果小，说明走得差。

把 REINFORCE 和 Actor-Critic 放在一起对比：

|                     | REINFORCE               | Actor-Critic                |
| ------------------- | ----------------------- | --------------------------- |
| 动作评估            | $G_t$（整条轨迹的回报） | $\delta$（一步的 TD Error） |
| 需要跑完 episode 吗 | 是                      | 否，每走一步就能更新        |
| 方差                | 高（整条轨迹的随机性）  | 低（只有一步的随机性）      |
| 偏差                | 无偏                    | 有偏（用估计值更新估计值）  |
| 类比                | 跑完马拉松再看总分      | 每跑一公里就调整配速        |

Actor-Critic 用一步的信息就更新策略，代价是引入了偏差——因为 Critic 自己的估计 $V(s')$ 可能不准。但实践中，用一点偏差换来大幅降低的方差，几乎总是划算的。

## Actor-Critic：两个网络，各司其职

把上面三步整合起来，就得到了强化学习中最经典的架构。Actor 负责选择动作，Critic 负责评估动作的好坏，两者通过优势函数 $A(s,a)$ 协作：

```
Actor-Critic 数据流

  状态 s
    │
    ├──→ Actor（策略网络）
    │      π(a|s) → 选动作 a
    │                  │
    │              执行动作 a
    │                  │
    │                  ▼
    │              环境 → 返回 r, s'
    │                  │
    ├──→ Critic（价值网络）  │
    │      V(s)  ──────────┤
    │      V(s') ──────────┤
    │                      │
    │      δ = r + γV(s') - V(s)
    │            │
    │            ▼
    │      Actor 更新：θ ← θ + α·∇log π(a|s)·δ
    │      Critic 更新：V(s) ← V(s) + α·δ
    │
    └──→ 下一步，重复以上过程
```

两个网络共享同一个输入（状态 $s$），但各做各的事：

| 网络             | 角色     | 输入     | 输出                 | 学习目标         |
| ---------------- | -------- | -------- | -------------------- | ---------------- |
| Actor（演员）    | 选择动作 | 状态 $s$ | 动作概率 $\pi(a\|s)$ | 最大化累积奖励   |
| Critic（评论家） | 评估局面 | 状态 $s$ | 价值估计 $V(s)$      | 准确预测未来回报 |

如果你仔细看 Critic 的更新规则，$V(s) \leftarrow V(s) + \alpha \cdot \delta$——这不就是第 3 章的 TD Learning 吗？**Critic 本质上就是第 3 章价值函数 $V(s)$ 的神经网络实现**，它独立地学习"每个状态值多少分"。Actor 则是策略 $\pi(a|s)$ 的神经网络实现，它根据 Critic 提供的评估来调整自己的行为。

两个函数逼近器（第 3 章从表格到深度学习的过渡）协同工作——Critic 帮 Actor 判断"这个动作比平均好多少"，Actor 根据判断调整策略，然后新的策略又产生新的数据让 Critic 学得更好。这就是 Actor-Critic 名字的由来。

### 用 PyTorch 实现 Actor-Critic

Actor-Critic 的代码比 REINFORCE 多了一个 Critic 网络，但结构依然清晰：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

# ==========================================
# 1. Actor-Critic 网络（共享特征提取层）
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 共享的特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )
        # Actor 头：输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic 头：输出状态价值
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

# ==========================================
# 2. 训练循环（每步更新，不需要等 episode 结束）
# ==========================================
env = gym.make("CartPole-v1")
model = ActorCritic(state_dim=4, action_dim=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
gamma = 0.99

reward_history = []

for episode in range(500):
    state, _ = env.reset()
    total_reward = 0

    while True:
        state_t = torch.FloatTensor(state)

        # Actor 选动作，Critic 评估状态
        probs, value = model(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward

        # Critic 评估下一个状态
        with torch.no_grad():
            _, next_value = model(torch.FloatTensor(next_state))
            next_value = 0 if done else next_value

        # TD Error = 优势估计
        td_target = reward + gamma * next_value
        td_error = td_target - value

        # Actor 损失：策略梯度 × 优势
        actor_loss = -log_prob * td_error.detach()

        # Critic 损失：让 V(s) 接近 TD Target
        critic_loss = td_error.pow(2)

        # 总损失
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        if done:
            break

    reward_history.append(total_reward)
    if (episode + 1) % 50 == 0:
        avg = np.mean(reward_history[-50:])
        print(f"Episode {episode+1} | Avg Reward: {avg:.1f}")
```

和 REINFORCE 的代码相比，关键区别是：多了一个 Critic 网络（输出 $V(s)$），用 TD Error（`td_target - value`）替代了 $G_t$，Critic 有自己的损失函数（MSE），而且不需要跑完 episode 才更新。

### CartPole 上的 Actor-Critic 训练曲线

```
Actor-Critic 在 CartPole 上的训练曲线

 500 ┤
     │                              ━━━━━━━━━━━━━━━
 400 ┤                         ━━━━
     │                    ━━━━
 300 ┤              ━━━━━
     │         ━━━━
 200 ┤    ━━━━
     │ ━━
 100 ┤╱
     └────────────────────────────────────────────
     0    50   100  150  200  250  300  350  400  450  500
                    Episode

 对比 REINFORCE 的典型曲线（更多锯齿、更慢收敛）
```

Actor-Critic 在 CartPole 上通常在 200-300 个 episode 内就能稳定到 500 分（满分），而 REINFORCE 可能需要 500+ episode 且曲线锯齿明显。这就是"用偏差换方差"的收益——每一步都有更稳定的梯度信号，策略更新不再被运气牵着走。

## Actor-Critic 的后续演进

Actor-Critic 不是终点，而是一个骨架。后续章节中你会看到它的各种变体：

| 章节         | 变体              | 关键改进                                          |
| ------------ | ----------------- | ------------------------------------------------- |
| 第 6 章 PPO  | PPO-Clip          | 限制策略更新幅度，防止"步子迈太大"                |
| 第 6 章 GAE  | 广义优势估计      | 多步 TD Error 的指数加权和，精确控制偏差-方差权衡 |
| 第 8 章 DPO  | 隐式 Actor-Critic | 用偏好数据替代 Critic，去掉 on-policy 的限制      |
| 第 8 章 GRPO | 去掉 Critic       | 用组内均值替代 $V(s)$，省掉一个网络               |

所有的变体都共享同一个骨架：一个负责选择的网络 + 一个负责评估的信号。变化的只是"评估信号怎么来"和"选择网络怎么更新"。

<details>
<summary>思考题：既然 Actor-Critic 比 REINFORCE 好，为什么不用纯 Critic（只用 V）？</summary>

因为只有 Critic 没办法直接输出策略。Critic 学的是 $V(s)$ 或 $Q(s,a)$，从中推导策略需要用 $\arg\max_a Q(s,a)$——但在连续动作空间中，这个 $\arg\max$ 不存在解析解（你不可能对无限多个连续值逐一比较）。

Actor 的价值在于：它直接输出动作概率，天然适用于连续动作空间。这就是为什么需要两个网络——Critic 负责"评价"，Actor 负责"选择"，缺一不可。

</details>

<details>
<summary>思考题：Actor-Critic 的"偏差"从哪来？它有害吗？</summary>

偏差来自 Critic 的自举（Bootstrapping）——Critic 用自己的估计 $V(s')$ 来更新 $V(s)$。如果 $V(s')$ 本身就不准确，误差会传播回来。这就像你用一把不准的尺子去校准另一把尺子——误差会累积。

但这种偏差不一定是坏事。适度的偏差可以换来更低的方差，整体上可能比无偏但高方差的 REINFORCE 收敛更快。第 6 章的 GAE 就是在精确控制这个"偏差-方差权衡"——用参数 $\lambda$ 在纯 TD（高偏差低方差）和纯 MC（无偏高方差）之间平滑插值。

</details>

现在让我们回到代码，用实验亲眼看到基线的效果——[基线实验与总结](./baseline-experiment)。

---

[^2]: Sutton, R. S., et al. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.
