# 4.4 Mountain Car 与稀疏奖励

## 本节导读

**核心内容**

- 理解稀疏奖励环境的本质困难：到达目标之前没有任何正向反馈，随机探索几乎不可能偶然成功。
- 掌握乐观初始化（Optimistic Initialization）如何通过初始值设计隐式驱动探索。
- 学会函数逼近的泛化能力在稀疏奖励中的优势，以及奖励塑形策略的核心思路。

**核心公式**

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right] \quad \text{（Q-Learning 更新：在稀疏奖励下依然成立，但信号不足导致学习停滞）}
$$

$$
Q_0(s,a) = Q_{\text{init}}, \quad Q_{\text{init}} > Q^*(s,a) \quad \text{（乐观初始化：用高于真实动作价值的初始估计驱动探索）}
$$

$$
r_t^{\text{shaped}} = r_t + \gamma \Phi(s') - \Phi(s) \quad \text{（势函数奖励塑形：在不改变最优策略的前提下增强中间信号）}
$$

**为什么需要这些公式**

上一节我们验证了经验回放和目标网络在 DQN 中的作用，但那些实验有一个隐含前提——每一步都有足够密集的奖励信号。Mountain Car 打破了这个前提：在到达山顶之前，环境只返回 -1，没有任何"快到了"或"方向对了"的提示。Q-Learning 的更新公式本身没有问题——你还记得吧，每走一步，TD Target = $r + \gamma \max_{a'} Q(s',a')$，用"实际奖励 + 下一状态最优价值"来修正当前 $Q$ 值。但当 TD Target 永远是 -1 时，$Q$ 值对所有状态-动作对都收敛到相同的负值，智能体根本无法区分哪个动作更好。乐观初始化通过把 $Q$ 的初始值设得远高于真实回报，让智能体"以为"每个动作都很有前途，从而在被现实"打脸"的过程中自然地探索更多状态-动作对。奖励塑形则从另一个方向切入——不改算法，而是给环境加中间信号。这三条公式分别对应"问题诊断""探索增强""信号增强"三个层面，是应对稀疏奖励的基本工具箱。

上一节我们分析了 DQN 训练过程中经验回放和目标网络各自的作用。那些实验有一个共同的前提：**奖励信号足够密集**——CartPole 每步都有 +1，GridWorld 每步都有 -1。但现实中的 RL 问题不总是这么"友善"。本节换一个环境，看看当奖励变得极度稀疏时会发生什么。

## Mountain Car 环境

**MountainCar-v0** 是 Gymnasium 中经典的稀疏奖励环境。一辆小车被困在两座山之间的谷底，目标是爬上右侧山顶（位置 ≥ 0.5）。动作空间只有 3 个：向左推（0）、不推（1）、向右推（2）。状态是二维连续向量 `[位置, 速度]`。

关键在于**奖励设计**：每一步的奖励都是 -1，直到到达山顶才结束。注意，到达山顶的那一步奖励依然是 -1；只是 episode 结束以后，后续价值按 0 处理。没有"快到了"的提示，没有"方向对了"的鼓励——在到达目标之前，无论你怎么挣扎，环境都只回一个冷冰冰的 -1。

```python
import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0")
rng = np.random.default_rng(0)
obs, info = env.reset()
print(f"状态空间: {env.observation_space}")
print(f"动作空间: {env.action_space}")
print(f"初始状态: {obs}")
# 状态空间: Box([-1.2  -0.07], [0.6  0.07], (2,), float32)
# 动作空间: Discrete(3)
# 初始状态: [-0.48  0.  ]   ← 谷底附近
```

与 CartPole 对比：

| 环境             | 状态     | 动作     | 每步奖励                     | 到达目标的难度           |
| ---------------- | -------- | -------- | ---------------------------- | ------------------------ |
| CartPole         | 4 维连续 | 2 个离散 | +1（每步）                   | 保持平衡就行             |
| GridWorld        | 4×4 离散 | 4 个方向 | -1（每步）                   | 最多 6 步                |
| **Mountain Car** | 2 维连续 | 3 个离散 | **-1（每步，且无正向信号）** | **需要左右摇摆积蓄动能** |

Mountain Car 的难点不在于"路太长"——在于随机探索几乎不可能偶然到达山顶。

## 动手：Q-Learning 在 Mountain Car 上挣扎

先用上一节的 Q-Learning 试试。由于状态是连续的，需要离散化。

```python
import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0")

# 离散化：把连续状态分成 20 个格子
N_BINS = 20
pos_bins = np.linspace(-1.2, 0.6, N_BINS + 1)
vel_bins = np.linspace(-0.07, 0.07, N_BINS + 1)

def discretize(obs):
    p = np.digitize(obs[0], pos_bins) - 1
    v = np.digitize(obs[1], vel_bins) - 1
    return (np.clip(p, 0, N_BINS - 1), np.clip(v, 0, N_BINS - 1))

# Q 表格
Q = np.zeros((N_BINS, N_BINS, 3))
alpha = 0.1
gamma = 0.99
epsilon = 0.1
episodes = 1000

rewards = []
for ep in range(episodes):
    obs, _ = env.reset(seed=ep)
    s = discretize(obs)
    total_reward = 0
    for step in range(200):  # 最多 200 步
        if rng.random() < epsilon:
            a = int(rng.integers(3))
        else:
            a = int(np.argmax(Q[s[0], s[1]]))

        obs, reward, terminated, truncated, _ = env.step(a)
        total_reward += reward
        s_next = discretize(obs)

        # Q-Learning 更新：真正终止时不再 bootstrap
        bootstrap = 0.0 if terminated else np.max(Q[s_next[0], s_next[1]])
        Q[s[0], s[1], a] += alpha * (
            reward + gamma * bootstrap - Q[s[0], s[1], a]
        )
        s = s_next
        if terminated or truncated:
            break
    rewards.append(total_reward)

print(f"前 100 轮平均回报: {np.mean(rewards[:100]):.1f}")
print(f"后 100 轮平均回报: {np.mean(rewards[-100:]):.1f}")
print(f"最佳回报: {max(rewards):.1f}")
```

预期输出：

```
前 100 轮平均回报: -200.0
后 100 轮平均回报: -197.7
最佳回报: -161.0
```

这里固定了随机种子，方便你复现实验。不同随机种子下的最佳回报会波动；关键不在单个 episode 是否偶尔更短，而在后 100 轮平均回报仍接近 -200。**1000 轮训练后，智能体通常没有稳定学会策略。**

### 为什么失败了？

Mountain Car 的状态空间是连续的二维区域，离散化为 20×20 = 400 个格子。ε-greedy 以 10% 的概率随机探索，而到达山顶需要一系列精确的"左-右-左-右"摇摆动作。随机探索 200 步内恰好走出一条到达山顶的路径——这个概率极低。

这不是算法的问题，而是**探索不足**的问题。ε-greedy 均匀地随机探索，不区分"哪个方向更值得试"，在稀疏奖励环境中几乎无用。

## 解决方案 1：Optimistic Initialization

先看一个常见但容易误解的技巧：乐观初始化。它不是要求把 Q 表格初始化为"正值"，而是要求初始估计**高于真实动作价值**。在 Mountain Car 里，回报通常是负数，所以 `0`、`-5` 甚至某些状态下的 `-100` 都可能比真实 Q 值更乐观；正值当然也可以，但不是必要条件。

```python
# 关键改动：把 Q 初始化为高于真实动作价值的估计

# MountainCar-v0 每一步 reward 都是 -1，包括到达终点那一步。
# 真正终止以后没有后续状态，所以终止后的 bootstrap 价值才是 0。
Q = np.ones((N_BINS, N_BINS, 3)) * 0.0    # 对大多数状态动作都非常乐观

# 也可以用负数，只要它仍高于真实动作价值
Q = np.ones((N_BINS, N_BINS, 3)) * (-5)   # 仍然很乐观

# -100 比失败时的 -200 乐观，但不一定高于所有状态动作的最优价值
Q = np.ones((N_BINS, N_BINS, 3)) * (-100) # 是否乐观要看具体状态
```

**直觉**：如果 agent 相信某个动作能得到比真实情况更好的回报，那么它第一次尝试后往往会发现"没有想象中好"，于是这个动作的 Q 值被拉低。被频繁尝试的动作会逐渐接近真实值，而还没怎么试过的动作仍保持较高初始值——这就自然地产生了"优先探索不确定动作"的压力。

```python
env = gym.make("MountainCar-v0")
rng = np.random.default_rng(0)
N_BINS = 20
pos_bins = np.linspace(-1.2, 0.6, N_BINS + 1)
vel_bins = np.linspace(-0.07, 0.07, N_BINS + 1)

def discretize(obs):
    p = np.digitize(obs[0], pos_bins) - 1
    v = np.digitize(obs[1], vel_bins) - 1
    return (np.clip(p, 0, N_BINS - 1), np.clip(v, 0, N_BINS - 1))

# 关键区别：乐观初始化。0 高于 Mountain Car 中大多数真实回报。
Q = np.ones((N_BINS, N_BINS, 3)) * 0.0

alpha = 0.1
gamma = 0.99
epsilon = 0.05  # 粗离散化下仍保留少量随机探索更稳妥
episodes = 500

rewards_opt = []
for ep in range(episodes):
    obs, _ = env.reset(seed=ep)
    s = discretize(obs)
    total_reward = 0
    for step in range(200):
        if rng.random() < epsilon:
            a = int(rng.integers(3))
        else:
            # 随机打破并列最大值，避免总是偏向动作 0
            q_values = Q[s[0], s[1]]
            best_actions = np.flatnonzero(q_values == q_values.max())
            a = int(rng.choice(best_actions))
        obs, reward, terminated, truncated, _ = env.step(a)
        total_reward += reward
        s_next = discretize(obs)
        bootstrap = 0.0 if terminated else np.max(Q[s_next[0], s_next[1]])
        Q[s[0], s[1], a] += alpha * (
            reward + gamma * bootstrap - Q[s[0], s[1], a]
        )
        s = s_next
        if terminated or truncated:
            break
    rewards_opt.append(total_reward)

print(f"前 50 轮平均回报:  {np.mean(rewards_opt[:50]):.1f}")
print(f"后 50 轮平均回报:  {np.mean(rewards_opt[-50:]):.1f}")
print(f"最佳回报:          {max(rewards_opt):.1f}")
```

一种可能的输出：

```
前 50 轮平均回报:  -200.0
后 50 轮平均回报:  -200.0
最佳回报:          -173.0
```

这组结果看起来没有"神奇变好"，反而正好说明了一个更严谨的事实：**乐观初始化是一种探索压力，不是 Mountain Car 的稳定解法**。在粗离散化、稀疏奖励、200 步上限同时存在时，如果 agent 很少真正到达山顶，表格里就缺少能向前传播的成功信号。初始值可以推动它多试，但不能替代足够的状态表示、探索预算和函数泛化。

## 解决方案 2：函数逼近 + 更大的探索预算

乐观初始化能提供探索压力，但离散化仍然丢失了状态空间的连续结构。一个更好的做法是直接在连续状态空间上用函数逼近（这正是 DQN 的思路）。

```python
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from collections import deque
import random

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

env = gym.make("MountainCar-v0")
env.action_space.seed(seed)

class SimpleQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.net(x)

net = SimpleQNet()
target_net = SimpleQNet()
target_net.load_state_dict(net.state_dict())
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

buffer = deque(maxlen=50000)
batch_size = 64
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
epsilon = epsilon_start

rewards_dqn = []
for ep in range(500):
    obs, _ = env.reset(seed=seed + ep)
    total_reward = 0
    for step in range(200):
        state = torch.FloatTensor(obs)
        if random.random() < epsilon:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = int(net(state).argmax())

        next_obs, reward, terminated, truncated, _ = env.step(a)
        episode_done = terminated or truncated
        total_reward += reward
        buffer.append((obs, a, reward, next_obs, float(terminated)))
        obs = next_obs

        # 训练
        if len(buffer) >= batch_size:
            batch = random.sample(buffer, batch_size)
            s = torch.FloatTensor(np.array([b[0] for b in batch]))
            a_idx = torch.LongTensor([b[1] for b in batch])
            r = torch.FloatTensor([b[2] for b in batch])
            s_next = torch.FloatTensor(np.array([b[3] for b in batch]))
            done = torch.FloatTensor([b[4] for b in batch])

            q_values = net(s).gather(1, a_idx.unsqueeze(1)).squeeze()
            with torch.no_grad():
                q_next = target_net(s_next).max(1)[0]
                target = r + gamma * q_next * (1 - done)

            loss = nn.MSELoss()(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode_done:
            break

    # 更新目标网络（每 10 个 episode）
    if ep % 10 == 0:
        target_net.load_state_dict(net.state_dict())

    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    rewards_dqn.append(total_reward)

    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}, Avg(50): {np.mean(rewards_dqn[-50:]):.1f}, ε: {epsilon:.3f}")

print(f"\n最终 50 轮平均回报: {np.mean(rewards_dqn[-50:]):.1f}")
print(f"最佳回报: {max(rewards_dqn):.1f}")
```

预期输出：

```
Episode 100, Avg(50): -200.0, ε: 0.606
Episode 200, Avg(50): -200.0, ε: 0.367
Episode 300, Avg(50): -200.0, ε: 0.222
Episode 400, Avg(50): -199.5, ε: 0.135
Episode 500, Avg(50): -200.0, ε: 0.082

最终 50 轮平均回报: -200.0
最佳回报: -141.0
```

这个最小 DQN 版本在连续状态空间上直接工作，不需要离散化；但它仍然只是教学骨架，500 轮内也未必能稳定解决 Mountain Car。真正的问题在于：神经网络有**泛化能力**，相似的连续状态会得到相似的 Q 值；可如果成功经验太少、探索预算太小、网络和超参数又很朴素，泛化也没有足够材料可学。换句话说，DQN 给了我们正确的工具形状，但还不等于自动解决稀疏奖励。

## 三种方法对比

| 方法                    | 平均回报（后 50 轮） | 训练轮数 | 关键技巧           |
| ----------------------- | -------------------- | -------- | ------------------ |
| Q-Learning + ε-greedy   | -200.0（没学会）     | 1000     | 无                 |
| Q-Learning + 乐观初始化 | 仍常接近 -200        | 500      | 初始值提供探索压力 |
| 最小 DQN + 经验回放     | 仍常接近 -200        | 500      | 函数逼近骨架       |

稀疏奖励的核心教训：**探索策略、状态表示、训练预算和函数逼近的泛化能力，必须一起配合**。这也是为什么第 5 章的 Policy-Based 方法强调探索、第 7 章的 PPO 用裁剪来稳定策略更新；单独换一个算法名字，并不会自动让稀疏奖励变简单。

## 本节收获

- **稀疏奖励**是 RL 中最常见的挑战之一：到达目标之前没有任何正向反馈，随机探索几乎不可能偶然成功
- **Optimistic Initialization** 是最简单的探索增强之一：把 Q 初始化为高于真实动作价值的估计，让 agent "以为"没试过的动作更好，自然驱动探索
- **函数逼近的泛化**让一次成功经验可以传播到相似状态——这是神经网络比表格方法的核心优势
- 这也解释了为什么 DQN 需要**经验回放**：在稀疏奖励环境中，好不容易收集到的成功经验必须被反复利用

下一节我们来看 DQN 的后续演进——Double DQN、Dueling DQN 和 Rainbow，看看研究者在"让 Q 值估计更准确"这条路上还做了哪些改进。[DQN 家族与视角迁移](./dqn-family)

## 参考文献

[^1]: Moore, A. W. (1990). _Efficient Memory-Based Learning for Robot Control_. PhD thesis, University of Cambridge.

[^2]: Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_, Chapter 8. MIT Press.
