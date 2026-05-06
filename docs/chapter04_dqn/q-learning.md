# 4.1 动手：Q-Learning 与 GridWorld

## 本节导读

**核心内容**

- 掌握 Q-Learning 的更新规则：用 TD 思想估计 $Q(s,a)$，走一步就能修正一张"动作打分表"。
- 在 4×4 GridWorld 中手算 Q 值，亲眼看到 TD Error 从非零逐渐收敛到零。
- 理解 off-policy 与 on-policy 的本质区别（Q-Learning vs SARSA），以及它们在 Cliff Walking 中的不同行为。

**核心公式**

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right] \quad \text{（Q-Learning 更新：用 TD 方法逼近最优动作价值）}
$$

> **Q-Learning 更新规则 (Q-Learning Update Rule)：**
>
> - $Q(s, a)$：当前对"在状态 $s$ 做动作 $a$ 值多少分"的估计。
> - $r + \gamma \max_{a'} Q(s', a')$：TD Target——即时奖励加上下一状态中最优动作的价值估计。
> - $\max_{a'} Q(s', a')$：到了 $s'$ 之后，所有可选动作中得分最高的那个。它意味着"假设下一步你会做出最聪明的选择"。
> - $r + \gamma \max_{a'} Q(s', a') - Q(s, a)$：TD Error——预测与现实之间的落差。大于 0 说明这一步比预想的好，小于 0 说明比预想的差。
> - $\alpha$：学习率，控制每次修正的幅度。$\alpha = 0.1$ 意味着"只往新方向挪 10%"，不急着一步到位。

**为什么需要这些公式**

第 3 章我们学了两件事：**$Q(s,a)$ 给每个动作贴分数牌**——在状态 $s$ 下做动作 $a$，未来一共能拿多少分；**TD 方法走一步就能更新价值估计**——不需要等一整局结束，走一步就能用"实际奖励 + 下一状态的估计"来修正当前预测。Q-Learning 做的事情就是把这两步合二为一——用 TD 的"走一步结小账"思想来估计 $Q$ 而不是 $V$。一旦学到了准确的 $Q^*$，决策就变成了比大小：选分数最高的动作就行。这条"学 Q $\to$ 选最大 Q $\to$ 得到最优策略"的路线，正是 Value-Based 强化学习最核心的范式。

第 3 章还速览了三种估计价值的方法——**DP**（需要环境模型，不实用）、**MC**（要等一整局结束才能更新）、**TD**（走一步就能更新，最实用）。

本节将 TD 方法应用到 $Q$ 上，得到强化学习最经典的算法之一——Q-Learning。

先不急着看公式——跑一个最小的例子，亲眼看看 Q-Learning 在做什么，然后再拆解原理。

## 动手：4×4 GridWorld

GridWorld 是强化学习的"果蝇"——基因简单、繁殖快、容易观察，但揭示的规律适用于从细菌到人类的一切学习系统。它最早出现在 Sutton & Barto 的经典教材中 [^3]，此后几乎所有 RL 论文都用它来做算法验证。原因很简单：16 个格子、4 个动作、确定性的转移规则，整个状态空间只有 64 个 $(s, a)$ 对——小到可以手算，大到足以展示真实算法的完整行为。你在这一页看到的所有现象（TD Error 从非零到零、Q 值从终点倒推回起点、探索策略影响收敛速度），在 CartPole、Atari、甚至大模型的 RLHF 训练中都以同样的模式出现，只不过规模大了几个数量级。

你可能会问：说了这么多，Q-Learning 跑起来到底长什么样？先用一个最小的例子来建立直觉——亲眼看看 TD Error 是怎么从非零逐渐收敛到零的，然后再回头拆解原理。

### 环境设定

<div style="display:grid;grid-template-columns:repeat(4,72px);gap:4px;justify-content:center;margin:20px 0;">
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:linear-gradient(135deg,#6366f1,#818cf8);color:#fff;font-size:22px;font-weight:700;">S</div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;color:#94a3b8;font-size:13px;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;color:#94a3b8;font-size:13px;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;color:#94a3b8;font-size:13px;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;color:#94a3b8;font-size:13px;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;color:#94a3b8;font-size:13px;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;color:#94a3b8;font-size:13px;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;color:#94a3b8;font-size:13px;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;color:#94a3b8;font-size:13px;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;color:#94a3b8;font-size:13px;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;color:#94a3b8;font-size:13px;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;color:#94a3b8;font-size:13px;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;color:#94a3b8;font-size:13px;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;color:#94a3b8;font-size:13px;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;color:#94a3b8;font-size:13px;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:linear-gradient(135deg,#f59e0b,#fbbf24);color:#fff;font-size:22px;font-weight:700;">G</div>
</div>

4×4 网格，左上角紫色格子是起点 $S$（智能体每次从这里出发），右下角金色格子是终点 $G$（到了就结束这一局）。每走一步扣 1 分（$r = -1$），到达终点不扣分（$r = 0$）。这意味着**走 6 步的最短路径被扣 6 分，走 10 步的弯路被扣 10 分**——奖励函数在逼智能体尽快到达终点。每一步只能选一个方向：上、下、左、右。撞墙就原地不动，白扣 1 分。

GridWorld 有很多变体。我们用的是最简单的"空地"版本——所有格子都能走，只是每步扣 1 分。常见的变体还包括：

- **带墙版本**：某些格子不可通行，智能体必须绕路。墙壁把状态空间切成不同区域，增加了规划的难度。
- **带陷阱版本**：某些格子可以走但扣大分（比如 -100），和墙壁的区别在于你可以走，但代价很高——稍后看到的 Cliff Walking 就是这种设计。
- **随机版本**：选了"向上"但有概率滑到相邻方向。比如选 ↑，实际有 80% 向上、10% 向左、10% 向右——MDP 中的转移概率 $P(s'|s,a)$ 不再是确定性的，而是一个分布。这正是第 3 章讲过的"随机环境"：你做了决策，但环境不一定按你的意思来，更接近现实世界的不确定性。这种情况下 Q-Learning 依然有效，只不过需要更多 episode 来"平均掉"随机性 [^3]。
- **8 方向版本**：动作空间扩展为 {↑, ↗, →, ↘, ↓, ↙, ←, ↖}，可以斜着走。最短路径更短了（对角线只需 3 步），但动作空间从 4 变成了 8，Q 表格的行数翻倍——这是探索效率与动作空间大小的经典权衡。

本节先用最简单的确定性、4 方向版本把 Q-Learning 的机制讲清楚，后面的 Cliff Walking 会引入陷阱（悬崖），让你看到奖励函数设计如何改变智能体的行为。

### 第一步：用 MDP 五元组建模

第 3 章我们学过，任何 RL 问题都可以用 MDP 五元组 $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$ 来描述。GridWorld 也不例外。为什么要做这一步？因为 MDP 是所有 RL 算法的**共同语言**——不管你用 Q-Learning、DQN 还是 PPO，底层都在解同一个 MDP。先把问题描述清楚，后面选什么算法才有依据。

| MDP 要素                  | GridWorld 的具体含义                     | 直觉                            |
| ------------------------- | ---------------------------------------- | ------------------------------- |
| $\mathcal{S}$（状态集合） | 16 个格子，编号 0~15                     | "你在网格里的哪个位置"          |
| $\mathcal{A}$（动作集合） | {↑, →, ↓, ←}，4 个方向                   | "你往哪走"                      |
| $P$（转移概率）           | 确定性：向右走就一定向右移一格，撞墙不动 | "你选了方向，环境怎么反应"      |
| $R$（奖励函数）           | 每步 -1，到达 G 奖励 0                   | "走一步扣 1 分，鼓励你走最短路" |
| $\gamma$（折扣因子）      | 0.9                                      | "未来的惩罚比眼前的惩罚轻一点"  |

### 第二步：从 MDP 到 Q 表格——为什么需要给每个动作打分？

MDP 定义了"游戏规则"，但没有告诉你**怎么选动作**。我们需要一个决策工具。

第 3 章我们学过 $V(s)$（状态价值函数），它告诉你"这个局面值多少分"。但光知道局面好不好还不够——站在同一个格子上，往右走和往下走的结果可能完全不同。我们需要的是更细的信息：**不只是"这个格子好不好"，而是"在这个格子往这个方向走好不好"**。

这就是**动作价值函数** $Q(s,a)$ 的由来。"Q"代表 Quality（质量），最早由 Watkins 在 1989 年的博士论文中提出 [^1]。他的核心想法是：如果能为每个"状态-动作对"打一个分，那决策就退化成了比大小——选分数最高的动作就行。这个分数记录的是"在状态 $s$ 做动作 $a$，之后按照最优策略走，未来一共能拿多少分"。

GridWorld 有 16 个格子、4 个方向，所以 Q 表格有 $16 \times 4 = 64$ 行。比如 $Q((0,0), \text{右})$ 记录的是"站在起点向右走，以后一路选最好的走法，总共会被扣多少分"。如果这个值是 -4.10，而 $Q((0,0), \text{下})$ 是 -4.22，那在 (0,0) 应该向右走——因为 -4.10 > -4.22，扣得更少。

但这里有一个问题：**这张表一开始是空白的**。智能体完全不知道网格长什么样，所有 $Q$ 值都是 0——它不知道向右好还是向下好。它必须通过**反复试错**来把正确的分数填进去。这就是 Q-Learning 要做的事。

### 第三步：不同路径，不同代价

在介绍 Q-Learning 怎么填这张表之前，先直观感受一下"选对路"和"选错路"差多少。4×4 网格从 $S(0,0)$ 到 $G(3,3)$，最短路径恰好需要 6 步（3 次向右 + 3 次向下）。任何更长路径都必须走回头路——白费步数、多扣分。下面看三条走法，每格标注了**步数、方向和累计折扣回报**（$\gamma = 0.9$，每步 $r = -1$，到达 G 奖励 $r = 0$）。这些分数就是 Q 表格最终要学会的东西——算法的目标，就是让每个格子的 $Q$ 值尽量接近这条最优路径的得分。

**路径 1：最短路径（6 步）** — 直奔终点，不回头。每格上方是本步折扣得分，下方是累计 $\Sigma$。

<div style="display:grid;grid-template-columns:repeat(4,84px);gap:4px;justify-content:center;margin:16px 0;">
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#dbeafe;border:2px solid #3b82f6;line-height:1.25;text-align:center;"><b style="color:#1d4ed8;font-size:12px;">① →</b><span style="font-size:10px;color:#3b82f6;">-1.00</span><span style="font-size:10px;color:#1e40af;font-weight:600;">Σ -1.00</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#dbeafe;border:2px solid #3b82f6;line-height:1.25;text-align:center;"><b style="color:#1d4ed8;font-size:12px;">② →</b><span style="font-size:10px;color:#3b82f6;">-0.90</span><span style="font-size:10px;color:#1e40af;font-weight:600;">Σ -1.90</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#dbeafe;border:2px solid #3b82f6;line-height:1.25;text-align:center;"><b style="color:#1d4ed8;font-size:12px;">③ →</b><span style="font-size:10px;color:#3b82f6;">-0.81</span><span style="font-size:10px;color:#1e40af;font-weight:600;">Σ -2.71</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#dbeafe;border:2px solid #3b82f6;line-height:1.25;text-align:center;"><b style="color:#1d4ed8;font-size:12px;">④ ↓</b><span style="font-size:10px;color:#3b82f6;">-0.73</span><span style="font-size:10px;color:#1e40af;font-weight:600;">Σ -3.44</span></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#dbeafe;border:2px solid #3b82f6;line-height:1.25;text-align:center;"><b style="color:#1d4ed8;font-size:12px;">⑤ ↓</b><span style="font-size:10px;color:#3b82f6;">-0.66</span><span style="font-size:10px;color:#1e40af;font-weight:600;">Σ -4.10</span></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#dbeafe;border:2px solid #3b82f6;line-height:1.25;text-align:center;"><b style="color:#1d4ed8;font-size:12px;">⑥ ↓</b><span style="font-size:10px;color:#3b82f6;">+0.00</span><span style="font-size:10px;color:#1e40af;font-weight:600;">Σ -4.10</span></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:linear-gradient(135deg,#f59e0b,#fbbf24);color:#fff;font-size:12px;font-weight:700;text-align:center;">G<span style="font-size:10px;font-weight:400;display:block;color:#fffcc0;">到达! 总分-4.10</span></div>
</div>

$$G_0 = -1 - 0.9 - 0.81 - 0.729 - 0.656 + 0 = \mathbf{-4.10}$$

**路径 2：绕远路（8 步）** — 先往右走了一步，又回头往下，再沿底边走

<div style="display:grid;grid-template-columns:repeat(4,84px);gap:4px;justify-content:center;margin:16px 0;">
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fef3c7;border:2px solid #f59e0b;line-height:1.25;text-align:center;"><b style="color:#b45309;font-size:12px;">① →</b><span style="font-size:10px;color:#d97706;">-1.00</span><span style="font-size:10px;color:#92400e;font-weight:600;">Σ -1.00</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fef3c7;border:2px solid #f59e0b;line-height:1.25;text-align:center;"><b style="color:#b45309;font-size:12px;">② ↓</b><span style="font-size:10px;color:#d97706;">-0.90</span><span style="font-size:10px;color:#92400e;font-weight:600;">Σ -1.90</span></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fef3c7;border:2px solid #f59e0b;line-height:1.25;text-align:center;"><b style="color:#b45309;font-size:12px;">④ ↓</b><span style="font-size:10px;color:#d97706;">-0.73</span><span style="font-size:10px;color:#92400e;font-weight:600;">Σ -3.44</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fef3c7;border:2px solid #f59e0b;line-height:1.25;text-align:center;"><b style="color:#b45309;font-size:12px;">③ ←</b><span style="font-size:10px;color:#d97706;">-0.81</span><span style="font-size:10px;color:#92400e;font-weight:600;">Σ -2.71</span></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fef3c7;border:2px solid #f59e0b;line-height:1.25;text-align:center;"><b style="color:#b45309;font-size:12px;">⑤ ↓</b><span style="font-size:10px;color:#d97706;">-0.66</span><span style="font-size:10px;color:#92400e;font-weight:600;">Σ -4.10</span></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fef3c7;border:2px solid #f59e0b;line-height:1.25;text-align:center;"><b style="color:#b45309;font-size:12px;">⑥ →</b><span style="font-size:10px;color:#d97706;">-0.59</span><span style="font-size:10px;color:#92400e;font-weight:600;">Σ -4.69</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fef3c7;border:2px solid #f59e0b;line-height:1.25;text-align:center;"><b style="color:#b45309;font-size:12px;">⑦ →</b><span style="font-size:10px;color:#d97706;">-0.53</span><span style="font-size:10px;color:#92400e;font-weight:600;">Σ -5.22</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fef3c7;border:2px solid #f59e0b;line-height:1.25;text-align:center;"><b style="color:#b45309;font-size:12px;">⑧ →</b><span style="font-size:10px;color:#d97706;">+0.00</span><span style="font-size:10px;color:#92400e;font-weight:600;">Σ -5.22</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:linear-gradient(135deg,#f59e0b,#fbbf24);color:#fff;font-size:12px;font-weight:700;text-align:center;">G<span style="font-size:10px;font-weight:400;display:block;color:#fffcc0;">到达! 总分-5.22</span></div>
</div>

$$G_0 = -1 - 0.9 - 0.81 - 0.729 - 0.656 - 0.590 - 0.531 + 0 = \mathbf{-5.22}$$

路径：①→ ②↓ ③← ④↓ ⑤↓ ⑥→ ⑦→ ⑧→G。注意第③步是**回头路**——从 (1,1) 向左回到 (1,0)，白白多走了一步。

**路径 3：乱走（10 步）** — 迷路了，回头两次

<div style="display:grid;grid-template-columns:repeat(4,84px);gap:4px;justify-content:center;margin:16px 0;">
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fce7f3;border:2px solid #ec4899;line-height:1.25;text-align:center;"><b style="color:#be185d;font-size:12px;">① →</b><span style="font-size:10px;color:#db2777;">-1.00</span><span style="font-size:10px;color:#9d174d;font-weight:600;">Σ -1.00</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fce7f3;border:2px solid #ec4899;line-height:1.25;text-align:center;"><b style="color:#be185d;font-size:12px;">② →</b><span style="font-size:10px;color:#db2777;">-0.90</span><span style="font-size:10px;color:#9d174d;font-weight:600;">Σ -1.90</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fce7f3;border:2px solid #ec4899;line-height:1.25;text-align:center;"><b style="color:#be185d;font-size:12px;">③ ↓</b><span style="font-size:10px;color:#db2777;">-0.81</span><span style="font-size:10px;color:#9d174d;font-weight:600;">Σ -2.71</span></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fce7f3;border:2px solid #ec4899;line-height:1.25;text-align:center;"><b style="color:#be185d;font-size:12px;">⑥ ↓</b><span style="font-size:10px;color:#db2777;">-0.59</span><span style="font-size:10px;color:#9d174d;font-weight:600;">Σ -4.69</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fce7f3;border:2px solid #ec4899;line-height:1.25;text-align:center;"><b style="color:#be185d;font-size:12px;">⑤ ←</b><span style="font-size:10px;color:#db2777;">-0.66</span><span style="font-size:10px;color:#9d174d;font-weight:600;">Σ -4.10</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fce7f3;border:2px solid #ec4899;line-height:1.25;text-align:center;"><b style="color:#be185d;font-size:12px;">④ ←</b><span style="font-size:10px;color:#db2777;">-0.73</span><span style="font-size:10px;color:#9d174d;font-weight:600;">Σ -3.44</span></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fce7f3;border:2px solid #ec4899;line-height:1.25;text-align:center;"><b style="color:#be185d;font-size:12px;">⑦ →</b><span style="font-size:10px;color:#db2777;">-0.53</span><span style="font-size:10px;color:#9d174d;font-weight:600;">Σ -5.22</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fce7f3;border:2px solid #ec4899;line-height:1.25;text-align:center;"><b style="color:#be185d;font-size:12px;">⑧ ↓</b><span style="font-size:10px;color:#db2777;">-0.48</span><span style="font-size:10px;color:#9d174d;font-weight:600;">Σ -5.70</span></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fce7f3;border:2px solid #ec4899;line-height:1.25;text-align:center;"><b style="color:#be185d;font-size:12px;">⑨ →</b><span style="font-size:10px;color:#db2777;">-0.43</span><span style="font-size:10px;color:#9d174d;font-weight:600;">Σ -6.13</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:#fce7f3;border:2px solid #ec4899;line-height:1.25;text-align:center;"><b style="color:#be185d;font-size:12px;">⑩ →</b><span style="font-size:10px;color:#db2777;">+0.00</span><span style="font-size:10px;color:#9d174d;font-weight:600;">Σ -6.13</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:88px;border-radius:10px;background:linear-gradient(135deg,#f59e0b,#fbbf24);color:#fff;font-size:12px;font-weight:700;text-align:center;">G<span style="font-size:10px;font-weight:400;display:block;color:#fffcc0;">到达! 总分-6.13</span></div>
</div>

$$G_0 = -1 - 0.9 - 0.81 - 0.729 - 0.656 - 0.590 - 0.531 - 0.478 - 0.430 + 0 = \mathbf{-6.13}$$

路径：①→ ②→ ③↓ ④← ⑤← ⑥↓ ⑦→ ⑧↓ ⑨→ ⑩→G。第④⑤步连续向左回头，第⑧步又绕了个弯——每次回头都是白扣的分。

**对比**

| 路径     | 步数 | 回头次数 | $G_0$     | 多扣的分              |
| -------- | ---- | -------- | --------- | --------------------- |
| 最短路径 | 6    | 0        | **-4.10** | —                     |
| 绕远路   | 8    | 1        | **-5.22** | 比 shortest 多扣 1.12 |
| 乱走     | 10   | 2        | **-6.13** | 比 shortest 多扣 2.03 |

折扣因子 $\gamma$ 的作用：越晚扣的分越"便宜"（$0.9^t$ 越来越小），但多走一步就是多扣一分。$\gamma < 1$ 让智能体既想尽快到达终点，又不用把每一步看得一样重。Q-Learning 的目标就是学出一张打分表，让智能体在每一步都选出总扣分最少的方向。

### 手算第 1 个 Episode：6 步走到终点

现在让我们一步一步地看 Q-Learning 在第一个 episode 中做了什么。假设智能体运气不错，走了最短路径 S→→→↓↓↓。初始 Q 表全是 0，$\alpha = 0.1$，$\gamma = 0.9$。

**第 1 步：S(0,0) → 右 → (0,1)**

<div style="display:grid;grid-template-columns:repeat(4,72px);gap:4px;justify-content:center;margin:12px 0;">
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#fef3c7;border:2px solid #f59e0b;font-size:12px;line-height:1.4;padding:4px;text-align:center;"><b>S</b><span style="color:#b45309;">Q(→)=-0.1</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#dbeafe;border:2px solid #3b82f6;font-size:12px;line-height:1.4;padding:4px;text-align:center;"><b>(0,1)</b><span style="color:#1d4ed8;">Q=全0</span></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;font-size:12px;color:#94a3b8;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;font-size:12px;color:#94a3b8;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;font-size:12px;color:#94a3b8;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;font-size:12px;color:#94a3b8;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;font-size:12px;color:#94a3b8;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;font-size:12px;color:#94a3b8;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;font-size:12px;color:#94a3b8;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;font-size:12px;color:#94a3b8;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;font-size:12px;color:#94a3b8;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;font-size:12px;color:#94a3b8;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;font-size:12px;color:#94a3b8;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;font-size:12px;color:#94a3b8;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;font-size:12px;color:#94a3b8;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;font-size:12px;color:#94a3b8;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:linear-gradient(135deg,#f59e0b,#fbbf24);color:#fff;font-size:14px;font-weight:700;">G</div>
</div>

- 即时奖励 $r = -1$（MDP 中的 $R$ 函数给出）
- 下一状态 $(0,1)$ 的所有 Q 值都是 0（还没学过）
- TD Target $= r + \gamma \max_{a'} Q(s', a') = -1 + 0.9 \times 0 = \mathbf{-1}$
- TD Error $= -1 - 0 = \mathbf{-1}$（"我猜这步不亏不赚，结果扣了 1 分"）
- $Q((0,0), \text{右}) \leftarrow 0 + 0.1 \times (-1) = \mathbf{-0.1}$

**第 2 步：(0,1) → 右 → (0,2)**

<div style="display:grid;grid-template-columns:repeat(4,72px);gap:4px;justify-content:center;margin:12px 0;">
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#fef3c7;font-size:12px;line-height:1.4;padding:4px;text-align:center;"><b>S</b><span style="color:#b45309;">Q(→)=-0.1</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#fef3c7;border:2px solid #f59e0b;font-size:12px;line-height:1.4;padding:4px;text-align:center;"><b>(0,1)</b><span style="color:#b45309;">Q(→)=-0.1</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#dbeafe;border:2px solid #3b82f6;font-size:12px;line-height:1.4;padding:4px;text-align:center;"><b>(0,2)</b><span style="color:#1d4ed8;">Q=全0</span></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;font-size:12px;color:#94a3b8;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:linear-gradient(135deg,#f59e0b,#fbbf24);color:#fff;font-size:14px;font-weight:700;">G</div>
</div>

- $r = -1$，下一状态 $(0,2)$ 的 Q 值全为 0
- TD Target $= -1 + 0.9 \times 0 = \mathbf{-1}$，TD Error $= \mathbf{-1}$
- $Q((0,1), \text{右}) \leftarrow 0 + 0.1 \times (-1) = \mathbf{-0.1}$

**第 3 步：(0,2) → 右 → (0,3)** — 同理，$Q((0,2), \text{右}) = -0.1$

**第 4 步：(0,3) → 下 → (1,3)**

<div style="display:grid;grid-template-columns:repeat(4,72px);gap:4px;justify-content:center;margin:12px 0;">
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#fef3c7;font-size:12px;line-height:1.4;padding:4px;text-align:center;"><b>S</b><span style="color:#b45309;">Q(→)=-0.1</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#fef3c7;font-size:12px;line-height:1.4;padding:4px;text-align:center;"><b>(0,1)</b><span style="color:#b45309;">Q(→)=-0.1</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#fef3c7;font-size:12px;line-height:1.4;padding:4px;text-align:center;"><b>(0,2)</b><span style="color:#b45309;">Q(→)=-0.1</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#fef3c7;border:2px solid #f59e0b;font-size:12px;line-height:1.4;padding:4px;text-align:center;"><b>(0,3)</b><span style="color:#b45309;">Q(↓)=-0.1</span></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#dbeafe;border:2px solid #3b82f6;font-size:12px;line-height:1.4;padding:4px;text-align:center;"><b>(1,3)</b><span style="color:#1d4ed8;">Q=全0</span></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:linear-gradient(135deg,#f59e0b,#fbbf24);color:#fff;font-size:14px;font-weight:700;">G</div>
</div>

- 同理：TD Target = -1，$Q((0,3), \text{下}) = -0.1$

**第 5 步：(1,3) → 下 → (2,3)** — $Q((1,3), \text{下}) = -0.1$

**第 6 步（到达终点）：(2,3) → 下 → G(3,3)**

<div style="display:grid;grid-template-columns:repeat(4,72px);gap:4px;justify-content:center;margin:12px 0;">
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#fef3c7;font-size:12px;line-height:1.4;padding:4px;text-align:center;"><b>S</b><span style="color:#b45309;">Q(→)=-0.1</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#fef3c7;font-size:12px;line-height:1.4;padding:4px;text-align:center;"><b>(0,1)</b><span style="color:#b45309;">Q(→)=-0.1</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#fef3c7;font-size:12px;line-height:1.4;padding:4px;text-align:center;"><b>(0,2)</b><span style="color:#b45309;">Q(→)=-0.1</span></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#fef3c7;font-size:12px;line-height:1.4;padding:4px;text-align:center;"><b>(0,3)</b><span style="color:#b45309;">Q(↓)=-0.1</span></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#fef3c7;font-size:12px;line-height:1.4;padding:4px;text-align:center;"><b>(1,3)</b><span style="color:#b45309;">Q(↓)=-0.1</span></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#fef3c7;border:2px solid #f59e0b;font-size:12px;line-height:1.4;padding:4px;text-align:center;"><b>(2,3)</b><span style="color:#b45309;">Q(↓)=0</span></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:72px;border-radius:10px;background:linear-gradient(135deg,#f59e0b,#fbbf24);border:2px solid #f59e0b;color:#fff;font-size:14px;font-weight:700;">G<br><span style="font-size:11px;color:#fffcc0;">r=0 到达!</span></div>
</div>

这一步和前面不同——到达了终点 G！

- 即时奖励 $r = 0$（到达终点，MDP 的 $R$ 函数定义为 0）
- 终点状态没有后续，$\max_{a'} Q(G, a') = 0$
- TD Target $= 0 + 0.9 \times 0 = \mathbf{0}$
- TD Error $= 0 - 0 = \mathbf{0}$（"猜的 0 分，实际也是 0 分，没猜错"）
- $Q((2,3), \text{下}) \leftarrow 0 + 0.1 \times 0 = \mathbf{0}$

**第 1 个 Episode 结束后的 Q 表**

走完第一个 episode，Q 表里只有路径上经过的 6 个格子有了非零值（其实全是 -0.1，最后一步没变）。其余 10 个格子还是一片空白——因为智能体还没去过那里。

这就是为什么 Q-Learning 需要**反复训练**——一个 episode 只能更新走过的格子，而且更新的幅度很小（$\alpha = 0.1$）。跑上千个 episode 后，信息才会从终点 G 逐层"倒推"到起点 S，Q 值才会收敛。

### 用代码验证

```python
import numpy as np

np.random.seed(3)  # 固定随机种子，确保结果可复现

# 4x4 GridWorld Q-Learning
Q = np.zeros((16, 4))  # 16 个状态, 4 个动作 (上右下左)
alpha, gamma, epsilon = 0.1, 0.9, 0.1
goal = 15  # 右下角的索引

def state_to_idx(row, col):
    return row * 4 + col

def step(state, action):
    """执行动作，返回 (下一状态, 奖励, 是否结束)"""
    row, col = state // 4, state % 4
    if action == 0: row = max(row - 1, 0)      # 上
    elif action == 1: col = min(col + 1, 3)     # 右
    elif action == 2: row = min(row + 1, 3)     # 下
    elif action == 3: col = max(col - 1, 0)     # 左
    next_state = state_to_idx(row, col)
    reward = 0 if next_state == goal else -1
    done = next_state == goal
    return next_state, reward, done

# 训练 1000 个 episode
for ep in range(1000):
    state = 0  # 起点 S
    while state != goal:
        # ε-贪婪：90% 选最优，10% 随机探索
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[state])

        next_state, reward, done = step(state, action)

        # Q-Learning 更新
        td_target = reward + gamma * np.max(Q[next_state])
        td_error = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

        state = next_state

# 打印收敛结果
print("收敛后的 Q((0,0), 右) =", Q[0, 1].round(2))
print("最优路径（从 S 出发的动作序列）：")
state = 0
actions = ["↑", "→", "↓", "←"]
path = []
while state != goal:
    a = np.argmax(Q[state])
    path.append(actions[a])
    state, _, _ = step(state, a)
print(" → ".join(path))
```

预期输出：

```
收敛后的 Q((0,0), 右) = -4.10
最优路径（从 S 出发的动作序列）：
→ → → ↓ ↓ ↓
```

<div style="display:grid;grid-template-columns:repeat(4,72px);gap:4px;justify-content:center;margin:20px 0;">
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:linear-gradient(135deg,#6366f1,#818cf8);color:#fff;font-size:20px;font-weight:700;">S →</div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#dbeafe;font-size:20px;color:#3b82f6;font-weight:600;">→</div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#dbeafe;font-size:20px;color:#3b82f6;font-weight:600;">→</div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#dbeafe;font-size:20px;color:#3b82f6;font-weight:600;">↓</div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#dbeafe;font-size:20px;color:#3b82f6;font-weight:600;">↓</div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#f1f5f9;"></div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:#dbeafe;font-size:20px;color:#3b82f6;font-weight:600;">↓</div>
  <div style="display:flex;align-items:center;justify-content:center;height:72px;border-radius:10px;background:linear-gradient(135deg,#f59e0b,#fbbf24);color:#fff;font-size:22px;font-weight:700;">G</div>
</div>

### 收敛过程

经过大量训练后，Q 值会收敛。以 $Q(S, \text{右})$ 为例：从 $S$ 到 $G$ 最短路径需要 6 步，前 5 步每步 -1，第 6 步到达 G 奖励 0，考虑 $\gamma = 0.9$ 的折扣后：

$$Q((0,0), \text{右}) \approx -1 - 0.9 - 0.81 - 0.729 - 0.656 + 0 = -4.095$$

实际值约 -4.10（与理论值吻合）。此时 TD Error $\approx 0$——预判和实际一致了，学习完成。

这个过程揭示了 Q-Learning 的本质：TD Error 从一开始的 -1，通过成百上千次的微调，逐渐趋近于 0。每一次微调都是在说"上次猜错了，这次修一点"。

::: info 从直觉到公式
跑完了例子，你应该有了一个直觉：Q-Learning 就是"走一步，看一眼，修一点"。接下来的问题是——这"修一点"到底在数学上做了什么？为什么它最终能收敛到最优？现在可以放心地看公式了。
:::

## 从 TD 到 Q-Learning

跑完了例子，你可能已经注意到代码里这几行在反复执行：

```python
td_target = reward + gamma * np.max(Q[next_state])
td_error = td_target - Q[state, action]
Q[state, action] += alpha * td_error
```

这三行代码就是 Q-Learning 的全部。现在来看看它们在数学上到底做了什么。

还记得第 3 章的 TD 方法吗？它用以下公式更新状态价值 $V(s)$：

$$V(s) \leftarrow V(s) + \alpha \underbrace{\left[ r + \gamma V(s') - V(s) \right]}_{\text{TD Error } \delta}$$

Q-Learning 做的事情完全类似，只是把 $V$ 换成 $Q$，并且在 TD Target 中用 $\max$ 代替对下一状态的估计：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

逐项拆解：

| 符号                                       | 含义                                           |
| ------------------------------------------ | ---------------------------------------------- |
| $Q(s, a)$                                  | 当前对"在状态 $s$ 做动作 $a$ 值多少分"的估计   |
| $r + \gamma \max_{a'} Q(s', a')$           | TD Target：即时奖励 + 下一状态中最好动作的价值 |
| $\max_{a'} Q(s', a')$                      | "到了 $s'$ 之后，最好的动作值多少分"           |
| $r + \gamma \max_{a'} Q(s', a') - Q(s, a)$ | TD Error：预测与现实的落差                     |

你可能会问：那个 $\max_{a'}$ 为什么这么重要？它不看所有动作的平均，只看最好的那个。这意味着 Q-Learning 学的是**最优动作价值 $Q^*$**，不管当前用什么策略在探索。这就是离策略（off-policy）学习：用 $\varepsilon$-贪婪策略收集数据，但学的是最优策略的 $Q$ 值。打个比方：你在驾校练车时可能到处乱开（行为策略），但你心里学的永远是"最完美的开法"（目标策略）。

回过头看刚才的手算：TD Target $= -1 + 0.9 \times 0 = -1$，就是"即时奖励 $r=-1$ 加上下一步的最好估计 $0$"。TD Error $= -1 - 0 = -1$，就是"预判的 0 和实际的 -1 之间的落差"。这个落差乘以学习率 $\alpha = 0.1$，就是 Q 值的修正量。

## ε-贪婪：平衡探索与利用

到目前为止有一个问题被我们悄悄跳过了：Q-Learning 需要数据来学习，但它学的是最优 $Q^*$，而不是当前策略的 $Q$。那收集数据时到底用什么策略？

最常用的选择是 **$\varepsilon$-贪婪（$\varepsilon$-greedy）**：

$$a = \begin{cases} \arg\max_a Q(s, a) & \text{以概率 } 1 - \varepsilon \text{（利用）} \\ \text{随机动作} & \text{以概率 } \varepsilon \text{（探索）} \end{cases}$$

$\varepsilon$ 控制探索的程度：$\varepsilon = 0.1$ 意味着 90% 的时间选当前最好的动作，10% 的时间随机尝试。这正是第 3 章讨论的探索-利用困境在路线一中的具体体现——用一个参数来人工平衡。

代码里 `if np.random.random() < epsilon: action = np.random.randint(4)` 就是这行公式的直接翻译。

## Q-Learning 的关键性质

搞清楚了更新规则和探索策略，现在把 Q-Learning 的关键性质做一个总结——这些性质决定了它在什么场景下能用，什么场景下不能用。

| **性质**   | **说明**                                       |
| ---------- | ---------------------------------------------- |
| Off-policy | 学的是 $Q^*$（最优），但可以用任何策略收集数据 |
| Model-free | 不需要知道环境的 $P$ 和 $R$                    |
| 逐步更新   | 每走一步就更新，不需要等 episode 结束          |
| 收敛性     | 在表格情况下，Q-Learning 保证收敛到 $Q^*$ [^5] |

### 收敛性

你可能最关心的问题：Q-Learning 能保证收敛吗？答案是肯定的——至少在表格情况下。Watkins & Dayan (1992) [^5] 证明了：只要满足以下条件，Q-Learning 保证收敛到最优动作价值 $Q^*$：

1. 所有状态-动作对 $(s, a)$ 被无限次访问
2. 学习率 $\alpha$ 满足 $\sum_t \alpha_t = \infty$ 且 $\sum_t \alpha_t^2 < \infty$

条件 1 由 ε-贪婪策略保证（只要 $\varepsilon > 0$，每个动作都有非零概率被选中）。条件 2 要求学习率逐渐减小但不能减得太快——实践中常用 $\alpha_t = 1/t$ 或固定的小常数（如 0.1）。

### Decaying ε：让探索逐渐减少

固定 $\varepsilon = 0.1$ 意味着训练后期仍然有 10% 的时间在随机探索——明明已经知道该怎么走了，却还有 10% 的概率瞎走，这在不必要地损失回报。更实用的做法是**衰减 ε（decaying ε）**：一开始大胆探索，等学得差不多了，就逐渐减少探索，把精力集中在利用上。

$$\varepsilon_t = \max\left(\varepsilon_{\min}, \, \varepsilon_0 - \frac{t}{T_{\text{decay}}}(\varepsilon_0 - \varepsilon_{\min})\right)$$

例如 $\varepsilon_0 = 1.0$，$\varepsilon_{\min} = 0.01$，$T_{\text{decay}} = 10000$：前 10000 步从完全随机线性衰减到 1%，之后保持 1%。这保证了早期充分探索，后期稳定利用。

### On-policy vs Off-policy：SARSA 对比

到目前为止，我们一直在说 Q-Learning 是 off-policy 的。但"off-policy"到底是什么意思？它和"on-policy"的区别有多大？要回答这个问题，最好的办法是把 Q-Learning 和它的 on-policy 版本——SARSA——放在一起比一比。

Q-Learning 的更新中用了 $\max_{a'} Q(s', a')$——它假设下一步会选最优动作。但实际策略（ε-贪婪）在下一步可能随机选了一个非最优动作。这种"学的是最优，做的不是最优"的分离就是 off-policy。

SARSA 是 Q-Learning 的 on-policy 版本，由 Rummery & Niranjan (1994) 提出 [^2]。它的更新公式和 Q-Learning 几乎一模一样，只差一个符号：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

看到区别了吗？Q-Learning 用 $\max_{a'} Q(s', a')$（假设最优），SARSA 用 $Q(s', a')$（实际选的动作 $a'$）。一个只差——$\max$ 变成了 $a'$——但含义完全不同。

|           | **Q-Learning (off-policy)**      | **SARSA (on-policy)**  |
| --------- | -------------------------------- | ---------------------- |
| TD Target | $r + \gamma \max_{a'} Q(s', a')$ | $r + \gamma Q(s', a')$ |
| 学的是    | $Q^*$（最优策略）                | $Q^\pi$（当前策略）    |
| 行为      | 乐观——假设下一步选最优           | 保守——考虑实际探索风险 |

光看公式可能还觉得抽象。让我们用一个经典例子来直观感受两者的差异。

### 动手：Cliff Walking 对比实验

想象一个场景：你面前有一条贴着悬崖的最短路线，走 12 步就到终点，但稍微走偏一步就掉下去（扣 100 分）。Q-Learning 会怎么走？SARSA 又会怎么走？用 Gymnasium 的 CliffWalking-v0 [^4] 来亲眼看看两种算法学到的路径有什么不同。

```python
import gymnasium as gym
import numpy as np

env = gym.make("CliffWalking-v0")
# 4×12 网格，起点 (3,0)，终点 (3,11)
# 最后一行 (3,1)~(3,10) 是悬崖，掉下去回到起点并扣 100 分

def train_qlearning(env, episodes=500, alpha=0.5, gamma=0.95, epsilon=0.1):
    Q = np.zeros((48, 4))  # 48 个状态，4 个动作
    rewards = []
    for ep in range(episodes):
        s, _ = env.reset()
        total = 0
        for step in range(200):
            if np.random.random() < epsilon:
                a = env.action_space.sample()
            else:
                a = int(np.argmax(Q[s]))
            s_next, r, terminated, truncated, _ = env.step(a)
            total += r
            # Q-Learning: 用 max（off-policy）
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) * (1 - terminated) - Q[s, a])
            s = s_next
            if terminated:
                break
        rewards.append(total)
    return Q, rewards

def train_sarsa(env, episodes=500, alpha=0.5, gamma=0.95, epsilon=0.1):
    Q = np.zeros((48, 4))
    rewards = []
    for ep in range(episodes):
        s, _ = env.reset()
        if np.random.random() < epsilon:
            a = env.action_space.sample()
        else:
            a = int(np.argmax(Q[s]))
        total = 0
        for step in range(200):
            s_next, r, terminated, truncated, _ = env.step(a)
            total += r
            # SARSA: 先选下一个动作 a'（on-policy）
            if np.random.random() < epsilon:
                a_next = env.action_space.sample()
            else:
                a_next = int(np.argmax(Q[s_next]))
            Q[s, a] += alpha * (r + gamma * Q[s_next, a_next] * (1 - terminated) - Q[s, a])
            s = s_next
            a = a_next
            if terminated:
                break
        rewards.append(total)
    return Q, rewards

Q_ql, r_ql = train_qlearning(env)
Q_sa, r_sa = train_sarsa(env)

# 提取学到的路径
def extract_path(Q, env):
    s, _ = env.reset()
    path = [s]
    for _ in range(50):
        a = int(np.argmax(Q[s]))
        s, _, terminated, _, _ = env.step(a)
        path.append(s)
        if terminated:
            break
    return path

def path_to_grid(path):
    grid = [['.' for _ in range(12)] for _ in range(4)]
    grid[3][0] = 'S'
    grid[3][11] = 'G'
    for i in range(1, 11):
        grid[3][i] = 'C'  # 悬崖
    for s in path:
        r, c = s // 12, s % 12
        if grid[r][c] not in ('S', 'G'):
            grid[r][c] = '→' if s != path[-1] else '★'
    return grid

path_ql = extract_path(Q_ql, env)
path_sa = extract_path(Q_sa, env)

print("Q-Learning 学到的路径（贴着悬崖）:")
for row in path_to_grid(path_ql):
    print(" ".join(row))
print(f"路径长度: {len(path_ql)-1} 步")

print("\nSARSA 学到的路径（绕开悬崖）:")
for row in path_to_grid(path_sa):
    print(" ".join(row))
print(f"路径长度: {len(path_sa)-1} 步")

print(f"\n后 100 轮平均回报: Q-Learning={np.mean(r_ql[-100:]):.1f}, SARSA={np.mean(r_sa[-100:]):.1f}")
```

预期输出：

**Q-Learning 学到的路径（贴着悬崖，12 步）**

<div style="display:grid;grid-template-columns:repeat(12,44px);gap:2px;justify-content:center;margin:16px 0;">
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <!-- Row 4: S →→→→→→→→→→→ ★ -->
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#6366f1,#818cf8);color:#fff;font-weight:700;font-size:14px;">S</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dbeafe;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dbeafe;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dbeafe;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dbeafe;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dbeafe;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dbeafe;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dbeafe;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dbeafe;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dbeafe;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dbeafe;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#f59e0b,#fbbf24);color:#fff;font-weight:700;font-size:14px;">★</div>
</div>

**SARSA 学到的路径（绕开悬崖，14 步）**

<div style="display:grid;grid-template-columns:repeat(12,44px);gap:2px;justify-content:center;margin:16px 0;">
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <div style="height:40px;border-radius:6px;background:#f1f5f9;"></div>
  <!-- Row 3: →→→→→→→→→→→↓ -->
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dcfce7;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dcfce7;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dcfce7;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dcfce7;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dcfce7;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dcfce7;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dcfce7;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dcfce7;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dcfce7;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dcfce7;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dcfce7;font-size:16px;">→</div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:#dcfce7;font-size:16px;">↓</div>
  <!-- Row 4: S C C C C C C C C C C ★ -->
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#6366f1,#818cf8);color:#fff;font-weight:700;font-size:14px;">S</div>
  <div style="height:40px;border-radius:6px;background:#fecaca;"></div>
  <div style="height:40px;border-radius:6px;background:#fecaca;"></div>
  <div style="height:40px;border-radius:6px;background:#fecaca;"></div>
  <div style="height:40px;border-radius:6px;background:#fecaca;"></div>
  <div style="height:40px;border-radius:6px;background:#fecaca;"></div>
  <div style="height:40px;border-radius:6px;background:#fecaca;"></div>
  <div style="height:40px;border-radius:6px;background:#fecaca;"></div>
  <div style="height:40px;border-radius:6px;background:#fecaca;"></div>
  <div style="height:40px;border-radius:6px;background:#fecaca;"></div>
  <div style="height:40px;border-radius:6px;background:#fecaca;"></div>
  <div style="height:40px;border-radius:6px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#f59e0b,#fbbf24);color:#fff;font-weight:700;font-size:14px;">★</div>
</div>

后 100 轮平均回报：Q-Learning = **-22.1**，SARSA = **-26.3**

**两个关键观察**：

1. **路径不同**：Q-Learning 走最短路径（12 步，贴崖边），SARSA 绕远路（14 步，走第 2 行安全路线）。Q-Learning 的 TD Target 用了 $\max$，所以它假设"到了崖边还能稳稳地继续走"——这是最优策略的行为。但 ε-greedy 有 10% 概率随机走进悬崖，训练期间 Q-Learning 实际上经常掉下去。SARSA 知道自己有随机探索的风险，所以学到了一条更安全的路。

2. **回报不同**：在 ε=0.1 的条件下，Q-Learning 的收敛回报更好（-22 vs -26），因为它的路径更短。但如果 ε 更大（比如 0.3），Q-Learning 的训练过程中会频繁掉崖，训练期间的回报反而比 SARSA 更差——这也是为什么在一些安全关键场景中，on-policy 方法可能更合适。

::: details On-policy vs Off-policy 的本质区别

**On-policy（SARSA）**：行为策略 = 目标策略。你用什么策略收集数据，就学什么策略的值函数。优点是训练稳定（学的和做的一致），缺点是不能复用旧数据。

**Off-policy（Q-Learning）**：行为策略 ≠ 目标策略。你用 ε-greedy 收集数据，但学的是最优策略的 Q\*。优点是样本效率高（可以用任何策略的数据来学），缺点是训练可能不稳定。

在大模型时代：

- **PPO 是 on-policy**：每次都要用当前模型重新生成回答来训练，所以 RLHF 训练非常吃算力
- **DQN 是 off-policy**：经验回放池里的旧数据可以反复利用，所以 Atari 训练更高效
- **DPO 更极端**：连在线生成都不需要，直接用固定的离线偏好数据训练

这个区分将在第 7-9 章反复出现，理解它对选择正确的算法至关重要。
:::

这些性质使 Q-Learning 成为最实用的 Value-Based 方法。但它有一个根本性的限制：**只能用表格存储 Q 值**。16 个格子的 GridWorld 没问题，但 CartPole 的状态是连续的（小车位置、速度、杆子角度……无穷多个状态），Atari 的画面有几十万像素（$210 \times 160 \times 3$，每个像素值都不同）——表格方法的存储需求远超物理设备的容量。

你可能会问：既然表格存不下，那用什么存？答案是第 2 章已经出场过的主角——**神经网络**。

下一节将展示如何用神经网络替代表格，解决状态空间爆炸的问题。[为什么需要 DQN](./from-q-to-dqn)

## 参考文献

[^1]: Watkins, C. J. C. H. (1989). _Learning from delayed rewards_. PhD thesis, King's College, Cambridge. — Q-Learning 算法的原始提出，首次定义了动作价值函数 $Q(s,a)$ 并给出了通过试错学习 Q 值的算法框架。

[^5]: Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. _Machine Learning_, 8(3), 279–292. https://doi.org/10.1007/BF00992698 — Q-Learning 的收敛性证明，证明了在满足一定条件下 Q-Learning 保证收敛到最优动作价值 $Q^*$。

[^2]: Rummery, G. A., & Niranjan, M. (1994). _On-line Q-learning using connectionist systems_. Technical Report CUED/F-INFENG/TR 166, Cambridge University Engineering Department. — SARSA 算法的原始论文，提出了 Q-Learning 的 on-policy 版本。

[^3]: Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press. http://incompleteideas.net/book/the-book.html — 强化学习经典教材，GridWorld 环境的系统介绍和 Q-Learning/SARSA 的详细推导。

[^4]: CliffWalking 环境: Gymnasium 文档 https://gymnasium.farama.org/environments/toy_text/cliff_walking/ — CliffWalking-v0 的官方环境说明。
