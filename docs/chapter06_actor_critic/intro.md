# 第6章：Actor-Critic——两条路线的融合

第 4 章走了路线一（Value-Based）：学 $Q(s,a)$，选分数最高的（回顾：[Q(s,a) 与贪心策略](../chapter03_mdp/value-q)）。打分准确，但不擅长探索，且只能处理离散动作。第 5 章走了路线二（Policy-Based）：直接优化 $J(\theta)$（回顾：[策略目标函数](../chapter03_mdp/policy-objective)）。擅长探索、支持连续动作，但方差太大——同一策略跑两次，梯度估计可能天差地别。

上一章末尾我们发现了一个关键线索：减掉基线可以降低方差（回顾：[Baseline 实验](../chapter05_policy_gradient/baseline-experiment)），而最好的基线就是 $V(s)$（回顾：[状态价值函数](../chapter03_mdp/bellman-equation)）。但 $V(s)$ 本身也需要学习——需要一个专门的网络来估计它。这个网络就是 **Critic**。

本章将把两条路线拼在一起：用路线一的方法训练一个 Critic 来评估动作好坏，用路线二的方法训练一个 Actor 来选动作。这就是 **Actor-Critic 架构**。

::: tip 前置知识回顾
本章是前面所有章节的综合运用，以下概念会在本章频繁出现：

- [状态价值 $V(s)$ 与贝尔曼方程](../chapter03_mdp/bellman-equation)——Critic 的理论基础：$V^\pi(s)$ 衡量"从状态 $s$ 出发，平均能拿多少分"
- [动作价值 $Q(s,a)$](../chapter03_mdp/value-q)——$Q$ 与 $V$ 的差值就是优势函数
- [DP/MC/TD 三种价值估计方法](../chapter03_mdp/dp-mc-td)——训练 Critic 的三种策略
- [TD Error $\delta = r + \gamma V(s') - V(s)$](../chapter03_mdp/dp-mc-td)——Critic 的核心训练信号
- [策略目标 $J(\theta)$ 与策略梯度](../chapter03_mdp/policy-objective)——Actor 的优化目标
- [REINFORCE 与基线实验](../chapter05_policy_gradient/baseline-experiment)——为什么需要 $V(s)$ 作为基线
  :::

## 本章结构

| 小节                                 | 核心问题                                            |
| ------------------------------------ | --------------------------------------------------- |
| [优势函数](./advantage-function)     | 优势函数 $A(s,a)$ 是什么？为什么它比 $G_t$ 更好？   |
| [Critic 训练方法](./critic-training) | 怎么训练 Critic 来估计 $V(s)$？DP/MC/TD 的具体实现  |
| [Actor-Critic 架构](./actor-critic)  | Actor 和 Critic 怎么协作？TD Error 如何替代 $G_t$？ |
| [项目：AlphaGo 简易复现](./alphago)  | Actor-Critic + MCTS 能做什么？                      |

让我们从优势函数开始——它是连接 Actor 和 Critic 的桥梁。[优势函数](./advantage-function)
