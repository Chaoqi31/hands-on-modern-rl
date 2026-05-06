# 第 7 章：PPO——稳定训练的艺术

上一章我们搭建了 [Actor-Critic 架构](../chapter06_actor_critic/actor-critic)——Actor 负责选择动作，Critic 负责评估动作的好坏，两者通过[优势函数](../chapter06_actor_critic/advantage-function) $A(s,a)$ 协作。在 CartPole 上，Actor-Critic 表现得相当不错。但当你把同样的架构搬到更复杂的环境（比如月球着陆器）或者更大的模型（比如数十亿参数的语言模型）时，一个严重的问题会浮出水面：**训练不稳定**。

[策略梯度方法](../chapter05_policy_gradient/policy-gradient)有一个臭名昭著的弱点——一步更新太大就会"策略崩溃"。想象你在学骑自行车，如果你一次调整太大的重心，结果不是骑得更好，而是直接摔车。Actor-Critic 的 [TD Error](../chapter06_actor_critic/critic-training) 信号虽然降低了方差，但并没有从根本上解决这个问题。我们需要一种机制来约束每一步更新的幅度，让策略"小步快跑"而不是"一步登天"。这就是 PPO（Proximal Policy Optimization）要解决的核心问题。

::: tip 前置知识回顾
本章会频繁用到以下概念：

- [策略梯度 $\nabla_\theta J$](../chapter05_policy_gradient/policy-gradient)——PPO 在策略梯度的基础上加约束
- [REINFORCE 的高方差问题](../chapter05_policy_gradient/baseline-experiment)——为什么需要各种改进
- [优势函数 $A(s,a)$](../chapter06_actor_critic/advantage-function)——PPO 的策略更新依赖优势信号
- [TD Error 与 Critic 训练](../chapter06_actor_critic/critic-training)——PPO 中 Critic 的训练方式
- [Actor-Critic 架构](../chapter06_actor_critic/actor-critic)——PPO 是 Actor-Critic 的变体
:::

本章沿着"动手 → 理论 → 约束 → 估计"的路径展开：先跑一个 PPO 训练实验，亲眼看到训练曲线和各种指标；再拆解背后的数学原理、裁剪机制和优势估计方法。

| 小节                                             | 你会回答的问题                                                                             |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| [动手：PPO 训练 LunarLander](./ppo-lunar-lander) | PPO 的训练过程长什么样？Reward、Entropy、Clip Fraction 这些指标怎么看？                    |
| [PPO 数学推导](./ppo-math)                       | PPO 的公式怎么来的？从策略梯度到裁剪代理目标的完整推导链条是什么？完整损失函数包含哪几项？ |
| [信任域与裁剪](./trust-region-clipping)          | 为什么一步更新太大会崩溃？TRPO 的 KL 约束和 PPO 的裁剪各是怎么做的？                       |
| [GAE、奖励模型与 LLM 对齐](./gae-reward-model)   | GAE 如何在偏差和方差之间插值？LLM 对齐中的 PPO 需要几个模型同时跑？                        |

让我们先跑起来看看效果——[动手：PPO 训练 LunarLander](./ppo-lunar-lander)。
