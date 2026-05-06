# 3.7 算法数据来源

## 本节导读

**核心内容**

- 分清两件事：**数据由谁采出来**，以及**训练时还能不能继续采新数据**。
- 理解行为策略 $\mu$ 与目标策略 $\pi_\theta$ 的区别：前者负责产生样本，后者是我们正在学习或评估的对象。
- 理解 On-policy 与 Off-policy 的判断条件：一次更新使用的数据，是否来自当前目标策略或足够接近它的采样策略。
- 理解 Online RL 与 Offline RL 的判断条件：训练过程中，数据集是否还能继续增长。
- 用这两条轴定位 REINFORCE、Sarsa、Q-Learning、DQN、PPO、GRPO、CQL、DPO 等算法。

**先记住一句话**

On-policy / Off-policy 问的是：

> 这批样本是不是由“我要更新的策略”自己采出来的？

Online / Offline 问的是：

> 训练过程中，我还能不能继续让策略去采新样本？

这两个问题很像，但不是一回事。DQN 可以一边继续玩游戏、一边复用旧经验，所以它通常是 **online + off-policy**；DPO 可以在一份固定偏好数据上训练，所以它更接近 **offline 偏好优化**；PPO/GRPO 通常每轮都让当前模型生成新回答，再用这批回答更新模型，所以常被看作 **online + on-policy**。

## 为什么要单独讲“数据从哪里来”

前两节我们已经走过两条优化路线：

- 路线一学习 $Q(s,a)$：先给动作打分，再选高分动作。
- 路线二优化 $J(\theta)$：直接调整策略，让好动作更容易被选出来。

但真正开始训练时，还会冒出一个更实际的问题：**用来更新参数的那批样本，到底是谁产生的？**

看到一条训练样本：

$$
(s_t,a_t,r_t,s_{t+1})
$$

我们不能只问它奖励是多少，还要问三个问题：

1. **谁做出了动作 $a_t$？** 是当前策略、旧策略、专家、人类日志、teacher model，还是一个带探索噪声的策略？
2. **现在要更新谁？** 是同一个策略，还是另一个正在学习的目标策略？
3. **训练时还能补数据吗？** 如果发现数据覆盖不够，能不能再去环境里采一批？

这三个问题决定了算法能不能复用旧数据、要不要做重要性修正、会不会遇到分布外动作过估计，也决定了为什么 PPO 的 rollout 不能无限用、为什么 DQN 不是 offline RL、为什么 DPO 不能简单当成普通分类。

本节要建立的地图很简单：

| 判断问题                     | 对应概念               | 看什么                   |
| ---------------------------- | ---------------------- | ------------------------ |
| 这批数据由哪个策略产生？     | On-policy / Off-policy | 行为策略与目标策略的关系 |
| 训练时还能不能继续采新数据？ | Online / Offline       | 数据集是否继续增长       |

读这一节时，最好不要先从算法名字判断，而是先问这两个问题。名字只是结果，数据关系才是原因。

## 两个角色：行为策略与目标策略

先定义两个最重要的角色。

$$
\mu(a\mid s)=\text{行为策略：产生训练数据的策略}
$$

**行为策略（Behavior Policy）** 负责“出去跑数据”。它决定了为什么训练集中会出现状态 $s$ 下的动作 $a$。

在经典 RL 里，行为策略可能是：

- 带 $\epsilon$-greedy 探索的策略；
- replay buffer 里某个旧版本的策略；
- 人类专家或脚本策略；
- 混合了很多历史版本的策略。

在大模型后训练里，行为策略可能是：

- 当前模型；
- 某个旧 checkpoint；
- teacher model；
- 人类写下来的回答或偏好日志；
- 已经固定下来的 preference pairs。

再看另一个角色：

$$
\pi_\theta(a\mid s)=\text{目标策略：正在被评估、学习或改进的策略}
$$

**目标策略（Target Policy）** 是我们真正想学好的策略，参数是 $\theta$。它回答的问题不是“数据当初怎么来的”，而是“我们希望最后学成什么样”。

例如：

- 训练 Sarsa 时，采样和更新都围绕同一个带探索的策略进行。
- 训练 Q-Learning 时，实际采样可以用 $\epsilon$-greedy，但更新目标里用的是 $\max_{a'}Q(s',a')$，也就是朝最优贪心策略学习。
- 训练 PPO/GRPO 时，采样模型通常是当前策略的一个旧快照 $\pi_{\theta_{\text{old}}}$，更新对象是新的 $\pi_\theta$。
- 训练 DPO 时，数据来自固定偏好对，更新对象是当前语言模型策略。

只要把这两个角色分开，后面的概念就会顺很多。

## 第一条轴：On-policy 与 Off-policy

On-policy / Off-policy 判断的是：**一次参数更新所用的数据，和这次要更新的目标策略，是不是来自同一个策略分布。**

### On-policy：用自己的数据更新自己

如果训练数据由当前正在评估或改进的策略产生，或者由一个足够接近当前策略的旧版本产生，就称为 **on-policy** 学习。

可以粗略写成：

$$
\mu(a\mid s)\approx \pi_{\theta_{\text{collect}}}(a\mid s)
$$

这里的 $\pi_{\theta_{\text{collect}}}$ 是采样时用的策略。注意这个写法用了“约等于”，不是严格等号。原因是：在实际训练中，数据通常先由旧参数采出来，然后我们才用它更新成新参数。只要新旧策略的差距被控制住，这批数据仍然可以按 on-policy 的思路使用。

这也是 PPO 可以在同一批 rollout 上训练多个 epoch 的原因。PPO 不是说“样本必须只用一次”，而是说：**当你复用这批样本时，新策略不能离采样策略太远。**

PPO 用下面这个比率衡量新旧策略的偏离：

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
$$

如果 $r_t(\theta)$ 离 1 很远，说明当前策略已经和当初采样的策略差很多。这个时候继续把旧 rollout 当成“当前策略数据”，梯度就会越来越不可信。所以 PPO 会用 clipping 或 KL 约束限制这种偏离。

**典型例子**

REINFORCE 用当前策略采样完整 episode，再用这些 episode 的回报更新同一个策略。Sarsa 在更新时使用实际执行的下一个动作 $a'$，这个动作仍然来自当前行为策略，所以也是 on-policy。PPO 和 GRPO 通常每轮用当前模型生成回答，再用这批回答更新当前模型，因此也按 on-policy 方法理解。

**主要特点**

| 方面     | On-policy                            |
| -------- | ------------------------------------ |
| 数据来源 | 当前策略，或足够接近当前策略的旧快照 |
| 数据复用 | 可以短期复用，但不能长期无约束复用   |
| 优点     | 采样分布和优化目标一致，梯度解释清楚 |
| 代价     | 样本效率较低，需要频繁采样           |
| 典型算法 | REINFORCE、Sarsa、PPO、GRPO          |

### Off-policy：可以用别人的数据更新自己

如果行为策略 $\mu$ 和目标策略 $\pi_\theta$ 可以不同，就称为 **off-policy** 学习。

可以写成：

$$
\mu(a\mid s)\neq \pi_\theta(a\mid s)
$$

这表示：数据可以由探索策略、旧策略、人类日志、专家策略、teacher model 或其他模型产生，而学习目标仍然是另一个策略。

Q-Learning 是最经典的 off-policy 方法。智能体实际采样时可以用 $\epsilon$-greedy 探索，但更新目标使用的是：

$$
Q(s,a)\leftarrow Q(s,a)+\alpha
\left[
r+\gamma\max_{a'}Q(s',a')-Q(s,a)
\right]
$$

这里的 $\max_{a'}Q(s',a')$ 不关心下一步实际执行了哪个动作，而是假设未来会选择当前估计下价值最高的动作。也就是说，采样可以带探索，但学习目标朝着贪心最优策略走。Watkins 和 Dayan 的 Q-learning 论文证明，在表格情形、学习率合适且状态动作被充分访问的条件下，这类更新可以收敛到最优动作价值。[^watkins1992]

DQN 把 Q-Learning、经验回放和深度网络结合起来。replay buffer 里的样本来自不同时间点的旧策略，但它们会被反复抽出来训练当前 Q 网络。[^mnih2015]

**主要特点**

| 方面     | Off-policy                                |
| -------- | ----------------------------------------- |
| 数据来源 | 可以来自当前目标策略之外的策略            |
| 数据复用 | 可以较充分地复用旧数据                    |
| 优点     | 样本效率高，适合 replay buffer 和日志数据 |
| 代价     | 要处理分布差异、覆盖不足和估计方差        |
| 典型算法 | Q-Learning、DQN、SAC、TD3、CQL            |

### Off-policy 的关键条件：覆盖性

Off-policy 不等于“什么旧数据都能随便用”。它至少需要一个基本条件：**行为策略采出来的数据，要覆盖目标策略想学习的状态和动作。**

如果存在某个 $(s,a)$：

$$
\pi_\theta(a\mid s)>0
\quad\text{但}\quad
\mu(a\mid s)=0
$$

那么就会出问题。重要性采样比率

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}{\mu(a_t\mid s_t)}
$$

会出现除以 0；即使没有严格除以 0，只要 $\mu(a_t\mid s_t)$ 很小，$\rho_t$ 也可能非常大，导致估计方差爆炸。

更直观地说：如果数据里从来没有出现过某个动作，算法就很难可靠判断这个动作会带来什么后果。函数近似器还可能在这些数据覆盖不到的区域胡乱外推。Sutton 和 Barto 在讨论 off-policy Monte Carlo control 时专门强调了这个 coverage 要求。[^sutton-barto] Precup、Sutton 和 Singh 的 off-policy policy evaluation 工作也把“从一个策略的数据中评估另一个策略”作为核心设定，并自然引出了重要性采样。[^precup2000]

所以，off-policy 的能力来自数据复用，但风险也来自数据复用：你复用得越远，就越需要处理分布差异。

## 第二条轴：Online 与 Offline

Online / Offline 判断的是另一个问题：**训练过程中，数据集还能不能继续增长。**

它不关心行为策略和目标策略是不是同一个，只关心你能不能继续采新数据。

### Online RL：训练时还能继续采样

如果训练过程中仍然可以与环境交互，并持续把新样本加入数据集，就称为 **Online RL**。

形式上：

$$
\mathcal{D}_{k+1}
=\mathcal{D}_k\cup\{\tau_k\}
$$

这里的环境不一定是游戏模拟器。对大模型来说，环境交互可以是：给模型一个 prompt，让它生成回答；让奖励模型、规则评测、人类或另一个模型给分；甚至让 agent 调用工具完成一轮任务。

Online 只说明“还能继续采样”，并不说明采样策略和目标策略是否相同。因此 Online RL 可以是 on-policy，也可以是 off-policy。

**例子**

- PPO/GRPO：每轮用当前模型生成新 rollout，再用这些 rollout 更新模型，通常是 online + on-policy。
- DQN：训练时继续和环境交互，同时把旧经验放进 replay buffer 反复使用，通常是 online + off-policy。

注意：DQN 的 replay buffer 里有很多旧数据，但这不让它变成 offline RL。只要训练过程中还在继续采新经验，数据集还在增长，它就是 online。

### Offline RL：训练时不能再采样

如果训练开始前数据集已经固定，训练过程中不能再与环境交互，也不能再用当前策略生成新样本，就称为 **Offline RL**。

形式上：

$$
\mathcal{D}=\mathcal{D}_{\text{fixed}}
$$

Offline RL 的核心约束是：算法只能从已有数据中学习，不能通过新的探索修正数据盲区。Levine 等人的 tutorial 将其描述为使用预先收集的数据、没有额外 online data collection 的 RL。[^levine2020] Prudencio 等人的 survey 也强调，off-policy 方法通常仍然可以继续采样，而 offline 方法的关键限制是训练阶段不再采样。[^prudencio2023]

**例子**

自动驾驶、医疗推荐和机器人控制中，直接让未成熟策略在线试错可能不可接受，因此常常只能使用历史日志或专家数据。大模型对齐中，DPO/IPO 等方法通常使用固定偏好数据训练；在这种设定下，偏好对已经提前收集好，fine-tuning 阶段不再每轮用当前模型采样并重新标注。DPO 论文强调，它把 RLHF 目标改写成分类式损失，并且 fine-tuning 阶段不需要继续从语言模型采样。[^dpo2023]

这里要加一个条件：如果某个系统在训练过程中持续让当前模型生成新回答、再让人类或 judge 产生新偏好标签，那它就不再是固定数据意义上的 offline 设定。是否 offline，要看训练阶段数据是否继续增长，而不是看损失函数长得像不像监督学习。

### Offline RL 的主要风险：看不见的地方容易高估

Offline RL 的困难不只是“没有环境接口”，而是没有办法用新探索纠正旧数据的盲区。

假设历史数据只覆盖了保守驾驶：慢速、少变道、很少接近复杂路口。现在目标策略想学得更大胆一点，但数据里几乎没有这些动作的后果。价值函数如果在这些区域给出高分，我们很难立刻发现它错了，因为训练期间不能让策略真的去试。

CQL 论文指出，标准 off-policy RL 方法直接用于 offline 数据时，容易因为数据分布与学习策略分布不一致而产生 Q 值过估计。[^cql2020] 这也是 CQL、IQL 等 offline RL 方法要引入保守估计、策略约束或分布约束的原因。

## 四象限地图

把两条轴放在一起，就得到下面这张表。

| 数据形态                 | 含义                                                           | 典型算法或场景                                  | 主要风险                           |
| ------------------------ | -------------------------------------------------------------- | ----------------------------------------------- | ---------------------------------- |
| **Online + On-policy**   | 训练时持续采新数据，并主要用当前策略或近邻旧快照的数据更新自己 | REINFORCE、Sarsa、PPO、GRPO                     | 样本效率低，旧 rollout 很快过期    |
| **Online + Off-policy**  | 训练时继续采样，但更新时允许复用旧策略或探索策略的数据         | Q-Learning、DQN、SAC、TD3                       | 要处理行为策略与目标策略的分布差异 |
| **Offline + Off-policy** | 不再采新数据，只能在固定历史数据上学习当前目标策略             | CQL、IQL、离线 Q 学习、固定偏好数据上的 DPO/IPO | 容易在数据覆盖不到的区域外推       |
| **Offline + On-policy**  | 数据固定，同时又要求数据代表当前目标策略                       | 固定策略评估、极少更新的模仿式设定              | 策略一更新，数据就不再代表新策略   |

这张表里最容易误解的是右下角。

严格的 on-policy 控制通常需要当前策略自己采样。一旦数据固定，而策略还要继续更新，数据很快就会变成旧策略的数据。因此 **offline + on-policy** 在控制问题里不常见；它更像一种边界情形，例如只评估一套固定策略，或者只做很小范围的模仿更新。

两个结论要特别记住：

1. **Off-policy 不等于 Offline。** DQN 是 off-policy，但通常仍然是 online，因为它会继续与环境交互并把新经验加入 replay buffer。
2. **Offline 不等于监督学习。** SFT 更像普通监督学习；但如果要从固定数据中估计长期回报，或者从偏好数据中推导策略改进方向，就会重新遇到 RL 的分布偏移、覆盖性和外推问题。

## 大模型 RL 中的对应关系

大模型训练里的“环境交互”通常比经典控制更贵。一次交互可能包括生成完整回答、调用工具、运行评测、用奖励模型打分，甚至让人类或另一个模型比较偏好。因此，数据形态会直接影响训练系统设计。

| 方法       | 数据视角                    | 说明                                                   |
| ---------- | --------------------------- | ------------------------------------------------------ |
| DQN 路线   | Online + Off-policy         | 继续采样，同时复用 replay buffer                       |
| PPO / GRPO | 通常是 Online + On-policy   | 当前模型或近邻旧快照生成回答，再用这批回答更新当前模型 |
| DPO / IPO  | 通常更接近 Offline 偏好优化 | 使用固定偏好数据训练，不在每轮更新中重新采样和标注     |
| Agentic RL | 依任务而定                  | 多轮工具调用让采样更贵，数据复用和分布偏移更关键       |

这里有两个容易漏掉的条件。

第一，PPO/GRPO 的 prompt 集合可能是固定的，但只要回答是每轮由当前模型重新生成，回答这部分数据就在增长；从策略动作数据的角度看，它仍然是 online 的。

第二，DPO 的损失形式看起来像分类，但它优化的是策略分布相对参考模型的移动方向。它可以用监督学习形式实现，却仍然继承了 RLHF 中“移动策略分布”的问题意识。固定偏好数据越窄，最终策略能可靠改进的范围就越受限制。

## 常见误解

### 误解一：On-policy 就是“样本只能用一次”

不是。PPO 说明 on-policy 数据可以在小范围内复用。关键不在于“用了几次”，而在于新策略有没有离采样策略太远。

如果完全不复用，会低估 PPO 的样本利用率；如果长期复用旧 rollout，又会让数据分布和当前策略严重不匹配，梯度信号开始失真。

### 误解二：Off-policy 就是“旧数据随便用”

也不是。Off-policy 能用旧数据，是有覆盖性条件的。行为策略必须给目标策略可能选择的状态动作提供足够样本。

如果覆盖不足，重要性采样比率可能变得极大，价值函数也可能在数据看不到的区域胡乱外推。在深度网络这类函数近似条件下，这种外推会让训练不稳定，甚至发散。

### 误解三：Off-policy 和 Offline 是一回事

DQN 虽然是 off-policy，但它通常还能继续采样；offline RL 则不能继续补数据。

把这两者混在一起，会误以为普通 replay buffer 足以解决 offline RL。事实上，replay buffer 只是复用旧经验；offline RL 的难点是固定数据集无法覆盖目标策略未来想去的所有区域。

### 误解四：Offline RL 更安全，所以更简单

Offline RL 避免了在线试错风险，但它把难点转移到了数据覆盖和分布偏移上。

如果数据只覆盖保守行为，模型很难可靠学出更激进但高回报的策略。如果数据混杂了多种策略，模型还要分清“这个动作本身好”和“只是某个采样策略经常这么做”。如果奖励或偏好标签有偏，模型也没有在线探索机会来纠正。

### 误解五：DPO 只是普通分类

DPO 的实现形式类似分类损失，但目标不是单纯预测哪个回答被偏好，而是根据偏好数据移动策略分布，并通过参考模型约束移动幅度。

所以理解 DPO 时，不能只看 loss 的样子，还要看数据来源、参考模型约束，以及固定偏好数据覆盖了哪些回答类型。

## 小结

本节最重要的是两条轴：

1. **On-policy / Off-policy 判断策略关系。** 行为策略和目标策略一致或足够接近，就是 on-policy；允许不一致，就是 off-policy。
2. **Online / Offline 判断数据是否继续增长。** 训练时还能采样，就是 online；只能使用固定数据集，就是 offline。
3. **Off-policy 不等于 Offline。** DQN 是 online + off-policy；CQL/IQL 才是典型 offline RL 方法。
4. **Off-policy 要看覆盖性。** 目标策略想学的状态动作，必须在行为策略数据中有足够覆盖。
5. **Offline RL 的核心困难是分布偏移。** 固定数据集无法覆盖所有目标策略可能访问的状态和动作。
6. **大模型后训练也受这两条轴约束。** PPO/GRPO 依赖当前模型采样；DPO/IPO 通常更接近固定偏好数据上的 offline 优化。

到这里，我们已经知道算法可以沿着 $Q(s,a)$ 或 $J(\theta)$ 两条路线优化，也知道训练数据可以来自当前策略、旧策略、日志数据或固定数据集。下一节继续追问一个更根本的问题：所有这些算法都在最大化奖励，但**奖励本身是谁写的？写歪了会怎样？**

← 上一节：[路线二：J(θ)——直接优化策略](./policy-objective) | 下一节：[奖励函数设计——优化的目标从哪来？](./reward-design)

## 课后检查

1. 为什么 Q-Learning 是 off-policy，而 Sarsa 是 on-policy？
2. DQN 为什么不是 offline RL？
3. 为什么 PPO 可以在同一批 rollout 上训练多个 epoch，但不能无限复用这些 rollout？
4. Offline RL 为什么容易出现 Q 值过估计？
5. DPO 为什么形式上像分类，但仍然需要从策略优化角度理解？

## 参考文献

[^sutton-barto]: Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press. 参见第 5.5、5.7、6.4、6.5 章关于 off-policy prediction、off-policy control、Sarsa 和 Q-learning 的讨论。在线 PDF 镜像：<https://web.stanford.edu/class/psych209/Readings/SuttonBartoRL.pdf>

[^precup2000]: Precup, D., Sutton, R. S., & Singh, S. (2000). Eligibility Traces for Off-Policy Policy Evaluation. _Proceedings of ICML 2000_, 759-766. <https://web.eecs.umich.edu/~baveja/Papers/OffPolicy.pdf>

[^watkins1992]: Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. _Machine Learning_, 8, 279-292. <https://www.gatsby.ucl.ac.uk/~dayan/papers/wd92.html>

[^mnih2015]: Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. _Nature_, 518, 529-533. <https://doi.org/10.1038/nature14236>

[^ppo2017]: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. _arXiv:1707.06347_. <https://arxiv.org/abs/1707.06347>

[^levine2020]: Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. _arXiv:2005.01643_. <https://arxiv.org/abs/2005.01643>

[^prudencio2023]: Prudencio, R. F., Maximo, M. R. O. A., & Colombini, E. L. (2023). A Survey on Offline Reinforcement Learning: Taxonomy, Review, and Open Problems. _IEEE Transactions on Neural Networks and Learning Systems_. <https://arxiv.org/abs/2203.01387>

[^cql2020]: Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-Learning for Offline Reinforcement Learning. _NeurIPS 2020_. <https://papers.nips.cc/paper_files/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html>

[^dpo2023]: Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. _arXiv:2305.18290_. <https://arxiv.org/abs/2305.18290>
