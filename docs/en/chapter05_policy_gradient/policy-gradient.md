---
title: 5.1 Why We Need Policy Gradients
---

# 5.1 Why We Need Policy Gradients

## Section Overview

**Key takeaways**

- Review the core idea of DQN from Chapter 4: learn $Q(s,a)$, then use $\arg\max$ to select actions.
- Understand the fundamental limitation of value-based methods: they can only handle a finite set of discrete actions.
- Understand why policy-based methods learn $\pi_\theta(a|s)$ directly, and the essential differences between the two routes in terms of action spaces, exploration mechanisms, and data efficiency.

## What DQN Got Right

The DQN from Chapter 4 follows a clear recipe: approximate $Q(s,a)$ with a neural network, assign a score to each action, then pick the highest-scoring one with $\arg\max_a Q(s,a)$. The underlying logic is: rather than learning "what to do" directly, first learn "how much each action is worth" and then select the best. The policy is implicit -- it is hidden inside the $\arg\max$ over the $Q$-value table.

Consider a concrete moment in CartPole. Suppose the current state is $s = [0.05,\; 0.1,\; -0.02,\; 0.3]$ (position, velocity, pole angle, angular velocity). The DQN network performs one forward pass on this state and outputs two $Q$ values:

| Action     | $Q(s,a)$ |
| ---------- | -------- |
| Push left  | $0.8$    |
| Push right | $1.2$    |

$\arg\max$ simply compares them and returns the action with the highest value:

$$
a^* = \arg\max_a Q(s,a) = \arg\max\{0.8,\; 1.2\} = \text{push right}.
$$

The key prerequisite is that the set of actions is finite and small, so we can compute a $Q$ value for each one and compare them all. CartPole has only 2 actions, so we compare 2 numbers; LunarLander has 4 actions, so we compare 4 numbers. Scale up to 10, 100, or even 1000 actions, and $\arg\max$ still works -- it just requires computing a few more $Q$ values and doing more comparisons, but the cost grows linearly with no fundamental difficulty.

| Number of actions | $Q$ values to compute | $\arg\max$ comparisons | Feasibility       |
| ----------------- | --------------------- | ---------------------- | ----------------- |
| 2                 | 2                     | 1                      | Easy              |
| 4                 | 4                     | 3                      | Easy              |
| 1000              | 1000                  | 999                    | Feasible          |
| $10^6$            | $10^6$                | $10^6 - 1$             | Feasible but slow |
| $\infty$          | $\infty$              | $\infty$               | Impossible        |

The last row is where the problem lies. When the action space is continuous, there are infinitely many actions. It is impossible to compute a $Q$ value for each one, let alone find the maximum among infinitely many values.

## Where $\arg\max$ Breaks Down

$\arg\max$ requires comparing the $Q$ values of all actions. As long as the number of actions is finite, this poses no problem. But many real-world tasks have continuous action spaces with infinitely many actions.

### Robot Arm: Curse of Dimensionality

Controlling a robot arm is a canonical example. The shoulder, elbow, and wrist joints each have multiple degrees of freedom, and each degree of freedom applies a continuous torque $\tau \in [-10, 10]$. With 6 joints, the action space is $[-10, 10]^6$ -- infinitely many points in a six-dimensional continuous space. It is impossible to compute a $Q$ value for every point, let alone find the $\arg\max$ over infinitely many points.

A natural idea is to discretize the continuous space and then apply $\arg\max$. For instance, allow only 100 torque values per joint, approximating the continuous space with a finite grid. With 6 joints each taking 100 values, the total number of actions is:

$$
N = 100^6 = 10^{12}.
$$

$10^{12}$ actions, each requiring one $Q$-value computation. If a single forward pass of the neural network takes $1\mu\text{s}$ ($10^{-6}$ seconds), then **a single action selection** requires:

$$
T = 10^{12} \times 10^{-6}\text{s} = 10^6\text{s} \approx 11.6 \text{ days}.
$$

11.6 days just to choose one action. By contrast, a policy network needs only one forward pass -- feed the state into the network and directly output the action vector, taking roughly $1\text{ms}$:

| Method               | Computation per action selection                     | Time                 |
| -------------------- | ---------------------------------------------------- | -------------------- |
| DQN + discretization | $10^{12}$ forward passes                             | $\approx 11.6$ days  |
| Policy network       | 1 forward pass, directly outputs $\mu = f_\theta(s)$ | $\approx 1\text{ms}$ |

And this is only 6 joints with each joint discretized to 100 values. With more joints or finer precision requirements, the number of discretized actions grows exponentially. This is the **curse of dimensionality**: each additional joint multiplies the total number of actions by a constant factor.

### LLM Generation: Probability Distributions Beat Greedy Decoding

Text generation in large language models faces a related issue. At each step, the model must select one token from tens of thousands. Suppose the vocabulary size is 50,000. $\arg\max$ itself is not difficult here -- comparing 50,000 numbers is computationally tractable. The problem is not computational feasibility, but generation quality.

Suppose the model is generating the next token and has output the following probabilities:

| token | $P(\text{token} \mid \text{context})$ |
| ----- | ------------------------------------- |
| "is"  | $0.40$                                |
| "are" | $0.25$                                |
| "was" | $0.15$                                |
| "be"  | $0.10$                                |
| ...   | ...                                   |

$\arg\max$ (greedy decoding) always selects the highest-probability "is". If the probability distributions at several subsequent positions are similar, greedy decoding will repeatedly output the same token. Sampling from the distribution, on the other hand, gives a 25% chance of selecting "are", a 15% chance of selecting "was" -- this randomness is precisely what fluent text generation requires. A policy network naturally outputs a probability distribution $\pi_\theta(a|s)$, and sampling is the generation process itself.

## Learning the Policy Directly

Since "score first, then select" does not work, take a different route: **skip $Q$ values and learn the policy $\pi_\theta(a|s)$ directly**. Instead of asking "how much is each action worth?", learn directly "what to do in which situation."

This is exactly the core idea of [Route 2: the policy objective $J(\theta)$](../chapter03_mdp/policy-objective) from Chapter 3 -- define a policy objective function $J(\theta)$, then directly optimize the parameters $\theta$ to maximize $J(\theta)$.

The difference between the two routes can be clarified with an analogy: value-based methods are like a food critic who scores every dish and then picks the highest-rated one; policy-based methods are like an experienced chef who, without scoring, simply knows what dish to make given the ingredients and the occasion.

### What the Policy Network Outputs

The policy network $\pi_\theta(a|s)$ does not output an action score, but a probability distribution. Consider CartPole: given the input state $s = [0.05, 0.1, -0.02, 0.3]$, the network performs a forward pass and finally outputs the probability of each action through a Softmax layer:

$$
\pi_\theta(\text{left} \mid s) = 0.3, \quad \pi_\theta(\text{right} \mid s) = 0.7.
$$

| Symbol                        | Meaning                                                          |
| ----------------------------- | ---------------------------------------------------------------- |
| $\pi_\theta$                  | Policy network parameterized by $\theta$                         |
| $\pi_\theta(a \mid s)$        | Probability of selecting action $a$ in state $s$                 |
| $s = [0.05, 0.1, -0.02, 0.3]$ | Current state (position, velocity, pole angle, angular velocity) |
| $[0.3, 0.7]$                  | Action probability vector output by the network                  |

Action selection is done by **sampling**, not comparison. From the distribution $[0.3, 0.7]$, draw a sample: generate a uniform random number $u \in [0,1)$; if $u < 0.3$, push left; otherwise, push right. For example, with $u = 0.65$, since $0.65 > 0.3$, the action is "push right."

### Comparison with DQN

Given the same state, the two methods follow completely different paths:

**DQN path:** Network outputs $Q$ values $\to$ $\arg\max$ $\to$ deterministic action

$$
[0.8,\; 1.2] \;\xrightarrow{\arg\max}\; \text{push right} \quad (\text{always this one}).
$$

**Policy network path:** Network outputs probabilities $\to$ sampling $\to$ stochastic action

$$
[0.3,\; 0.7] \;\xrightarrow{\text{sample}}\; \begin{cases} \text{push left} & \text{probability } 0.3 \\ \text{push right} & \text{probability } 0.7 \end{cases}
$$

The most important distinction: once trained, DQN always outputs the same action for the same state (deterministic policy); a policy network may output different actions for the same state (stochastic policy). This stochasticity is not a flaw but a feature -- it naturally incorporates exploration without requiring a separate $\varepsilon$-greedy mechanism.

For continuous action spaces, the policy network switches to outputting the parameters of a Gaussian distribution. For example, a robot arm that needs to output torques for 6 joints: the policy network outputs a mean vector $\mu_\theta(s) \in \mathbb{R}^6$ and a standard deviation $\sigma_\theta(s) \in \mathbb{R}^6$, then samples the action from $\mathcal{N}(\mu_\theta(s),\; \text{diag}(\sigma_\theta^2(s)))$. No discretization, no $\arg\max$, one forward pass.

## Differences Between the Two Routes

|                          | Value-Based (DQN)                             | Policy-Based (Policy Gradient)                                     |
| ------------------------ | --------------------------------------------- | ------------------------------------------------------------------ |
| What it learns           | $Q(s,a)$: how much each action is worth       | $\pi_\theta(a\|s)$: what probability to assign each action         |
| Action selection         | $\arg\max_a Q(s,a)$ (pick the highest score)  | Sample from $\pi_\theta(\cdot\|s)$                                 |
| Policy form              | Deterministic (always pick the highest score) | Stochastic (outputs a probability distribution)                    |
| Action space             | Discrete only                                 | Discrete + continuous                                              |
| Exploration              | External ($\varepsilon$-greedy)               | Built-in (probability distribution naturally includes exploration) |
| Data reuse               | Off-policy (replay buffer can reuse old data) | On-policy (must use fresh data from the current policy)            |
| Variance                 | Low (TD targets are relatively stable)        | High (Monte Carlo returns fluctuate widely)                        |
| Representative algorithm | DQN (Chapter 4)                               | REINFORCE (this chapter) $\to$ PPO (Chapter 7)                     |

Key differences explained row by row.

**Action space** -- This is the primary criterion for choosing between the routes. DQN's $\arg\max$ is simply not computable in continuous spaces. Policy gradients output a probability distribution directly -- Softmax for discrete actions, Gaussian for continuous actions -- just swap the output layer.

**Exploration** -- DQN's policy is deterministic (always pick $\arg\max$), so exploration must be injected via $\varepsilon$-greedy (review: [the three components of DQN](../chapter04_dqn/dqn-components)). The $\varepsilon$ schedule must be tuned by hand: too large wastes experience, too small under-explores. Policy gradients naturally output a probability distribution, so exploration is built in -- if the network believes an action has a 30% chance of being worth trying, it will try it 30% of the time.

**Data reuse** -- This is the most practical engineering difference between the two routes. DQN is off-policy: the replay buffer stores old data that can be reused repeatedly for training. Policy gradients are on-policy: the $\mathbb{E}_{\pi_\theta}$ in the gradient estimator requires data generated by the current policy $\pi_\theta$. Once the policy updates, old data is invalidated. Data efficiency is inherently lower than DQN, and this is the biggest engineering weakness of policy gradients.

### Numerical Comparison on the Same Scenario

Walk through both routes on a concrete scenario. Setup: 3 states $\{s_1, s_2, s_3\}$, 2 actions $\{a_1, a_2\}$, discount factor $\gamma = 0.9$.

**DQN route: learn $Q$ values, select with $\arg\max$**

Suppose after training, DQN has learned the following $Q$ table:

| State | $Q(s, a_1)$ | $Q(s, a_2)$ |
| ----- | ----------- | ----------- |
| $s_1$ | $1.5$       | $2.3$       |
| $s_2$ | $0.8$       | $-0.4$      |
| $s_3$ | $3.1$       | $2.9$       |

Apply $\arg\max$ at each state:

$$
\pi(s_1) = \arg\max\{1.5,\; 2.3\} = a_2,
$$

$$
\pi(s_2) = \arg\max\{0.8,\; -0.4\} = a_1,
$$

$$
\pi(s_3) = \arg\max\{3.1,\; 2.9\} = a_1.
$$

The result is a deterministic policy table: every state always selects the same action. If exploration is needed, $\varepsilon$-greedy must be added on top. For example, with $\varepsilon = 0.1$, there is a 10% probability of choosing randomly:

| State | Probability of $a_1$          | Probability of $a_2$          |
| ----- | ----------------------------- | ----------------------------- |
| $s_1$ | $0.1 \times 0.5 = 0.05$       | $0.9 + 0.1 \times 0.5 = 0.95$ |
| $s_2$ | $0.9 + 0.1 \times 0.5 = 0.95$ | $0.1 \times 0.5 = 0.05$       |
| $s_3$ | $0.9 + 0.1 \times 0.5 = 0.95$ | $0.1 \times 0.5 = 0.05$       |

$\varepsilon$-greedy exploration is uniform: the 10% random exploration is split equally between $a_1$ and $a_2$. Even though $Q(s_3, a_2) = 2.9$ is very close to $Q(s_3, a_1) = 3.1$ (the two actions are nearly equally good), the exploration probability allocation is identical to $s_1$ where the gap is large.

**Policy gradient route: learn $\pi_\theta(a|s)$, select by sampling**

Suppose the policy network has learned the following probability distributions:

| State | $\pi(a_1 \mid s)$ | $\pi(a_2 \mid s)$ |
| ----- | ----------------- | ----------------- |
| $s_1$ | $0.2$             | $0.8$             |
| $s_2$ | $0.9$             | $0.1$             |
| $s_3$ | $0.55$            | $0.45$            |

At $s_3$, the policy network considers the two actions nearly equally good ($0.55$ vs $0.45$), so the exploration ratio is naturally high; at $s_1$ and $s_2$, one action clearly dominates, so exploration is naturally low. No manual $\varepsilon$ tuning is needed -- the probability distribution itself encodes "how much to explore."

Placing the key numbers from both routes side by side:

| Dimension               | DQN at $s_3$                                         | Policy gradient at $s_3$                               |
| ----------------------- | ---------------------------------------------------- | ------------------------------------------------------ |
| Network output          | $Q(s_3, a_1) = 3.1$, $Q(s_3, a_2) = 2.9$             | $\pi(a_1 \mid s_3) = 0.55$, $\pi(a_2 \mid s_3) = 0.45$ |
| Action selection        | $\arg\max\{3.1, 2.9\} = a_1$                         | Sampling: 55% chance $a_1$, 45% chance $a_2$           |
| Exploration             | External $\varepsilon$-greedy (uniform random)       | Built-in (adaptive probability distribution)           |
| Continuous action space | Not applicable (requires discretized $Q$-value grid) | Applicable (directly outputs Gaussian parameters)      |

The core difference is in the last row: DQN's $\arg\max$ confines the action space to a finite discrete set; policy gradients skip the step of "scoring every action" and directly output a probability distribution over "how to choose actions," removing the barrier of continuous action spaces entirely.

## The Two Routes Are Not Opposed

Each route has its strengths and weaknesses, but they are not mutually exclusive. Chapter 6's Actor-Critic will merge the two: a policy network for decision-making, a value network for variance reduction. Before that, however, we need to establish the mathematical foundations of the policy-based route.

The next section starts from the policy objective function, derives the policy gradient theorem, and introduces the REINFORCE algorithm: [REINFORCE Algorithm](./reinforce).
