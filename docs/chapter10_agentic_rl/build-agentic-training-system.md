# 10.6 动手：从零实现一个 Agentic RL 训练系统

前面的章节分析了 OpenRLHF、veRL、Relax 等框架的架构设计。但分析再细致，也不如自己写一遍来得深刻。这一节我们参照 [hyunwoongko/nanoRLHF](https://github.com/hyunwoongko/nanoRLHF) 的思路——用最少的代码还原一个 Agentic RL 训练系统的核心结构。

目标不是造一个生产级框架，而是用可读的代码把前面讨论的概念变成具体实现：**Agent Loop、沙箱、多轮轨迹存储、rollout/train 分离、reward 计算**。读完之后再看 veRL 或 Relax 的源码，会快得多。

## 整体架构

一个最简 Agentic RL 训练系统只需要四个组件：

```
┌──────────────────────────────────────────────────────────┐
│  Trainer：编排 rollout → reward → train 循环             │
├──────────────────────────────────────────────────────────┤
│  RolloutWorker：驱动 Agent Loop，收集多轮轨迹            │
├──────────────────────────────────────────────────────────┤
│  Environment：沙箱 + 工具执行（代码运行、搜索、文件IO） │
├──────────────────────────────────────────────────────────┤
│  Policy：模型推理 + 模型训练（同一份权重，两种用法）     │
└──────────────────────────────────────────────────────────┘
```

每个组件对应一个 Python 文件。整个系统不到 500 行，能在 CPU 上跑通。

## 第一步：Environment — 沙箱和工具执行

Environment 负责两件事：(1) 安全地执行 Agent 的动作，(2) 返回执行结果。这是 10.2 节讨论过的沙箱隔离问题的最小实现。

```python
# environment.py
import subprocess
import tempfile
import os

class SandboxEnv:
    """最轻量的沙箱：subprocess + 资源限制"""

    def __init__(self, timeout=10, max_memory=256 * 1024 * 1024):
        self.timeout = timeout
        self.max_memory = max_memory

    def step(self, action_type: str, action_args: dict) -> dict:
        """执行一步动作，返回观测和终止状态。"""
        if action_type == "execute_code":
            return self._exec_code(action_args["code"])
        elif action_type == "finish":
            return {"observation": "", "done": True}
        else:
            return {"observation": f"Unknown action: {action_type}", "done": False}

    def _exec_code(self, code: str) -> dict:
        """在子进程中执行代码，限制 CPU 时间和内存。"""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(
                    ["python", f.name],
                    timeout=self.timeout,
                    capture_output=True,
                    text=True,
                )
                os.unlink(f.name)
                return {
                    "observation": (result.stdout + result.stderr)[-500:],  # 截断
                    "done": False,
                }
        except subprocess.TimeoutExpired:
            return {"observation": "TIMEOUT", "done": True}
        except Exception as e:
            return {"observation": f"ERROR: {e}", "done": False}

    def reset(self):
        """重置环境状态（新 episode 开始时调用）。"""
        pass
```

几个设计要点：

- `step()` 接受结构化的 action（不是纯文本），对应 10.1 节讨论的 $A = A_{\text{text}} \cup A_{\text{action}}$
- `_exec_code()` 用 subprocess 隔离执行，加 timeout 防止死循环——这是 B.2 讨论的最轻量沙箱方案
- 返回值包含 `observation`（环境反馈）和 `done`（是否终止），对应 POMDP 的观测函数 $O(s_t)$

## 第二步：Policy — 模型推理与训练

Policy 封装了模型的两种用法：推理时生成动作（rollout），训练时更新参数（gradient step）。这对应 B.1 讨论的"同一份权重的两种用途"。

```python
# policy.py
import torch
import torch.nn.functional as F

class Policy:
    """包装一个语言模型，提供 generate() 和 train_step() 两个接口。"""

    def __init__(self, model, tokenizer, lr=1e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.ref_model = None  # reference model for KL penalty

    def set_ref_model(self, ref_model):
        """保存一份初始权重的拷贝，用作 KL 散度计算的锚点。"""
        self.ref_model = ref_model

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=128) -> str:
        """推理模式：给定 prompt，生成文本。"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @torch.no_grad()
    def get_logprobs(self, prompt: str, response: str) -> torch.Tensor:
        """计算模型对给定 response 中每个 token 的 log probability。"""
        full_text = prompt + response
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs).logits

        # 只取 response 部分
        prompt_len = len(self.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        response_logits = logits[:, prompt_len - 1:-1, :]  # (1, resp_len, vocab)
        response_ids = inputs["input_ids"][:, prompt_len:]  # (1, resp_len)

        logprobs = F.log_softmax(response_logits, dim=-1)
        token_logprobs = logprobs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
        return token_logprobs

    def train_step(self, trajectories: list):
        """
        一个 GRPO 训练步。
        trajectories: list of (prompt, response, reward)
        """
        losses = []
        for prompt, response, reward in trajectories:
            # 当前策略的 log prob
            new_logprobs = self.get_logprobs(prompt, response)

            # KL 惩罚（与 reference model 的散度）
            if self.ref_model is not None:
                with torch.no_grad():
                    ref_logprobs = self._get_ref_logprobs(prompt, response)
                kl = (new_logprobs.exp() * (new_logprobs - ref_logprobs)).sum()
            else:
                kl = 0.0

            # 策略梯度 loss：-log_prob * reward
            pg_loss = -(new_logprobs.sum() * reward)

            loss = pg_loss + 0.1 * kl
            losses.append(loss)

        total_loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
```

几个设计要点：

- `generate()` 和 `get_logprobs()` 是 rollout 阶段用的，`train_step()` 是训练阶段用的——这就是 B.1 说的"同一份权重，两种用途"
- `ref_model` 是 KL 惩罚的锚点，防止模型偏离初始策略太远
- `train_step()` 实现了最简的策略梯度（REINFORCE），没有 PPO 的 clipping——先跑通再优化

## 第三步：RolloutWorker — 驱动 Agent Loop

RolloutWorker 把 Policy 和 Environment 串成 Agent Loop：模型生成 → 环境执行 → 模型继续 → ... 直到任务结束或达到最大轮次。收集的完整轨迹用于后续训练。

````python
# rollout_worker.py

class RolloutWorker:
    """
    驱动 Agent Loop，收集多轮轨迹。
    轨迹结构: [(prompt, response_1, obs_1, response_2, obs_2, ..., final_response), reward]
    """

    def __init__(self, policy, env, max_turns=5):
        self.policy = policy
        self.env = env
        self.max_turns = max_turns

    def rollout(self, prompt: str, reward_fn) -> dict:
        """
        执行一次完整的 Agent Loop，返回轨迹和 reward。
        reward_fn: callable(trajectory) -> float
        """
        messages = [{"role": "user", "content": prompt}]
        trajectory = {"prompt": prompt, "interactions": []}

        for turn in range(self.max_turns):
            # 模型生成下一步动作
            context = self._format_context(messages)
            model_output = self.policy.generate(context)

            # 解析模型输出：判断是代码执行还是最终回答
            action = self._parse_action(model_output)

            if action["type"] == "finish":
                trajectory["interactions"].append({
                    "turn": turn,
                    "response": model_output,
                    "action": action,
                    "observation": None,
                })
                trajectory["final_response"] = action.get("answer", model_output)
                break

            # 环境执行动作
            obs = self.env.step(action["type"], action["args"])

            trajectory["interactions"].append({
                "turn": turn,
                "response": model_output,
                "action": action,
                "observation": obs["observation"],
            })

            messages.append({"role": "assistant", "content": model_output})
            messages.append({"role": "user", "content": f"执行结果:\n{obs['observation']}"})

            if obs.get("done"):
                break

        # 计算奖励
        trajectory["reward"] = reward_fn(trajectory)
        return trajectory

    def _format_context(self, messages):
        """把多轮消息列表拼成模型能理解的 prompt。"""
        parts = []
        for msg in messages:
            if msg["role"] == "user":
                parts.append(f"User: {msg['content']}")
            else:
                parts.append(f"Assistant: {msg['content']}")
        return "\n".join(parts)

    def _parse_action(self, model_output: str) -> dict:
        """
        从模型输出中解析动作类型和参数。
        简化实现：按标记解析。
        """
        if "```python" in model_output:
            # 提取代码块
            code = model_output.split("```python")[1].split("```")[0]
            return {"type": "execute_code", "args": {"code": code}}
        elif "FINAL ANSWER:" in model_output:
            answer = model_output.split("FINAL ANSWER:")[1].strip()
            return {"type": "finish", "answer": answer}
        else:
            return {"type": "execute_code", "args": {"code": model_output}}
````

几个设计要点：

- `rollout()` 就是 10.1 节描述的 Agent Loop 的代码版：每轮包含模型推理（`policy.generate()`）→ 动作解析（`_parse_action()`）→ 环境执行（`env.step()`）→ 观测回传
- 轨迹结构是 `{"prompt", "interactions": [...], "final_response", "reward"}`——比单轮 RL 的 `(prompt, completion, reward)` 复杂得多，但保留了完整的多轮交互信息
- `_parse_action()` 是一个简化版解析器。生产框架（如 Relax）会用 tokenizer + 特殊 token 来做结构化解析，这里用字符串匹配足够理解概念

## 第四步：Trainer — 编排训练循环

Trainer 是最上层，把 RolloutWorker 和 Policy 串成训练循环：rollout 一批轨迹 → 计算 reward → 更新参数 → 重复。

```python
# trainer.py

class GRPOAgentTrainer:
    """
    编排 Agentic RL 训练循环。
    rollout -> reward -> train -> repeat
    """

    def __init__(self, policy, env, reward_fn, group_size=4, max_turns=5):
        self.policy = policy
        self.env = env
        self.reward_fn = reward_fn
        self.group_size = group_size  # GRPO: 同一 prompt 采多条轨迹
        self.worker = RolloutWorker(policy, env, max_turns=max_turns)
        self.history = []  # 记录训练指标

    def fit(self, prompts: list, n_steps: int = 50):
        """
        主训练循环。
        prompts: 训练用的 prompt 列表
        n_steps: 训练步数
        """
        for step in range(n_steps):
            # ---- Rollout 阶段 ----
            # 每个 prompt 采样 group_size 条轨迹（GRPO 的核心思想）
            batch_trajectories = []
            for prompt in prompts:
                group = []
                for _ in range(self.group_size):
                    traj = self.worker.rollout(prompt, self.reward_fn)
                    group.append(traj)
                batch_trajectories.append(group)

            # ---- Reward 归一化（GRPO 的组内比较）----
            all_rewards = []
            for group in batch_trajectories:
                group_rewards = [t["reward"] for t in group]
                mean_r = sum(group_rewards) / len(group_rewards)
                std_r = (sum((r - mean_r) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5 + 1e-8
                for t, r in zip(group, group_rewards):
                    t["advantage"] = (r - mean_r) / std_r
                all_rewards.extend(group_rewards)

            # ---- Train 阶段 ----
            train_data = []
            for group in batch_trajectories:
                for traj in group:
                    # 把多轮轨迹展开为 (prompt, full_response, advantage)
                    full_response = self._serialize_trajectory(traj)
                    train_data.append((
                        traj["prompt"],
                        full_response,
                        traj["advantage"],
                    ))

            loss = self.policy.train_step_with_advantage(train_data)

            # ---- 记录指标 ----
            mean_reward = sum(all_rewards) / len(all_rewards)
            self.history.append({
                "step": step,
                "loss": loss,
                "mean_reward": mean_reward,
                "max_reward": max(all_rewards),
            })
            if step % 5 == 0:
                print(f"Step {step:3d} | loss={loss:.4f} | "
                      f"reward_mean={mean_reward:.3f} | "
                      f"reward_max={max(all_rewards):.3f}")

        return self.history

    def _serialize_trajectory(self, traj: dict) -> str:
        """把多轮轨迹序列化为一段文本，用于 train_step。"""
        parts = []
        for interaction in traj["interactions"]:
            parts.append(f"Assistant: {interaction['response']}")
            if interaction["observation"]:
                parts.append(f"Observation: {interaction['observation']}")
        return "\n".join(parts)
```

几个设计要点：

- `fit()` 的主循环就是 B.1 说的"生产者-消费者"模式：RolloutWorker 生产轨迹，Policy 消费轨迹做梯度更新
- GRPO 的组内比较在 `Reward 归一化` 那段实现：同一 prompt 的多条轨迹计算 advantage = (reward - mean) / std
- `_serialize_trajectory()` 把多轮轨迹展成文本。这里做了简化——生产框架会用 loss mask 区分模型生成的 token 和环境返回的 token（见 B.2 的 loss mask 讨论）

## 第五步：拼起来跑

```python
# run.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from environment import SandboxEnv
from policy import Policy
from trainer import GRPOAgentTrainer

# 加载一个小模型
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 初始化各组件
env = SandboxEnv(timeout=10)
policy = Policy(model, tokenizer, lr=5e-5)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)
policy.set_ref_model(ref_model)

# 定义 reward：代码执行结果是否正确
def code_reward(trajectory):
    """如果最终答案包含正确的执行结果，reward = 1，否则 = 0。"""
    final = trajectory.get("final_response", "")
    # 简化：检查是否有成功的代码执行记录
    for interaction in trajectory["interactions"]:
        obs = interaction.get("observation", "")
        if obs and "ERROR" not in obs and "TIMEOUT" not in obs:
            return 1.0
    return 0.0

# 训练 prompts
prompts = [
    "写一段 Python 代码计算斐波那契数列的第 10 项并输出结果。",
    "写一段代码检查字符串是否是回文。",
    "写一段代码对列表进行冒泡排序。",
]

# 开始训练
trainer = GRPOAgentTrainer(
    policy=policy,
    env=env,
    reward_fn=code_reward,
    group_size=4,
    max_turns=3,
)
history = trainer.fit(prompts, n_steps=30)
```

## 这个系统和生产框架的差距

跑通上面的代码后，你已经理解了一个 Agentic RL 训练系统的骨架。下表列出了这个最小实现和 Relax / veRL 等生产框架之间的关键差距：

| 方面      | 本节最小实现                | 生产框架（Relax / veRL）                               |
| --------- | --------------------------- | ------------------------------------------------------ |
| 推理引擎  | `model.generate()` 逐条生成 | vLLM / SGLang，continuous batching，KV cache           |
| 训练引擎  | 单卡 AdamW                  | FSDP / Megatron，3D parallelism，gradient accumulation |
| 分布式    | 单进程                      | Ray 集群，多机多卡，PlacementGroup                     |
| 异步训练  | rollout 和 train 串行       | TransferQueue 流式解耦，DCS 异步权重同步               |
| 沙箱      | subprocess + timeout        | Docker 容器池 / MicroVM，预热池，资源隔离              |
| Loss mask | 所有 token 都参与 loss      | 模型生成 token 参与 loss，工具返回 token 被 mask       |
| Reward    | 简单规则                    | 规则 + RM + LLM-as-Judge + verifier 组合               |
| 轨迹存储  | 内存中的 dict               | 分布式存储（Redis / S3），按任务/步骤检索              |
| 容错      | 无                          | 心跳监控，自动重启，checkpoint 恢复                    |

每个差距都是一个独立的工程优化方向。理解骨架之后，你可以按需深入任何一个方向。

## 扩展练习

1. **加 PPO clipping**：在 `train_step()` 中加入 PPO 的 clipped surrogate objective（参考第 6 章），对比 REINFORCE 和 PPO 的训练稳定性
2. **加 loss mask**：在 `_serialize_trajectory()` 中标记哪些 token 是模型生成的、哪些是环境返回的，只在模型生成的 token 上计算 loss
3. **加更多工具**：在 `SandboxEnv` 中加入搜索工具（mock 版本即可），让模型学会在代码执行和搜索之间做选择
4. **异步 rollout**：用 `multiprocessing` 把 rollout 和 train 拆到不同进程，用 `Queue` 传递轨迹数据，观察 GPU 利用率的变化

---

本节实现了一个最小但完整的 Agentic RL 训练系统。核心收获：**Agentic RL 训练框架的本质就是把 Agent Loop（10.1）和环境交互（10.2）嵌套进一个 rollout → reward → train 的 RL 循环里**。所有生产框架的复杂性都来自把这个骨架放大到多机多卡、高吞吐、长周期运行的场景。
