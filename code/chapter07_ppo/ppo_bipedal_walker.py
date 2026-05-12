"""
第7章：用 Stable-Baselines3 的 PPO 训练 BipedalWalker-v3
——展示 PPO 在连续动作空间上的能力

运行方式：
    python ppo_bipedal_walker.py
    python ppo_bipedal_walker.py --total-timesteps 100000    # 快速验证
    python ppo_bipedal_walker.py --total-timesteps 2000000   # 充分训练

BipedalWalker-v3 的教学意义：
    1. 连续动作空间（4 维关节扭矩）—— DQN 无法直接处理，PPO 原生支持
    2. 24 维状态空间（10 个激光雷达 + 关节角度 + 速度）
    3. 比 LunarLander 更难，需要更长的训练时间
    4. 环境定义"解决"标准：100 个回合平均分 >= 300
"""

import argparse
import os
from pathlib import Path
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# 创建输出目录
os.makedirs("output", exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(description="PPO 训练 BipedalWalker-v3")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="总训练步数（默认 1000000）")
    return parser.parse_args()


# ==========================================
# 第一部分：自定义训练回调 —— 记录关键指标
# ==========================================
class TrainingMonitorCallback(BaseCallback):
    """
    自定义回调：在每次 rollout 结束后记录 PPO 的关键训练指标
    包括：回合奖励、策略熵、裁剪比例、近似 KL 散度
    """

    def __init__(self, check_freq=2048, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.entropy_list = []
        self.clip_fraction_list = []
        self.approx_kl_list = []
        self.timesteps_list = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        if self.num_timesteps % self.check_freq == 0 and self.num_timesteps > 0:
            logger = self.model.logger
            if hasattr(logger, "name_to_value"):
                name_to_value = logger.name_to_value

                entropy = name_to_value.get("train/entropy_loss", 0)
                clip_frac = name_to_value.get("train/clip_fraction", 0)
                approx_kl = name_to_value.get("train/approx_kl", 0)

                self.entropy_list.append(entropy)
                self.clip_fraction_list.append(clip_frac)
                self.approx_kl_list.append(approx_kl)
                self.timesteps_list.append(self.num_timesteps)

        return True


# ==========================================
# 第二部分：创建向量化环境
# ==========================================
args = parse_args()

print("=" * 50)
print("第7章：PPO 训练 BipedalWalker-v3")
print("=" * 50)

print("\n正在创建向量化环境（8 个并行环境）...")

def make_env():
    """环境工厂函数"""
    def _init():
        env = gym.make("BipedalWalker-v3")
        return env
    return _init

num_envs = 8
vec_env = DummyVecEnv([make_env() for _ in range(num_envs)])
print(f"已创建 {num_envs} 个并行环境")


# ==========================================
# 第三部分：配置 PPO 超参数
# ==========================================
print("\n配置 PPO 超参数...")

model = PPO(
    policy="MlpPolicy",       # 多层感知机策略
    env=vec_env,              # 向量化环境
    learning_rate=3e-4,       # 学习率
    n_steps=2048,             # 每次 rollout 采集的步数（每个环境）
    batch_size=256,           # 小批量大小（比 LunarLander 更大，提升稳定性）
    n_epochs=10,              # 每批数据的更新轮数
    clip_range=0.2,           # PPO 裁剪范围
    ent_coef=0.005,           # 熵系数（连续空间内在探索更多，稍低即可）
    vf_coef=0.5,              # 价值函数损失系数
    gamma=0.99,               # 折扣因子
    gae_lambda=0.95,          # GAE lambda
    verbose=1,
    seed=42,
    device="auto",
)

clip_val = model.clip_range(1.0) if callable(model.clip_range) else model.clip_range
print(f"  学习率:       {model.learning_rate}")
print(f"  Rollout 步数: {model.n_steps}")
print(f"  批量大小:     {model.batch_size}")
print(f"  更新轮数:     {model.n_epochs}")
print(f"  裁剪范围:     [{1 - clip_val:.1f}, {1 + clip_val:.1f}]")
print(f"  熵系数:       {model.ent_coef}")
print(f"  动作空间:     连续 {vec_env.num_envs} 维（关节扭矩）")


# ==========================================
# 第四部分：训练模型
# ==========================================
total_timesteps = args.total_timesteps
print(f"\n开始训练（{total_timesteps:,} 时间步）...")
print("-" * 50)

callback = TrainingMonitorCallback(check_freq=2048)

model.learn(
    total_timesteps=total_timesteps,
    callback=callback,
    progress_bar=True,
)

print("-" * 50)
print("训练完成！")


# ==========================================
# 第五部分：绘制训练曲线（4 张独立图）
# ==========================================
print("\n正在绘制训练曲线...")

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# 图1：回合奖励曲线
if callback.episode_rewards:
    rewards = callback.episode_rewards
    window = min(50, max(1, len(rewards)))
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(smoothed, color="#2196F3", alpha=0.8, linewidth=1.5, label="滑动平均")
    ax.axhline(y=300, color="green", linestyle="--", alpha=0.5, label="solved (300)")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.set_title("PPO BipedalWalker-v3 回合奖励", fontsize=14, fontweight="bold")
    ax.set_xlabel("回合")
    ax.set_ylabel("累计奖励")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ppo_bipedal_walker_reward.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  奖励曲线 → output/ppo_bipedal_walker_reward.png")

# 图2：策略熵
if callback.entropy_list:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(callback.timesteps_list, callback.entropy_list,
            color="#FF9800", alpha=0.8, linewidth=1.5)
    ax.set_title("PPO BipedalWalker-v3 策略熵", fontsize=14, fontweight="bold")
    ax.set_xlabel("时间步")
    ax.set_ylabel("熵")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ppo_bipedal_walker_entropy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  策略熵曲线 → output/ppo_bipedal_walker_entropy.png")

# 图3：裁剪比例
if callback.clip_fraction_list:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(callback.timesteps_list, callback.clip_fraction_list,
            color="#F44336", alpha=0.8, linewidth=1.5, label="clip fraction")
    ax.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5, label="clip_range=0.2")
    ax.set_title("PPO BipedalWalker-v3 裁剪比例", fontsize=14, fontweight="bold")
    ax.set_xlabel("时间步")
    ax.set_ylabel("被裁剪的比例")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ppo_bipedal_walker_clip.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  裁剪比例曲线 → output/ppo_bipedal_walker_clip.png")

# 图4：近似 KL 散度
if callback.approx_kl_list:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(callback.timesteps_list, callback.approx_kl_list,
            color="#4CAF50", alpha=0.8, linewidth=1.5)
    ax.set_title("PPO BipedalWalker-v3 近似 KL 散度", fontsize=14, fontweight="bold")
    ax.set_xlabel("时间步")
    ax.set_ylabel("KL 散度")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ppo_bipedal_walker_kl.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  KL 散度曲线 → output/ppo_bipedal_walker_kl.png")


# ==========================================
# 第六部分：评估训练好的模型
# ==========================================
print("\n正在评估最终模型（20 个测试回合）...")
print("-" * 50)

eval_env = gym.make("BipedalWalker-v3")
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=20, deterministic=True
)
print(f"20 回合测试结果：")
print(f"  平均奖励: {mean_reward:.2f}")
print(f"  标准差:   {std_reward:.2f}")

test_rewards = []
for ep in range(20):
    obs, _ = eval_env.reset()
    done, truncated = False, False
    total_reward = 0.0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = eval_env.step(action)
        total_reward += reward
    test_rewards.append(total_reward)

print(f"\n逐回合奖励：")
for i, r in enumerate(test_rewards):
    status = "达标" if r >= 300 else ("中等" if r >= 100 else "未达标")
    print(f"  回合 {i + 1:2d}: {r:8.2f}  [{status}]")

print(f"\n达标率（>= 300 分）: {sum(1 for r in test_rewards if r >= 300)}/20")
eval_env.close()


# ==========================================
# 第七部分：保存模型
# ==========================================
model.save("output/ppo_bipedal_walker")
print(f"\n模型已保存至: output/ppo_bipedal_walker.zip")
print("=" * 50)
