# train_policy_ppo.py
"""
PPO training script (discrete actions + multimodal medical jailbreak)
--------------------------------------------------------------------
- 按目录遍历 case（test-image/p*/s*）
- 每个 case 跑一个 episode（多轮对话）
- 收集 transition → 用 PPOTrainer.update() 更新策略网络
"""

import asyncio
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ====== 你的模块 ======
from rl.policy_model import PolicyNetwork
from rl.ppo_trainer import PPOTrainer
from env.interaction_env import InteractionEnvironment
from rl.obs_utils import DialogueEncoder, dialogue_encoder
from rl.action_space import ActionSpace

from agents.attack_llm import AttackLLM
from agents.target_llm import TargetLLM
from agents.judge_llm import JudgeLLM


# =============================================================
# 0. 初始化全局文本编码器（重要）
# =============================================================
# obs_utils 里有全局变量 dialogue_encoder，这里真正赋值
dialogue_encoder = DialogueEncoder()
print("[OK] 初始化 MiniLM 文本编码器 → 128 维投影已启用。")


# =============================================================
# 1. 数据集工具：遍历 case 目录
# =============================================================
def find_cases(root_dir: str):
    """
    自动在 root_dir 下找所有含有 .txt + 图片 的 leaf 目录，作为一个个 case。

    root_dir 例子：
        "test-image"  (结构类似：test-image/p10000032/s50414267/...)

    返回：
        List[Path]，每个元素是一个 case 的目录路径
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    case_dirs = []

    # 策略：扫描所有子目录，凡是里面同时有 txt + image 就当成一个 case
    for sub in root.rglob("*"):
        if not sub.is_dir():
            continue

        txt_files = list(sub.glob("*.txt"))
        if not txt_files:
            continue

        img_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            img_files.extend(list(sub.glob(ext)))

        if img_files:
            case_dirs.append(sub)

    case_dirs = sorted(case_dirs)
    print(f"[Dataset] Found {len(case_dirs)} cases under {root_dir}")
    return case_dirs


def load_case(case_root: Path):
    """
    给一个 case 目录（如 test-image/p10000032/s50414267），
    返回 (images, findings)
        - images: List[PIL.Image]
        - findings: str
    """
    # 读 findings
    txt_files = list(case_root.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"[load_case] No .txt found in {case_root}")
    findings = txt_files[0].read_text(encoding="utf-8").strip()

    # 读影像
    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        img_paths.extend(list(case_root.glob(ext)))
    if not img_paths:
        raise FileNotFoundError(f"[load_case] No images found in {case_root}")

    images = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        images.append(img)

    return images, findings


# =============================================================
# 2. 单个 episode：在一个 case 上跑一整轮对话并收集 PPO 轨迹
# =============================================================
async def run_episode_on_case(
    case_root: Path,
    env: InteractionEnvironment,
    policy: PolicyNetwork,
    max_turns: int,
    verbose: bool = True,
):
    """
    在一个具体 case 上跑一个 episode：
        - env.reset(images, findings)
        - while not done:
             obs -> policy.act
             env.step(action)
        - 收集 obs / actions / logps / rewards / values / dones
        - 返回一个 batch dict
    """
    images, findings = load_case(case_root)

    # reset 环境
    obs = env.reset(images, findings)

    obs_list = []
    actions = []
    logps = []
    values = []
    rewards = []
    dones = []

    done = False
    turn = 1

    if verbose:
        print(f"\n==============================")
        print(f"[Episode] case = {case_root}")
        print(f"==============================\n")

    while not done and turn <= max_turns:
        if verbose:
            print(f"\n------ TURN {turn} ------")

        # 1) 策略选动作
        action_id, logp, value = policy.act(obs)  # obs: [seq_len, obs_dim]

        # 2) 环境执行一步
        new_obs, reward, done_flag, info = await env.step(action_id)

        # 3) 记录轨迹
        obs_list.append(obs.copy())
        actions.append(action_id)
        logps.append(logp)
        values.append(value)
        rewards.append(reward)
        dones.append(done_flag)

        if verbose:
            print("prompt:", info["prompt"])
            print("response:", info["response"][:300], "..." if len(info["response"]) > 300 else "")
            print("severity:", info["judge"].get("severity", None))
            print("reward:", reward)
            print("done:", done_flag)

        # 4) 准备下一轮
        obs = new_obs
        turn += 1
        done = done_flag

    # 整理成 numpy
    if len(obs_list) == 0:
        # 极端情况：一步没走就 done，直接返回空 batch
        return None

    batch = {
        "obs": np.stack(obs_list, axis=0),      # [T, seq_len, obs_dim]
        "actions": np.array(actions, dtype=np.int64),  # [T]
        "logps": np.array(logps, dtype=np.float32),    # [T]
        "rewards": rewards,                    # list[float]
        "values": values,                      # list[float]
        "dones": dones,                        # list[bool]
    }
    return batch


# =============================================================
# 3. 合并多个 episode 的 batch
# =============================================================
def merge_batches(batches):
    """
    把多个 episode batch 合并成一个大 batch，方便送进 PPOTrainer.update()
    batches: List[dict]
    """
    obs_list = []
    act_list = []
    logp_list = []
    reward_list = []
    value_list = []
    done_list = []

    for b in batches:
        if b is None:
            continue
        obs_list.append(b["obs"])
        act_list.append(b["actions"])
        logp_list.append(b["logps"])
        reward_list.extend(b["rewards"])
        value_list.extend(b["values"])
        done_list.extend(b["dones"])

    if not obs_list:
        return None

    obs = np.concatenate(obs_list, axis=0)  # [N, seq_len, obs_dim]
    actions = np.concatenate(act_list, axis=0)
    logps = np.concatenate(logp_list, axis=0)

    merged = {
        "obs": obs,
        "actions": actions,
        "logps": logps,
        "rewards": reward_list,
        "values": value_list,
        "dones": done_list,
    }
    return merged


# =============================================================
# 4. 训练主循环（异步）
# =============================================================
async def train_ppo(
    dataset_root: str = "test-image",
    total_epochs: int = 3,
    max_turns: int = 10,
    print_every_epoch: bool = True,
):

    # ---------- 1) 构建数据集 case 列表 ----------
    case_dirs = find_cases(dataset_root)
    if len(case_dirs) == 0:
        print("[WARN] No cases found, abort training.")
        return

    # ---------- 2) 初始化 Policy 网络 ----------
    policy = PolicyNetwork(
        obs_dim=160,   # 32 结构化 + 128 对话 embedding
        seq_len=5,
        action_dim=18
    )

    # ---------- 3) 初始化 PPOTrainer ----------
    trainer = PPOTrainer(
        policy=policy,
        clip_epsilon=0.2,
        epochs=4,
        batch_size=16,
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=1.0,
    )

    # ---------- 4) 初始化 LLM 模块 ----------
    attack_llm = AttackLLM(
        model_name="gpt-4o",
        simulated=False,           # 真实调用
        paraphrase_with_llm=True   # 让攻击问题一定经过 LLM 改写
    )
    target_llm = TargetLLM(
        model_name="gpt-4o",
        simulated=False
    )
    judge_llm = JudgeLLM(
        model_name="gpt-4o",   # 例如用便宜一点的 judge
        simulated=False
    )

    # ---------- 5) 初始化环境 ----------
    env = InteractionEnvironment(
        attack_llm=attack_llm,
        target_llm=target_llm,
        judge_llm=judge_llm,
        seq_len=5,
        obs_size=160,
        max_turns=max_turns,
        action_dim=18
    )

    # ---------- 6) PPO 训练主循环 ----------
    for epoch in range(1, total_epochs + 1):
        print(f"\n==================== EPOCH {epoch}/{total_epochs} ====================\n")

        episode_batches = []
        episode_rewards = []

        # 遍历所有 case（你可以改成 random.shuffle + 采样前 N 个）
        for i, case_dir in enumerate(case_dirs, start=1):
            print(f"\n[Epoch {epoch}] Episode {i}/{len(case_dirs)} — case: {case_dir}")

            batch = await run_episode_on_case(
                case_root=case_dir,
                env=env,
                policy=policy,
                max_turns=max_turns,
                verbose=False,   # 如果想看详细对话可以改成 True
            )

            if batch is None:
                print("  [WARN] Empty episode (no steps), skip.")
                continue

            episode_batches.append(batch)
            episode_rewards.append(sum(batch["rewards"]))

        # 合并 batch
        merged_batch = merge_batches(episode_batches)
        if merged_batch is None:
            print("[WARN] No valid episodes in this epoch, skip PPO update.")
            continue

        # PPO 更新
        stats = trainer.update(merged_batch)

        if print_every_epoch:
            avg_return = float(np.mean(episode_rewards)) if episode_rewards else 0.0
            print(f"\n[Epoch {epoch}] PPO loss = {stats['loss']:.4f}, "
                  f"avg episode return = {avg_return:.3f}, "
                  f"episodes = {len(episode_rewards)}")

    # ---------- 7) 关闭 HTTP 客户端 ----------
    await attack_llm.close()
    await target_llm.close()
    await judge_llm.close()

    # 可选：保存策略
    policy.save("checkpoints/ppo_policy_final_18.pt")
    print("\n[Done] PPO training finished, policy saved to ppo_policy_final.pt")


# =============================================================
# 5. main
# =============================================================
if __name__ == "__main__":
    # 根目录按你真实数据集路径改，比如：
    #   "test-image"  或  "mimic_cxr_samples"
    asyncio.run(train_ppo(
        dataset_root="  ",   # 只跑一个 case 试试
        total_epochs= ,     # 先跑 1 个 epoch 看效果
        max_turns= ,
    ))
