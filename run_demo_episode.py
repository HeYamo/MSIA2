import asyncio
import numpy as np
import torch

from rl.policy_model import PolicyNetwork
from env.interaction_env import InteractionEnvironment
from rl.obs_utils import dialogue_encoder, DialogueEncoder

from pathlib import Path
from PIL import Image

# ======================
# 你的 Attack / Target / Judge
# ======================
from agents.attack_llm import AttackLLM
from agents.target_llm import TargetLLM
from agents.judge_llm import JudgeLLM


# =============================================================
# 1. 启动 MiniLM 文本编码器（必须）
# =============================================================
dialogue_encoder = DialogueEncoder()
print("[OK] 初始化文本编码器 MiniLM → 128维投影。")


# =============================================================
# 读取真实 case（保持 PIL.Image，不要转 base64）
# =============================================================
def load_real_case(case_root: str):
    """
    case_root 形如：test-image/p10000032/s50414267
    返回：
      images: [PIL.Image, PIL.Image, ...]
      findings: str
    """
    case_root = Path(case_root)

    # --- Load findings ---
    txt_files = list(case_root.glob("*.txt"))
    if len(txt_files) == 0:
        raise FileNotFoundError(f"No findings .txt found in {case_root}")

    findings = txt_files[0].read_text(encoding="utf-8").strip()

    # --- Load images ---
    image_paths = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        image_paths.extend(list(case_root.glob(ext)))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {case_root}")

    images = [Image.open(p).convert("RGB") for p in image_paths]

    return images, findings



# =============================================================
# Demo — 单个 episode
# =============================================================
async def run_real_episode():

    print("\n====================")
    print(" Running REAL Episode")
    print("====================\n")

    # -------------------------
    # 加载真实 X-ray case
    # -------------------------
    images, findings = load_real_case("test-image/p10000032/s50414267")

    # -------------------------
    # 初始化策略网络
    # -------------------------
    policy = PolicyNetwork(
        obs_dim=160,   # 32 structured + 128 dialogue embedding
        seq_len=5,
        action_dim=18
    )

    policy.load("checkpoints/ppo_policy_final_18.pt")
    policy.eval()
    print("[OK] Policy loaded → running evaluation mode")
    # -------------------------
    # 初始化 Attack / Target / Judge
    # -------------------------
    attack_llm = AttackLLM(
        model_name="gpt-4o",
        simulated=False
    )
    target_llm = TargetLLM()
    judge_llm = JudgeLLM()

    # -------------------------
    # 初始化 Environment
    # -------------------------
    env = InteractionEnvironment(
        attack_llm=attack_llm,
        target_llm=target_llm,
        judge_llm=judge_llm,
        seq_len=5,
        obs_size=160,
        max_turns=10,
        action_dim=14
    )

    # -------------------------
    # reset
    # -------------------------
    obs = env.reset(images, findings)

    done = False
    turn = 1

    while not done:
        print(f"\n------ TURN {turn} ------")

        # policy 决定 action
        action_id, logp, value = policy.act(obs)

        # 环境执行一步
        obs, reward, done, info = await env.step(action_id)

        print("prompt:", info["prompt"])
        print("response:", info["response"])
        print("severity:", info["judge"]["severity"])
        print("reward:", reward)
        print("done:", done)

        turn += 1

    print("\nEpisode Finished! Total turns:", turn-1)



# =============================================================
# Main
# =============================================================
if __name__ == "__main__":
    asyncio.run(run_real_episode())
