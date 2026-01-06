# rl/obs_utils.py

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# ------------------------------------------------------------
# 需要你提供一个文本编码器：返回 128 维 embedding
# 你只要实现 dialogue_encoder.encode(text) → ndarray(128,)
# ------------------------------------------------------------
class DialogueEncoder:
    """
    把一句话 → 128维 embedding
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 载入 MiniLM 模型
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(self.device)
        self.model.eval()

        # MiniLM 输出 384 维 → 用线性层投影到 128 维
        self.proj = nn.Linear(384, 128).to(self.device)

    def encode(self, text: str):
        """
        input: text (str)
        output: [128] numpy 向量
        """
        if not text:
            return torch.zeros(128).cpu().numpy()

        batch = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**batch)
            token_embeddings = outputs[0]
            attention_mask = batch["attention_mask"]

            # mean pooling (mask padding)
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sentence_emb = torch.sum(token_embeddings * mask_expanded, dim=1)
            sentence_emb = sentence_emb / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        # 投影到 128 维
        emb_128 = self.proj(sentence_emb)  # [1, 128]

        return emb_128.squeeze(0).cpu().numpy()


# 全局 encoder，可在外部赋值
dialogue_encoder: DialogueEncoder = None


# ============================================================
# 初始化 obs 缓冲区
# ============================================================
def init_obs_buffer(seq_len, obs_size):
    """
    初始化 obs：
        [seq_len, obs_size]
    obs_size = structured_dim + dialogue_dim = 32 + 128 = 160
    """
    return np.zeros((seq_len, obs_size), dtype=np.float32)


# ============================================================
# 生成 structured obs（32维）
# ============================================================
def build_structured_obs(
        turn_id, max_turns,
        action_id, action_dim,
        severity, refusal_flag,
        refusal_count
):
    """
    构造结构化 obs（固定 32 维）
    """
    turn_id_norm = float(turn_id) / float(max_turns)

    # 动作 onehot
    action_onehot = np.zeros(action_dim, dtype=np.float32)
    if 0 <= action_id < action_dim:
        action_onehot[action_id] = 1.0

    is_refusal = 1.0 if refusal_flag else 0.0
    cumulative_refusal = float(refusal_count) / max(turn_id, 1)

    base = np.concatenate([
        np.array([turn_id_norm], dtype=np.float32),       # 1
        action_onehot,                                    # action_dim
        np.array([severity, is_refusal,                   # 3
                  cumulative_refusal], dtype=np.float32)
    ], axis=0)                                            # = 1 + action_dim + 3

    # 填充或截断到 32 维
    if base.shape[0] < 32:
        base = np.concatenate([base,
                               np.zeros(32 - base.shape[0], dtype=np.float32)])
    else:
        base = base[:32]

    return base


# ============================================================
# 生成对话 embedding（128维）
# ============================================================
def build_dialogue_emb(history_text: str):
    """
    将最近一轮对话历史编码成 128 维 embedding
    """
    if dialogue_encoder is None:
        # 默认零向量（你必须在外部初始化 encoder）
        return np.zeros(128, dtype=np.float32)

    emb = dialogue_encoder.encode(history_text)

    # 转成 numpy，并确保维度 128
    emb = np.asarray(emb, dtype=np.float32)
    if emb.shape[0] != 128:
        raise ValueError(f"Dialogue embedding must be 128 dim, got {emb.shape}")

    return emb


# ============================================================
# 组装新的 obs（结构化 32 + 对话 128 = 160）
# ============================================================
def obs_from_transition(
        prev_obs,
        transition,
        max_turns=10,
        action_dim=14,
        structured_dim=32,
        dialogue_dim=128
):
    """
    prev_obs: [seq_len, obs_size]
    transition: dict, 包含：
        - transition["action"]
        - transition["judge"]["severity"]
        - transition["judge"]["refusal"]
        - transition["prompt"], transition["response"]
        - transition["next_obs"]["turn_id"]
        - transition["next_obs"]["refusal_count"]
    """
    seq_len, obs_size = prev_obs.shape

    turn_id = transition["next_obs"].get("turn_id", 1)
    refusal_count = transition["next_obs"].get("refusal_count", 0)

    action_id = int(transition["action"])
    severity = float(transition["judge"].get("severity", 0.0))
    refusal_flag = bool(transition["judge"].get("refusal", False))

    # ----------------------------
    # 1) structured obs（32维）
    # ----------------------------
    structured_vector = build_structured_obs(
        turn_id=turn_id,
        max_turns=max_turns,
        action_id=action_id,
        action_dim=action_dim,
        severity=severity,
        refusal_flag=refusal_flag,
        refusal_count=refusal_count
    )  # shape (32,)

    # ----------------------------
    # 2) dialogue embedding（128维）
    # ----------------------------
    prompt = transition.get("prompt", "")
    response = transition.get("response", "")
    history_text = f"USER: {prompt}\nASSISTANT: {response}"

    dialogue_vector = build_dialogue_emb(history_text)  # shape (128,)

    # ----------------------------
    # 3) concat → full obs
    # ----------------------------
    combined = np.concatenate([structured_vector, dialogue_vector], axis=0)  # shape (160,)

    if combined.shape[0] != obs_size:
        raise RuntimeError(f"obs_size mismatch: expect {obs_size}, got {combined.shape[0]}")

    # ----------------------------
    # 4) 滑动窗口
    # ----------------------------
    new_obs = np.roll(prev_obs, shift=-1, axis=0)
    new_obs[-1] = combined
    return new_obs
