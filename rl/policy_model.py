# rl/policy_model_pythia.py
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM


# rl/policy_model_pythia.py
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM

from rl.action_space import ActionSpace


class PolicyNetwork(nn.Module):

    def __init__(
        self,
        model_name="EleutherAI/pythia-2.8b",
        obs_dim=160,
        seq_len=5,
        action_dim=None,          # ← 不再作为真源
        lr=1e-5,
        freeze_backbone=True
    ):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # =================================================
        # 0) ActionSpace → action_dim（single source of truth）
        # =================================================
        self.action_space = ActionSpace()
        real_action_dim = int(self.action_space.get_total_actions())

        if action_dim is not None and int(action_dim) != real_action_dim:
            print(
                f"[WARN] Overriding policy action_dim {action_dim} -> {real_action_dim} "
                f"to match ActionSpace.get_total_actions()."
            )

        self.action_dim = real_action_dim

        # =================================================
        # 1) 本地加载 Pythia-2.8B（离线）
        # =================================================
        PYTHIA_PATH = Path(
            "/root/autodl-tmp/models--EleutherAI--pythia-2.8b/"
            "snapshots/2a259cdd96a4beb1cdf467512e3904197345f6a9"
        ).resolve()

        self.llm = AutoModelForCausalLM.from_pretrained(
            str(PYTHIA_PATH),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            output_hidden_states=True,
            local_files_only=True
        ).to(self.device)

        hidden_size = self.llm.config.hidden_size  # pythia-2.8b = 2560

        # 冻结 backbone
        if freeze_backbone:
            for p in self.llm.parameters():
                p.requires_grad = False

        # =================================================
        # 2) obs → hidden
        # =================================================
        self.obs_proj = nn.Linear(obs_dim, hidden_size).to(self.device)

        # =================================================
        # 3) Policy & Value heads（由 ActionSpace 决定维度）
        # =================================================
        self.policy_head = nn.Linear(hidden_size, self.action_dim).to(self.device)
        self.value_head = nn.Linear(hidden_size, 1).to(self.device)

        trainable = (
            list(self.obs_proj.parameters()) +
            list(self.policy_head.parameters()) +
            list(self.value_head.parameters())
        )

        self.optimizer = torch.optim.Adam(trainable, lr=lr)

        # 一致性断言（开发期保护）
        assert self.policy_head.out_features == self.action_space.get_total_actions(), \
            "Policy head dim != ActionSpace.get_total_actions()"




    # =====================================================
    # Forward
    # =====================================================
    def forward(self, obs):
        """
        obs: [B, seq_len, obs_dim]
        """
        obs = obs.to(self.device)

        # 1）obs → GPT hidden embedding
        emb = self.obs_proj(obs)  # [B, seq_len, hidden]
        emb = emb.to(dtype=self.llm.dtype)

        # 2）走 GPT-NeoX Transformer
        outputs = self.llm(
            inputs_embeds=emb,
            output_hidden_states=True
        )

        # 取最后一层 hidden state，第一个位置作为全局特征
        h_last = outputs.hidden_states[-1][:, 0, :]   # [B, hidden_size]
        h_last_fp32 = h_last.float()

        # 3) policy 的 logits
        logits = self.policy_head(h_last_fp32)

        # 4) value baseline
        value = self.value_head(h_last_fp32).squeeze(-1)

        return logits, value


    # =====================================================
    # 采样动作
    # =====================================================
    def act(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        logits, value = self.forward(obs_t)
        dist = torch.distributions.Categorical(logits=logits)

        action = dist.sample()
        logp = dist.log_prob(action)

        return int(action.item()), float(logp.item()), float(value.item())


    # =====================================================
    # PPO evaluate
    # =====================================================
    def evaluate_actions(self, obs, actions):
        logits, values = self.forward(obs)

        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        action_logps = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        return action_logps, values.squeeze(-1), entropy


    # =====================================================
    def save(self, path):
        torch.save({
            "obs_proj": self.obs_proj.state_dict(),
            "policy_head": self.policy_head.state_dict(),
            "value_head": self.value_head.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.obs_proj.load_state_dict(ckpt["obs_proj"])
        self.policy_head.load_state_dict(ckpt["policy_head"])
        self.value_head.load_state_dict(ckpt["value_head"])
