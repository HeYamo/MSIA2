# trainer_ppo_discrete.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class PPOTrainer:
    def __init__(
        self,
        policy,
        clip_epsilon=0.2,
        epochs=4,
        batch_size=16,
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=1.0,
    ):
        self.policy = policy
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = getattr(policy, "device", "cpu")

    # -------------------------
    def _to_tensor(self, x, dtype=torch.float32):
        if torch.is_tensor(x):
            return x.to(dtype=dtype, device=self.device)
        return torch.tensor(np.array(x, dtype=np.float32), dtype=dtype, device=self.device)

    # -------------------------
    def compute_returns_and_advantages(self, rewards, values, dones=None, last_value=0.0):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(values + [last_value], dtype=torch.float32, device=self.device)
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 if (dones is None or not dones[t]) else 0.0
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return returns.detach(), advantages.detach()

    # -------------------------
    def update(self, batch):
        obs = batch["obs"]
        actions = batch["actions"]
        old_logps = batch["logps"]
        rewards = batch["rewards"]
        values = batch["values"]
        dones = batch.get("dones", None)

        obs = self._to_tensor(obs)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        old_logps = torch.tensor(old_logps, dtype=torch.float32, device=self.device)

        # ✅ Returns & advantages
        returns, advantages = self.compute_returns_and_advantages(rewards, values, dones)

        # ✅ NaN-safe normalization
        adv_mean, adv_std = advantages.mean(), advantages.std()
        if torch.isnan(adv_std) or adv_std < 1e-6:
            adv_std = torch.tensor(1.0, device=self.device)
        advantages = (advantages - adv_mean) / adv_std
        advantages = torch.clamp(advantages, -10.0, 10.0)

        dataset = TensorDataset(obs, actions, old_logps, returns, advantages)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(obs)), shuffle=True)

        total_loss = 0.0
        for _ in range(self.epochs):
            for b_obs, b_actions, b_old_logps, b_returns, b_adv in loader:
                # Forward pass
                new_logps, values_pred, entropy = self.policy.evaluate_actions(b_obs, b_actions)

                # ✅ NaN detection
                if torch.isnan(new_logps).any() or torch.isnan(values_pred).any():
                    print("⚠️ NaN in log_probs or values → skipping batch")
                    continue

                ratio = torch.exp(new_logps - b_old_logps)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values_pred, b_returns)
                entropy_bonus = entropy.mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

                # ✅ Safe optimize
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                total_loss += float(loss.item())

        avg_loss = total_loss / max(1, (self.epochs * len(loader)))
        return {"loss": avg_loss}
