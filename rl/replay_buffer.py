"""
Rollout buffer (continuous-action version)
"""

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clear()

    def add(self, obs, action, logp, reward, value, done=False):
        self.obs.append(np.array(obs, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.float32))  # vector now
        self.logps.append(float(logp))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def add_transition(self, transition):
        self.add(
            transition.get("obs"),
            transition.get("action"),
            transition.get("logp"),
            transition.get("reward"),
            transition.get("value"),
            transition.get("done", False),
        )

    def sample(self):
        obs = np.array(self.obs, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.float32)
        logps = np.array(self.logps, dtype=np.float32)

        return {
            "obs": torch.tensor(obs, dtype=torch.float32, device=self.device),
            "actions": torch.tensor(actions, dtype=torch.float32, device=self.device),
            "logps": torch.tensor(logps, dtype=torch.float32, device=self.device),
            "rewards": self.rewards,
            "values": self.values,
            "dones": self.dones,
        }

    def clear(self):
        self.obs, self.actions, self.logps, self.rewards, self.values, self.dones = [], [], [], [], [], []
