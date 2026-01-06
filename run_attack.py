# run_attack.py
"""
Load a saved policy and run deterministic evaluation (uses simulated policy if no checkpoint).
"""
import argparse
import asyncio
import yaml
from pathlib import Path
import json
import time
from rl.policy_model import PolicyNetwork
from agents.attack_llm import AttackLLM
from agents.target_llm import TargetLLM
from agents.judge_llm import JudgeLLM
from rl.replay_buffer import RolloutBuffer

async def run_attack(config, simulated=False, checkpoint=None):
    model_name_1 = config.get("model", {}).get("attack", {})
    model_name_2 = config.get("model", {}).get("target", {})
    model_name_3  = config.get("model", {}).get("judge", {})
    attack = AttackLLM(model_name=model_name_1, simulated=simulated)
    target = TargetLLM(model_name=model_name_2, simulated=simulated, mode="mixed")
    judge = JudgeLLM(model_name= model_name_3, simulated=simulated)

    policy = PolicyNetwork(obs_size=32, action_size=4)
    if checkpoint and Path(checkpoint).exists():
        policy.load(checkpoint)

    history = []
    for step in range(1, config.get("environment",{}).get("max_turns",6)+1):
        obs = [0.0]*32
        try:
            action, logp, value = policy.act(obs)
        except Exception:
            import random
            action = random.randrange(4)
        style_map = {0:"direct",1:"worst",2:"hide",3:"empathetic"}
        q = await attack.generate_question(objective=config.get("environment",{}).get("objective",""), prev_q=None, prev_a=None)
        q = f"[{style_map[action]}]{q}"
        a = await target.respond(q)
        evalr = await judge.evaluate(q,a)
        history.append({"q":q, "a":a, "eval":evalr})
    out = {"history":history, "timestamp":int(time.time())}
    out_path = Path("outputs/conversations") / f"attack_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved attack conversation to {out_path}")
    await attack.close(); await target.close(); await judge.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--simulated", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    asyncio.run(run_attack(cfg, simulated=args.simulated, checkpoint=args.checkpoint))
