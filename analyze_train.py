import json, numpy as np
from collections import defaultdict

with open(r'C:\Users\chang.wei\.codely-cli\tmp\1ee5caba8e7d021797437383e283cff9cbcf1e8ef20e6d890473f16a7249abbf\tasks\8b079fb7-26f9-46bf-87e8-7de5a2bc7e15\task-5b1f07d1-1272-42ec-ad93-a5bf3808d9b7.output', encoding='utf-8') as f:
    data = [json.loads(l) for l in f if l.strip()]

trains = [d for d in data if d.get('type') == 'train' and d.get('step', 0) > 0]
evals = [d for d in data if d.get('type') == 'eval']

# 1. Per-config stats
config_stats = defaultdict(lambda: {'eps': 0, 'reward_sum': 0.0, 'lengths': [], 'count': 0, 'classic': 0, 'self_play': 0, 'pool': 0})
for t in trains:
    cfg = t.get('config', '?')
    opp = t.get('opponent', 'self_play')
    s = config_stats[cfg]
    s['eps'] += t.get('episodes', 0)
    s['reward_sum'] += t.get('mean_reward', 0) * t.get('episodes', 0)
    s['lengths'].append(t.get('mean_length', 0))
    s['count'] += 1
    if opp in ('classic', 'self_play', 'pool'):
        s[opp] += 1

print("=== Per-config training stats ===")
print(f"{'Config':<15} {'Rollouts':>8} {'Eps':>6} {'AvgRwd':>8} {'AvgLen':>8} {'Classic':>7} {'Self':>5} {'Pool':>5}")
for cfg, s in sorted(config_stats.items()):
    avg_r = s['reward_sum'] / max(s['eps'], 1)
    avg_l = np.mean(s['lengths'])
    print(f"{cfg:<15} {s['count']:>8} {s['eps']:>6} {avg_r:>8.3f} {avg_l:>8.1f} {s['classic']:>7} {s['self_play']:>5} {s['pool']:>5}")

# 2. Eval progression
print("\n=== Eval progression ===")
for e in evals:
    cfgs = e.get('configs', {})
    cfg_str = ' | '.join(f"{k}:{v:.0%}" for k, v in cfgs.items())
    print(f"Step {e['step']:>6}: avg={e['win_rate']:.1%} | {cfg_str} | rounds={e.get('avg_rounds', 0):.1f}")

# 3. Opponent distribution
print("\n=== Opponent distribution ===")
opp_counts = {'classic': 0, 'self_play': 0, 'pool': 0}
for t in trains:
    opp = t.get('opponent', 'self_play')
    opp_counts[opp] = opp_counts.get(opp, 0) + 1
total = sum(opp_counts.values())
for k, v in opp_counts.items():
    print(f"  {k}: {v}/{total} = {v/total:.1%}")

# 4. Reward by opponent type
print("\n=== Reward by opponent type ===")
opp_rewards = defaultdict(list)
for t in trains:
    opp = t.get('opponent', 'self_play')
    opp_rewards[opp].append(t.get('mean_reward', 0))
for opp in ['classic', 'self_play', 'pool']:
    rewards = opp_rewards.get(opp, [])
    if rewards:
        print(f"  {opp}: mean={np.mean(rewards):.3f} min={np.min(rewards):.3f} max={np.max(rewards):.3f} n={len(rewards)}")

# 5. Episode length trend (early vs late, per config)
print("\n=== Episode length trend (early 50% vs late 50%) ===")
for cfg in sorted(config_stats.keys()):
    cfg_trains = [t for t in trains if t.get('config') == cfg]
    mid = len(cfg_trains) // 2
    early = [t.get('mean_length', 0) for t in cfg_trains[:mid]]
    late = [t.get('mean_length', 0) for t in cfg_trains[mid:]]
    if early and late:
        print(f"  {cfg}: early={np.mean(early):.0f} -> late={np.mean(late):.0f}")

# 6. Reward trend (early vs late, per config)
print("\n=== Reward trend (early 50% vs late 50%) ===")
for cfg in sorted(config_stats.keys()):
    cfg_trains = [t for t in trains if t.get('config') == cfg]
    mid = len(cfg_trains) // 2
    early = [t.get('mean_reward', 0) for t in cfg_trains[:mid]]
    late = [t.get('mean_reward', 0) for t in cfg_trains[mid:]]
    if early and late:
        print(f"  {cfg}: early={np.mean(early):.3f} -> late={np.mean(late):.3f}")

# 7. approx_kl trend
print("\n=== approx_kl trend (early vs late) ===")
early_kl = [t.get('approx_kl', 0) for t in trains[:len(trains)//2]]
late_kl = [t.get('approx_kl', 0) for t in trains[len(trains)//2:]]
print(f"  early: {np.mean(early_kl):.4f} -> late: {np.mean(late_kl):.4f}")

# 8. Value loss trend
print("\n=== Value loss trend (early vs late) ===")
early_vl = [t.get('value_loss', 0) for t in trains[:len(trains)//2]]
late_vl = [t.get('value_loss', 0) for t in trains[len(trains)//2:]]
print(f"  early: {np.mean(early_vl):.4f} -> late: {np.mean(late_vl):.4f}")

# 9. Config-specific reward by opponent
print("\n=== Config x Opponent reward ===")
for cfg in sorted(config_stats.keys()):
    for opp in ['classic', 'self_play', 'pool']:
        rewards = [t.get('mean_reward', 0) for t in trains if t.get('config') == cfg and t.get('opponent') == opp]
        if rewards:
            print(f"  {cfg:>15} vs {opp:<10}: mean={np.mean(rewards):.3f} n={len(rewards)}")

# 10. Entropy trend
print("\n=== Entropy trend (early vs late) ===")
early_e = [t.get('entropy', 0) for t in trains[:len(trains)//2]]
late_e = [t.get('entropy', 0) for t in trains[len(trains)//2:]]
print(f"  early: {np.mean(early_e):.3f} -> late: {np.mean(late_e):.3f}")
