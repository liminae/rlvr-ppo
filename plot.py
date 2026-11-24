import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('ppo_rewards.json', 'r') as f:
    data = json.load(f)
steps   = np.array(data['steps'])
rewards = np.array(data['avg_rewards'])

sub_idx = steps % 20 == 0
ss_steps  = steps[sub_idx]
ss_rewards = rewards[sub_idx]

window = 100
ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
ma_steps = steps[window-1:]

plt.figure(figsize=(8,4))
plt.plot(ss_steps, ss_rewards, marker='o', linestyle='None', alpha=0.5, label='Raw (every 20)')
plt.plot(ma_steps, ma, linewidth=2, label=f'{window}-batch MA')
plt.xlabel('PPO Batch Step')
plt.ylabel('Avg Distance‑Reward')
plt.title('Improved PPO‑CoT Learning Curve')
plt.legend()
plt.tight_layout()
plt.savefig('ppo_learning_curve_improved.png')
print("Saved: ppo_learning_curve_improved.png")
