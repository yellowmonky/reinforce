import re
import matplotlib.pyplot as plt

# 1. 로그 파일 경로
log_path = 'dqn_pong_terminal'

# 2. 에피소드 번호와 보상(Last 10-ep Reward)을 저장할 리스트
episodes = []
rewards = []

# 3. 정규식 패턴: "Episode 10, Total Frames ..., Last 10-ep Reward ≈ -21.0, Epsilon 0.991"
pattern = re.compile(r'Episode\s+(\d+),.*Last\s+10-ep\s+Reward\s+≈\s+([-\d\.]+)')

with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        m = pattern.search(line)
        if m:
            ep = int(m.group(1))
            rew = float(m.group(2))
            episodes.append(ep)
            rewards.append(rew)

# 4. 시각화
plt.figure(figsize=(8, 5))
plt.plot(episodes, rewards, linestyle='-', linewidth=1, markersize=4)
plt.xlabel('Episode')
plt.ylabel('Last 10-episode Reward')
plt.title('DQN Pong Training Curve')
plt.grid(True)
plt.tight_layout()
plt.show()
