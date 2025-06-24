import gym
from stable_baselines3 import PPO

# 환경 생성 (렌더링 켜기)
env = gym.make("ALE/Pong-v5", render_mode="human")

# 모델 불러오기
model = PPO.load("ppo_pong")

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, truncated, info = env.step(action)

    if done or truncated:
        obs, _ = env.reset()
