import gym
from stable_baselines3 import PPO

# 환경 생성
env = gym.make("ALE/Pong-v5")

# PPO 모델 생성
model = PPO("CnnPolicy", env, verbose=1)

# 학습
model.learn(total_timesteps=1_000_000)

# 저장
model.save("ppo_pong")


#---------------------------------TEST 시각화--------------------------------------
# env = gym.make("ALE/Pong-v5", render_mode="human")

# obs, _ = env.reset()
# for _ in range(1000):
#     action = env.action_space.sample()  # 또는 model.predict(obs)
#     obs, reward, done, truncated, info = env.step(action)
#     if done:
#         obs, _ = env.reset()

