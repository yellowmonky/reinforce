import matplotlib.pyplot as plt
import numpy as np


class Bandit:
    def __init__(self, arms = 10):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms) # 확률 재조정
        if rate > np.random.rand():
            return 1
        else: 
            return 0

class Agent:
    def __init__(self, epsilon, action_size = 10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self. ns = np.zeros(action_size)

    def update(self, action, reward): # bandit 머신 땡기겠다
        self.ns[action] += 1 # 10개의 bandit 머신 중 어떤걸 땡겼는지 체크
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action] # Q값 없데이트

    def get_action(self):
        if np.random.rand() < self.epsilon: # epsilon의 확률로
            return np.random.randint(0, len(self.Qs)) # 무작위 행동 선택
        return np.argmax(self.Qs) # 탐욕 행동 선택
    
class AlphaAgent:
    def __init__(self, epsilon, alpha, actions=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.alpha = alpha

    def update(self, action, reward):
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)
    

runs = 200
steps = 1000
epsilon = 0.1
alpha = 0.8
all_rates = np.zeros((runs, steps))
alpha_all_rates = np.zeros_like(all_rates)

for run in range(runs):

    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action() # 몇번 슬롯 머신을 선택할지 정하기
        reward = bandit.play(action) # 결과 확인
        agent.update(action, reward) # 결과 기록, Q값 계산
        total_reward += reward

        total_rewards.append(total_reward)
        rates.append(total_reward / (step + 1))
    
    all_rates[run] = rates

    bandit = Bandit()
    alpha_agent = AlphaAgent(epsilon, alpha)
    alpha_total_reward = 0
    alpha_total_rewards = []
    alpha_rates = []

    for step in range(steps):
        action = alpha_agent.get_action() # 몇번 슬롯 머신을 선택할지 정하기
        reward = bandit.play(action) # 결과 확인
        alpha_agent.update(action, reward) # 결과 기록, Q값 계산
        alpha_total_reward += reward

        alpha_total_rewards.append(alpha_total_reward)
        alpha_rates.append(alpha_total_reward / (step + 1))

    alpha_all_rates[run] = alpha_rates

avg_rates = np.average(all_rates, axis = 0)
alpha_avg_rates = np.average(alpha_all_rates, axis = 0)

plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(avg_rates, label = 'sample average')
plt.plot(alpha_avg_rates, label = 'alpha const')
plt.legend()
plt.show()
