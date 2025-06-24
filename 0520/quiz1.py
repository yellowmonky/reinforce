import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L

class PolicyNet_V1(Model):
    def __init__(self, action_size=3):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        return F.softmax(self.l2(x))

class ValueNet_V1(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        return self.l2(x)

class Agent:
    def __init__(self):
        self.gamma   = 0.99
        self.lr_pi   = 2e-5
        self.lr_v    = 5e-5

        self.pi      = PolicyNet_V1()
        self.v       = ValueNet_V1()
        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v  = optimizers.Adam(self.lr_v).setup(self.v)
    # get_action, update: 원본과 동일
        self.action_size = 3

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.unchain()
        v = self.v(state)
        loss_v = F.mean_squared_error(v, target)

        delta = target - v
        delta.unchain()
        loss_pi = -F.log(action_prob) * delta

        self.v.cleargrads()
        self.pi.cleargrads()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.update()
        self.optimizer_pi.update()

    def update_trajectory(self, states, action_probs, rewards, next_states, dones):
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones).astype(float)
        action_probs = np.array(action_probs)
        
        # value 계산 (batch 연산)
        v = self.v(states).data.flatten()
        next_v = self.v(next_states).data.flatten()
        
        # target 계산
        targets = rewards + self.gamma * next_v * (1 - dones)
        delta = targets - v
        # advantage normalization
        adv = (delta - delta.mean()) / (delta.std() + 1e-8)
        
        # loss 계산 (배치 평균)
        loss_v = F.mean_squared_error(self.v(states), targets)
        loss_pi = 0
        for i in range(len(states)):
            loss_pi += -F.log(action_probs[i]) * adv[i]
        loss_pi /= len(states)
        
        self.v.cleargrads()
        self.pi.cleargrads()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.update()
        self.optimizer_pi.update()


episodes = 10000
env = gym.make('MountainCar-v0', render_mode='rgb_array')
agent = Agent()
reward_history = []

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    states, action_probs, rewards, next_states, dones = [], [], [], [], []

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # step 마다 reward shaping 적용
        reward += abs(next_state[0] - (-0.5))

        # 여기서 저장!
        states.append(state)
        action_probs.append(prob)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)

        state = next_state
        total_reward += reward

    reward_history.append(total_reward)

    if episode % 100 == 0:
        print("episode : {}, total reward : {:.1f}".format(episode, total_reward))
    agent.update_trajectory(states, action_probs, rewards, next_states, dones)

from common.utils import plot_total_reward
plot_total_reward(reward_history, 'quiz0')

env2 = gym.make('MountainCar-v0', render_mode='human')
state = env2.reset()[0]
done = False
total_reward = 0

while not done:
    action, prob = agent.get_action(state)
    next_state, reward, terminated, truncated, info = env2.step(action)
    done = terminated or truncated
    agent.update(state, prob, reward, next_state, done)
    state = next_state
    total_reward += reward
    env2.render()

print('Total Reward:', total_reward)
