import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L

def swish(x):
    return x * F.sigmoid(x)

class ActorCriticNet_V16(Model):
    def __init__(self, action_size=3):
        super().__init__()
        self.shared1 = L.Linear(128)
        self.shared2 = L.Linear(64)
        self.shared3 = L.Linear(32)
        self.pi_head = L.Linear(action_size)
        self.v_head  = L.Linear(1)

    def forward(self, x):
        x = swish(self.shared1(x))
        x = swish(self.shared2(x))
        x = swish(self.shared3(x))
        pi = F.softmax(self.pi_head(x))
        v  = self.v_head(x)
        return pi, v

class Agent:
    def __init__(self):
        self.gamma = 0.99
        self.lr_pi = 1e-4
        self.lr_v  = 5e-4
        self.entropy_beta = 0.05
        self.action_size = 3

        self.net = ActorCriticNet_V16(self.action_size)
        self.optimizer = optimizers.Adam(self.lr_pi).setup(self.net)

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs, _ = self.net(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]


    def update(self, state, action_prob, reward, next_state, done):
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        probs, v = self.net(state)
        _, next_v = self.net(next_state)

        target = reward + self.gamma * next_v * (1 - done)
        target.unchain()
        loss_v = F.mean_squared_error(v, target)

        delta = target - v
        delta.unchain()

        probs = probs[0]
        log_probs = F.log(probs)
        entropy = -F.sum(probs * log_probs)

        loss_pi = -F.log(action_prob) * delta - self.entropy_beta * entropy

        self.net.cleargrads()
        (loss_v + loss_pi).backward()
        self.optimizer.update()


episodes = 10000
env = gym.make('MountainCar-v0', render_mode='rgb_array')
agent = Agent()
reward_history = []

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.update(state, prob, reward, next_state, done)
        state = next_state
        total_reward += reward

    reward_history.append(total_reward)

    if episode % 100 == 0:
        print("episode : {}, total reward : {:.1f}".format(episode, total_reward))

from common.utils import plot_total_reward
plot_total_reward(reward_history, 'quiz13')

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
