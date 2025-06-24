from collections import deque
import random
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class Qnet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
    

class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000

        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        # 경험 버퍼를 생성한다.
        self.qnet = Qnet(self.action_size)
        # Q네트워크 초기화 출력 크기는 action size
        self.qnet_target = Qnet(self.action_size)
        # target Q네트워크 초기화
        self.optimizer = optimizers.Adam(self.lr)
        # 옵티마이저 생성
        self.optimizer.setup(self.qnet)

        self.batch_size = 32
        self.action_size = 2

    def sync_qnet(self):
        self.qnet_target=copy.deepcopy(self.qnet)

    def get_action(self, state):
        if np.random.rand()<self.epsilon:
            return np.random.choice(self.action_size)
        else:
            # 1) 배치 차원 추가
            state = state[np.newaxis, :]

        # - 원래 `state`가 (state_dim,) 형태의 1차원 배열인데, 신경망은 입력을 (batch_size, state_dim) 형태로 받으므로  
        # - `np.newaxis`를 사용해 첫 번째 축에 크기 1짜리 배치 차원을 추가해서 `(1, state_dim)` 형태로 만듭니다  

            # 2) Q-네트워크에 순전파
            qs = self.qnet(state)
        #     ```
        # - 배치 크기가 1인 `state`를 `self.qnet`에 입력해, 각 행동에 대한 Q값을 계산합니다  
        # - `qs`는 보통 `(1, action_size)` 형태의 텐서(또는 Variable)로, 0번 배치의 각 행동별 Q값을 담고 있습니다  

            # 3) 최고의 Q값을 갖는 행동 선택
            return qs.data.argmax()

        # - `qs.data`로 내부의 numpy 배열(혹은 tensor)을 꺼낸 뒤  
        # - `argmax()`를 호출해 배열에서 가장 큰 값의 인덱스(=가장 가치가 높은 행동)를 반환합니다  
        # - 반환된 정수가 에이전트가 선택할 행동 번호가 됩니다

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return # 아직 덜 참
        
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        # batch size만큼 가져옴
        qs = self.qnet(state)
        # state를 이용해서 qs 예측
        q = qs[np.arange(self.batch_size), action]
        # q는  state를 이용해 예측한 qs의 action을 batch사이즈 만큼 가져옴
        next_qs = self.qnet_target(next_state)
        # target_Q 네트워크에 next_state를 입력
        next_q = next_qs.max(axis=1)
        # 출력된 값의 첫번째 차원에 대한 값중 최댓값 가져옴
        next_q.unchain()
        # qnet 파라미터에 저장된 모든 기울기(grad)를 0으로 초기화함
        target = reward + (1 - done) * self.gamma * next_q
        # 벨만 방정식

        loss = F.mean_squared_error(q, target)

        self.qnet.cleargrads()
        # 올바른 설명: qnet 파라미터에 저장된 모든 기울기(grad)를 0으로 초기화함
        loss.backward()
        # 역전파, 최적화
        self.optimizer.update()
        # 계산된 기울기를 바탕으로 옵티마이저(Adam)가 qnet 파라미터를 업데이트(가중치 갱신)함
        

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)
    
    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data]) # 다차원 배열을 같은 형식으로 하나의 배열로 묶기 위함
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done
    

# replay_buffer = ReplayBuffer(buffer_size = 10000, batch_size = 32)

episodes = 300 # 에피소드 수
sync_interval = 20 # 신경망 동기화 주기
env = gym.make('CartPole-v0', render_mode='rgb_array')
agent = DQNAgent()
reward_history = []

for episode in range(episodes): #episode 300번 반복
    state = env.reset()[0] #state 초기화
    done = False # done Fasle 초기화
    total_reward = 0 # totla reward 초기화

    while not done: # 종료조건까지
        action = agent.get_action(state) # epsilon greedy
        next_state, reward, terminated, truncated, info = env.step(action)
        # action을 정하면 환경에 의해서 반환되는 값
        done = terminated | truncated
        # 종료조건 확인 (bool 값)
        agent.update(state, action, reward, next_state, done)
        # 환경에 의해서 반환되는 값을 업데이트 해준다.
        state = next_state
        # state transition
        total_reward += reward
        # reward 업데이트
    if episode % sync_interval == 0:
        agent.sync_qnet() # sync 주기에 맞춰서 

    reward_history.append(total_reward)
    # reward history에 추가
    if episode % 10 == 0:
        print("episode : {}, total reward:{}".format(episode, total_reward))
        # 상태 출력

plt.xlabel('episode')
plt.ylabel('total reward')
plt.plot(range(len(reward_history)), reward_history)
plt.show()

env2 = gym.make('CartPole-v0', render_mode = 'human')

agent.epsilon = 0
state = env2.reset()[0]
done = False
total_reward = 0

while not done:
    action = agent.get_action(state)
    next_state, reward, terminated, truncated, info = env2.step(action)
    done = terminated | truncated
    state = next_state
    total_reward += reward
    env2.render()
print('total reward', total_reward)
























# for episode in range(10):
#     state = env.reset()[0]
#     done = False

#     while not done:
#         action = 0

#         next_state, reward, terminated, truncated, info = env.step(action)
#         done = terminated | truncated

#         replay_buffer.add(state, action, reward, next_state, done)
#         state = next_state

# state, action, reward, next_state, done = replay_buffer.get_batch()

# print(state.shape)
# print(action.shape)
# print(reward.shape)
# print(next_state.shape)
# print(done.shape)
