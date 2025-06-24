"""
A2C (Advantage Actor–Critic) – Dezero 구현 예시
MountainCar-v0  (단일 환경 · n-step + GAE)

필요 패키지
    pip install gymnasium pyvirtualdisplay
    Dezero: https://github.com/oreilly-japan/dezero
"""

import numpy as np
import gym
from dezero import Model, optimizers, Variable
import dezero.functions as F
import dezero.layers as L
import matplotlib.pyplot as plt
from collections import deque
import dezero

# --------------------------- 네트워크 정의 --------------------------- #
class PolicyNet(Model):
    def __init__(self, action_size: int):
        super().__init__()
        self.l1 = L.Linear(256)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        return F.softmax(self.l2(x))           # (batch, action)

class ValueNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(256)
        self.l2 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        return self.l2(x)                      # (batch, 1)

# --------------------------- GAE 유틸 --------------------------- #
def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=1.0):
    n_steps = len(rewards)
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n_steps)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
        next_value = values[t]
    return adv

# --------------------------- A2C 에이전트 --------------------------- #
class A2CAgent:
    def __init__(
        self,
        action_size,
        n_steps=5,
        gamma=0.99,
        lam=1.0,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        lr=7e-4
    ):
        self.action_size = action_size
        self.n_steps = n_steps
        self.gamma = gamma
        self.lam = lam
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.pi = PolicyNet(action_size)
        self.v  = ValueNet()

        opt_params = list(self.pi.params()) + list(self.v.params())
        self.optimizer = optimizers.Adam(alpha=lr).setup(opt_params)

        # MountainCar 특화 스케일 정규화
        self.state_low  = np.array([-1.2, -0.07], dtype=np.float32)
        self.state_high = np.array([ 0.6,  0.07], dtype=np.float32)

    # --------- 헬퍼 --------- #
    def _preprocess(self, s):
        # [-1,1] 범위로 스케일
        return ((s - self.state_low) / (self.state_high - self.state_low)) * 2 - 1

    def select_action(self, state):
        x = Variable(self._preprocess(state)[np.newaxis, :])
        probs = self.pi(x)[0]                  # (action,)
        probs_np = probs.data
        a = np.random.choice(self.action_size, p=probs_np)
        logp = F.log(probs[a] + 1e-8)
        v = self.v(x)[0, 0]                    # scalar Variable
        return a, logp, v

    # --------- 메인 학습 루프 --------- #
    def train(self, env, episodes=5000, render=False):
        ep_returns = []
        recent = deque(maxlen=100)

        for episode in range(episodes):
            s, _    = env.reset()
            done    = False
            ep_ret  = 0.0
            step    = 0

            # rollout 버퍼
            states, actions_buf, values, rewards, dones = [], [], [], [], []

            while not done:
                if render: env.render()

                a, logp, v = self.select_action(s)
                s_next, r, terminated, truncated, _ = env.step(a)
                d = terminated or truncated

                # 버퍼 저장
                states.append(self._preprocess(s))
                actions_buf.append(a)
                values.append(v.data)          # detach
                rewards.append(r)
                dones.append(d)

                s = s_next
                ep_ret += r
                step   += 1

                # n-step 모아서 업데이트
                if len(rewards) == self.n_steps or d:
                    with dezero.no_grad():
                        next_v = self.v(Variable(self._preprocess(s)[np.newaxis, :]))[0,0].data if not d else 0.0
                    adv = compute_gae(np.array(rewards, dtype=np.float32),
                                      np.array(values, dtype=np.float32),
                                      np.array(dones, dtype=np.float32),
                                      next_v, self.gamma, self.lam)
                    ret = adv + np.array(values, dtype=np.float32)

                    # ------ 배치 텐서 ------ #
                    acts      = np.array(actions_buf, dtype=np.int32)      # (T,)
                    b_states = Variable(np.vstack(states))
                    #b_logps  = F.stack(logps)
                    b_adv    = Variable(adv[:, np.newaxis])
                    b_ret    = Variable(ret[:, np.newaxis])

                    # ------ forward ------ #
                    new_probs = self.pi(b_states)
                    entropy   = -F.sum(new_probs * F.log(new_probs + 1e-8), axis=1, keepdims=True)
                    # logπ(a|s)
                    idx = np.arange(len(states))
                    new_logp = F.log(new_probs[idx, acts] + 1e-8)       # quick reuse

                    pi_loss = -F.average(b_adv * new_logp)
                    v_pred  = self.v(b_states)
                    v_loss  = F.mean_squared_error(v_pred, b_ret)
                    entropy_loss = -F.mean(entropy)

                    loss = pi_loss + self.vf_coef * v_loss + self.ent_coef * entropy_loss

                    # ------ backward & update ------ #
                    self.optimizer.cleargrads()
                    loss.backward()
                    # clip grad
                    grad_norm = np.sqrt(sum([(p.grad ** 2).sum() for p in self.pi.params()] +
                                            [(p.grad ** 2).sum() for p in self.v.params()]))
                    if grad_norm > self.max_grad_norm:
                        for p in self.pi.params(): p.grad *= self.max_grad_norm / grad_norm
                        for p in self.v.params():  p.grad *= self.max_grad_norm / grad_norm

                    self.optimizer.update()

                    # rollout 버퍼 초기화
                    states, logps, values, rewards, dones = [], [], [], [], []

            ep_returns.append(ep_ret)
            recent.append(ep_ret)

            if episode % 100 == 0:
                avg100 = np.mean(recent) if recent else 0
                print(f"[{episode:4d}] return={ep_ret:7.1f}  avg100={avg100:6.1f}")

        return ep_returns

# --------------------------- 실행 --------------------------- #
if __name__ == "__main__":
    env = gym.make("MountainCar-v0", render_mode=None)
    agent = A2CAgent(action_size=env.action_space.n)

    rewards = agent.train(env, episodes=3000)

    # 결과 시각화
    plt.plot(rewards, label="Episode return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("A2C on MountainCar-v0 (Dezero)")
    plt.show()

    # 학습된 정책 시연
    env = gym.make("MountainCar-v0", render_mode="human")
    s, _ = env.reset()
    done = False
    while not done:
        a, _, _ = agent.select_action(s)
        s, _, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        env.render()
