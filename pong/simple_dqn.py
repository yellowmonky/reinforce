# simple_dqn_pong_fixed.py
# PyTorch DQN 구현 (classic Gym) — step() 반환값 5개 처리

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from gym.wrappers import AtariPreprocessing, FrameStack

# ----------------------------
# 1. Replay Buffer Definition
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity, batch_size, obs_shape, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # (capacity, 4, 84, 84) 모양으로 uint8 저장
        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

        self.idx = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.idx] = state
        self.next_states[self.idx] = next_state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)

        states = (
            torch.from_numpy(self.states[idxs].astype(np.float32) / 255.0)
            .to(self.device)
        )           # (batch, 4, 84, 84) float32
        next_states = (
            torch.from_numpy(self.next_states[idxs].astype(np.float32) / 255.0)
            .to(self.device)
        )
        actions = torch.from_numpy(self.actions[idxs]).to(self.device)   # (batch,)
        rewards = torch.from_numpy(self.rewards[idxs]).to(self.device)   # (batch,)
        dones = torch.from_numpy(self.dones[idxs].astype(np.uint8)).to(self.device)  # (batch,)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size


# ----------------------------
# 2. Q-Network Definition
# ----------------------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # DeepMind DQN-style conv layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)   # -> (32,20,20)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)            # -> (64,9,9)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)            # -> (64,7,7)
        self.fc = nn.Linear(64 * 7 * 7, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # 입력: (batch, 4,84,84)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)  # (batch, 64*7*7)
        x = F.relu(self.fc(x))      # (batch, 512)
        return x


class QNetwork(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.features = CNNFeatureExtractor(in_channels)  # in_channels=4
        self.output_layer = nn.Linear(512, n_actions)     # n_actions = 6

    def forward(self, x):
        x = self.features(x)        # (batch, 512)
        q = self.output_layer(x)    # (batch, n_actions)
        return q


# ----------------------------
# 3. DQN Agent Definition
# ----------------------------
class DQNAgent:
    def __init__(
        self,
        in_channels: int,
        n_actions: int,
        device: torch.device,
        replay_capacity: int = 200_000,
        batch_size: int = 32,
        gamma: float = 0.99,
        lr: float = 1e-4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_frames: int = 1_000_000,
        target_update_freq: int = 10_000,
        learning_starts: int = 50_000,
        train_freq: int = 4,
    ):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma

        # ε-greedy 파라미터
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_frames = epsilon_decay_frames

        # 경험 재생 버퍼
        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
            batch_size=batch_size,
            obs_shape=(in_channels, 84, 84),
            device=device,
        )
        self.batch_size = batch_size

        # Q-네트워크, 타깃 네트워크
        self.q_net = QNetwork(in_channels, n_actions).to(device)
        self.target_net = QNetwork(in_channels, n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # 학습 제어 플래그
        self.frame_idx = 0
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq

    def select_action(self, state: np.ndarray) -> int:
        """
        state: np.ndarray, shape=(4,84,84), dtype=uint8, [0,255]
        returns: int action
        """
        self.frame_idx += 1

        # ε 선형 감소
        if self.frame_idx < self.epsilon_decay_frames:
            self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * (
                self.frame_idx / self.epsilon_decay_frames
            )
        else:
            self.epsilon = self.epsilon_end

        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            # 네트워크에 넣기 위해 텐서 변환 + 정규화
            state_t = (
                torch.from_numpy(state.astype(np.float32) / 255.0)
                .unsqueeze(0)
                .to(self.device)
            )  # shape=(1,4,84,84)
            with torch.no_grad():
                q_vals = self.q_net(state_t)         # (1, n_actions)
                action = q_vals.argmax(dim=1).item()
            return action

    def update(self):
        # 버퍼에 충분한 샘플이 쌓이지 않으면 학습하지 않음
        if len(self.replay_buffer) < self.batch_size:
            return
        # 아직 warm-up 단계
        if self.frame_idx < self.learning_starts:
            return
        # train_freq에 맞춰 학습
        if self.frame_idx % self.train_freq != 0:
            return

        # 미니배치 샘플링
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # 현재 네트워크 Q(s,a)
        q_values = self.q_net(states)                               # (batch, n_actions)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch,)

        # 타깃 네트워크 Q_target(s',a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states)             # (batch, n_actions)
            next_q_max = next_q_values.max(dim=1)[0]                 # (batch,)
            q_target = rewards + (1.0 - dones.float()) * (self.gamma * next_q_max)

        # MSE Loss
        loss = F.mse_loss(q_value, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 타깃 네트워크 동기화
        if self.frame_idx % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def push_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)


# ----------------------------
# 4. Training Loop
# ----------------------------
def train():
    # 하이퍼파라미터
    env_id = "PongNoFrameskip-v4"   # 반드시 NoFrameskip 버전을 사용
    total_frames = 20_000_000       # 약 20M env 스텝(≈80M 실제 프레임)
    replay_capacity = 200_000
    batch_size = 32
    gamma = 0.99
    lr = 1e-4
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay_frames = 1_000_000
    target_update_freq = 10_000
    learning_starts = 50_000
    train_freq = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) 원본 "PongNoFrameskip-v4" 환경 생성 (classic Gym)
    raw_env = gym.make(env_id, render_mode=None)

    # 2) AtariPreprocessing을 씌워 frame_skip=4 처리
    env = AtariPreprocessing(
        raw_env,
        frame_skip=4,        # AtariPreprocessing이 4프레임씩 skip하게 함
        screen_size=84,      # 84×84 리사이즈
        grayscale_obs=True,  # 흑백으로 변환
        scale_obs=False      # uint8 그대로 유지
    )
    # 3) 마지막 4프레임을 쌓아서 (4,84,84) 형태 관측치 생성
    env = FrameStack(env, num_stack=4)

    n_actions = env.action_space.n   # Pong은 Discrete(6)
    in_channels = 4                  # stacked grayscF frames 개수

    # 에이전트 생성
    agent = DQNAgent(
        in_channels=in_channels,
        n_actions=n_actions,
        device=device,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        gamma=gamma,
        lr=lr,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_frames=epsilon_decay_frames,
        target_update_freq=target_update_freq,
        learning_starts=learning_starts,
        train_freq=train_freq,
    )

    # 환경 초기화 → reset()이 (obs, info) 튜플 반환 → [0]으로 obs만 가져옴
    state = env.reset()[0]             # shape=(4,84,84), dtype=uint8
    episode_reward = 0
    episode_count = 0
    total_steps = 0

    while total_steps < total_frames:
        # 1) 행동 선택
        action = agent.select_action(state)

        # 2) 환경에 동작 적용 → step()이 5개 값을 반환
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(next_obs)  # (4,84,84), uint8
        done_flag = terminated or truncated

        # 3) transition 저장 및 네트워크 업데이트
        agent.push_transition(state, action, reward, next_state, done_flag)
        agent.update()

        state = next_state
        episode_reward += reward
        total_steps += 1

        # 4) 에피소드 종료 시
        if done_flag:
            state = env.reset()[0]
            state = np.array(state)
            episode_count += 1
            if episode_count % 10 == 0:
                print(
                    f"Episode {episode_count}, Total Frames {total_steps}, "
                    f"Last 10-ep Reward ≈ {episode_reward:.1f}, Epsilon {agent.epsilon:.3f}"
                )
            episode_reward = 0

    # 학습 완료 후 Q-network 저장
    torch.save(agent.q_net.state_dict(), "dqn_pong_final_1.pth")
    print("Training completed. Model saved as dqn_pong_final.pth")


if __name__ == "__main__":
    train()
