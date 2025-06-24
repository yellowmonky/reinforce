# simple_dqn_pong_fixed.py
# PyTorch DQN 구현 (classic Gym) — step() 반환값 5개 처리

import random                              # Python 내장 랜덤 모듈 불러오기
import numpy as np                         # NumPy를 np라는 이름으로 불러오기 (배열 연산용)
import torch                               # PyTorch 메인 모듈 불러오기
import torch.nn as nn                      # PyTorch의 신경망 모듈 불러오기
import torch.nn.functional as F            # PyTorch의 함수형 신경망 유틸리티 불러오기
import torch.optim as optim                # PyTorch의 최적화 알고리즘 모듈 불러오기
import gym                                 # OpenAI Gym 불러오기 (강화학습 환경)
from gym.wrappers import AtariPreprocessing, FrameStack  
#   - AtariPreprocessing: Atari 환경에 표준 전처리 (프레임 스킵, 흑백, 리사이즈 등) 적용
#   - FrameStack: 연속된 프레임을 쌓아 (stack) 입력으로 사용하도록 래핑

# ----------------------------
# 1. Replay Buffer Definition
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity, batch_size, obs_shape, device):
        self.capacity = capacity                # 저장할 최대 transition 개수
        self.batch_size = batch_size            # 학습 시 한 번에 샘플링할 배치 크기
        self.device = device                    # 연산을 수행할 디바이스 (CPU/GPU)

        # 상태(state)와 다음 상태(next_state)를 uint8 형식으로 저장할 배열 초기화
        # shape = (capacity, 채널(4), 높이(84), 너비(84))
        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        # 행동(action), 보상(reward), 종료 플래그(done) 등 저장할 배열 초기화
        self.actions = np.zeros((capacity,), dtype=np.int64)   # 정수 행동
        self.rewards = np.zeros((capacity,), dtype=np.float32) # 부동소수점 보상
        self.dones = np.zeros((capacity,), dtype=np.bool_)     # 불리언 종료 여부

        self.idx = 0                              # 현재 저장 위치 인덱스
        self.size = 0                             # 현재 저장된 transition 개수

    def push(self, state, action, reward, next_state, done):
        # 인자로 받은 transition을 버퍼에 저장
        self.states[self.idx] = state             # 현재 인덱스 위치에 state 저장
        self.next_states[self.idx] = next_state   # next_state 저장
        self.actions[self.idx] = action           # action 저장
        self.rewards[self.idx] = reward           # reward 저장
        self.dones[self.idx] = done               # done 플래그 저장

        # 인덱스를 한 칸 증가시키되, capacity를 초과하면 0으로 돌아가서 덮어쓰기
        self.idx = (self.idx + 1) % self.capacity
        # 저장된 개수를 올바르게 유지 (max는 capacity)
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        # 학습 배치를 샘플링: 현재 저장된 transition 중 무작위로 batch_size만큼 인덱스 선택
        idxs = np.random.randint(0, self.size, size=self.batch_size)

        # NumPy 배열을 PyTorch 텐서로 변환하고 [0,1] 범위로 정규화
        states = (
            torch.from_numpy(self.states[idxs].astype(np.float32) / 255.0)
            .to(self.device)
        )           # (배치 크기, 4, 84, 84), float32
        next_states = (
            torch.from_numpy(self.next_states[idxs].astype(np.float32) / 255.0)
            .to(self.device)
        )           # (배치 크기, 4, 84, 84), float32

        # 행동, 보상, 종료 여부를 텐서로 변환
        actions = torch.from_numpy(self.actions[idxs]).to(self.device)   # (배치 크기,)
        rewards = torch.from_numpy(self.rewards[idxs]).to(self.device)   # (배치 크기,)
        dones = torch.from_numpy(self.dones[idxs].astype(np.uint8)).to(self.device)  # (배치 크기,)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size  # 현재 버퍼에 저장된 transition 개수를 반환


# ----------------------------
# 2. Q-Network Definition
# ----------------------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # DeepMind DQN 구조와 동일한 합성곱 레이어(Conv) 세트 정의
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)   # → (32, 20, 20)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)            # → (64, 9, 9)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)            # → (64, 7, 7)
        self.fc = nn.Linear(64 * 7 * 7, 512)                                # FC 레이어

    def forward(self, x):
        x = F.relu(self.conv1(x))   # 1단계: Conv1 + ReLU 활성화 (입력: (배치, 4, 84, 84))
        x = F.relu(self.conv2(x))   # 2단계: Conv2 + ReLU (→ (배치, 64, 9, 9))
        x = F.relu(self.conv3(x))   # 3단계: Conv3 + ReLU (→ (배치, 64, 7, 7))
        x = x.flatten(start_dim=1)  # 4단계: flatten (→ (배치, 64*7*7 = 3136))
        x = F.relu(self.fc(x))      # 5단계: FC 레이어 + ReLU (→ (배치, 512))
        return x                    # 최종 특징 벡터 반환 (배치, 512)


class QNetwork(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        # 앞서 정의한 CNNFeatureExtractor를 사용해 특징 추출
        self.features = CNNFeatureExtractor(in_channels)  # in_channels = 4
        # 추출된 특징(512-d)로부터 행동값(Q값)을 출력하는 완전연결 레이어
        self.output_layer = nn.Linear(512, n_actions)     # n_actions = 6

    def forward(self, x):
        x = self.features(x)        # 특징 추출 (→ (배치, 512))
        q = self.output_layer(x)    # Q값 계산 (→ (배치, n_actions))
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
        self.device = device           # 연산 디바이스 (GPU/CPU)
        self.n_actions = n_actions     # 행동 공간 크기 (Pong은 6)
        self.gamma = gamma             # 할인 계수 γ

        # ε-greedy 탐험 파라미터 초기화
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_frames = epsilon_decay_frames

        # 경험 재생 버퍼 초기화
        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity,            # 버퍼 크기
            batch_size=batch_size,               # 배치 크기
            obs_shape=(in_channels, 84, 84),     # 관측치 형태 (4,84,84)
            device=device,                       # 텐서를 저장할 디바이스
        )
        self.batch_size = batch_size

        # Q-네트워크(온라인) 및 타깃 네트워크 생성
        self.q_net = QNetwork(in_channels, n_actions).to(device)      # 온라인 네트워크
        self.target_net = QNetwork(in_channels, n_actions).to(device) # 타깃 네트워크
        # 타깃 네트워크 가중치를 온라인 네트워크와 동일하게 초기화
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # 타깃 네트워크는 평가 모드로 고정

        # 최적화 알고리즘 (Adam) 설정
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # 학습 제어 플래그 초기화
        self.frame_idx = 0               # 전체 env.step 호출 횟수
        self.learning_starts = learning_starts  # 학습 시작 전 warm-up 단계
        self.train_freq = train_freq          # 몇 프레임마다 한 번 학습할지
        self.target_update_freq = target_update_freq  # 타깃 네트워크 동기화 빈도

    def select_action(self, state: np.ndarray) -> int:
        """
        현재 상태를 받아서 ε-greedy 정책으로 행동 선택
        state: np.ndarray, shape=(4,84,84), dtype=uint8, 값 범위 [0,255]
        반환: 정수형 행동 (0~5)
        """
        self.frame_idx += 1  # env.step 호출 1회만큼 카운트

        # ε를 선형으로 줄여나가기 (epsilon decay)
        if self.frame_idx < self.epsilon_decay_frames:
            self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * (
                self.frame_idx / self.epsilon_decay_frames
            )
        else:
            self.epsilon = self.epsilon_end

        # 무작위로 행동할 확률 ε
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)  # 무작위 행동 선택
        else:
            # 탐색이 끝난 뒤, 네트워크를 통해 Q값 최대 행동 선택
            state_t = (
                torch.from_numpy(state.astype(np.float32) / 255.0)
                .unsqueeze(0)  # 배치 차원 추가 → (1, 4, 84, 84)
                .to(self.device)
            )
            with torch.no_grad():
                q_vals = self.q_net(state_t)         # Q 네트워크 출력 → (1, n_actions)
                action = q_vals.argmax(dim=1).item() # 가장 큰 Q값을 갖는 행동 인덱스
            return action

    def update(self):
        """
        경험 재생 버퍼에서 샘플을 뽑아 Q 네트워크를 한 번 학습
        """
        # 버퍼가 충분한 크기보다 작으면 학습하지 않음
        if len(self.replay_buffer) < self.batch_size:
            return
        # warm-up 단계 (learning_starts 이전)에는 학습하지 않음
        if self.frame_idx < self.learning_starts:
            return
        # 지정된 train_freq마다 한 번씩 학습
        if self.frame_idx % self.train_freq != 0:
            return

        # 배치 샘플링
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        # states: (batch,4,84,84), actions/rewards/dones: (batch,)

        # 온라인 네트워크 Q(s,a) 계산
        q_values = self.q_net(states)                               # (batch, n_actions)
        # 선택했던 행동에 대응하는 Q값만 선택 → (batch,)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 타깃 네트워크로 다음 상태의 Q값 계산 (detach)
        with torch.no_grad():
            next_q_values = self.target_net(next_states)             # (batch, n_actions)
            next_q_max = next_q_values.max(dim=1)[0]                 # 각 배치마다 최대 Q값 → (batch,)
            # 벨만 타깃: r + γ * max_a' Q_target(s', a') * (1 - done)
            q_target = rewards + (1.0 - dones.float()) * (self.gamma * next_q_max)

        # MSE 손실 계산: (Q(s,a) - Q_target)^2
        loss = F.mse_loss(q_value, q_target)

        # 역전파 및 파라미터 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 지정된 빈도마다 타깃 네트워크 동기화
        if self.frame_idx % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def push_transition(self, state, action, reward, next_state, done):
        """
        주어진 transition을 ReplayBuffer에 저장
        """
        self.replay_buffer.push(state, action, reward, next_state, done)


# ----------------------------
# 4. Training Loop
# ----------------------------
def train():
    # 하이퍼파라미터 정의
    env_id = "PongNoFrameskip-v4"   # NoFrameskip 버전을 사용해야 전처리에서 중복 스킵 방지
    total_frames = 20_000_000       # 학습할 env.step 총 횟수 (≈80M 실제 프레임)
    replay_capacity = 200_000       # ReplayBuffer 크기
    batch_size = 32                 # 배치 크기
    gamma = 0.99                    # 할인 계수 γ
    lr = 1e-4                       # 학습률
    epsilon_start = 1.0             # ε 초기값 (완전 탐험)
    epsilon_end = 0.01              # ε 최종값 (거의 탐험하지 않음)
    epsilon_decay_frames = 1_000_000# ε를 이 env.step 동안 선형 감소
    target_update_freq = 10_000     # 타깃 네트워크 동기화 빈도 (env.step)
    learning_starts = 50_000        # replay buffer 채우고 나서 학습 시작
    train_freq = 4                  # 매 4 env.step마다 한번 학습

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) 원본 "PongNoFrameskip-v4" 환경 생성 (classic Gym)
    raw_env = gym.make(env_id, render_mode=None)  # render_mode=None → 학습 중 렌더링 없음

    # 2) AtariPreprocessing을 씌워 frame_skip=4 처리
    env = AtariPreprocessing(
        raw_env,
        frame_skip=4,        # 내부에서 4프레임씩 skip
        screen_size=84,      # 흑백 프레임을 84×84로 리사이즈
        grayscale_obs=True,  # 흑백 관측
        scale_obs=False      # uint8(0~255) 그대로 유지
    )
    # 3) 마지막 4프레임을 쌓아서 (4,84,84) 형태 관측치 생성
    env = FrameStack(env, num_stack=4)

    # 행동 공간 크기 및 입력 채널 수
    n_actions = env.action_space.n   # Pong은 Discrete(6)
    in_channels = 4                  # stacked grayscale frames 개수

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

    # 환경 초기화 → reset() 호출 시 (obs, info) 튜플 리턴 → [0]으로 obs만 가져옴
    state = env.reset()[0]             # shape=(4,84,84), dtype=uint8
    episode_reward = 0                 # 현재 에피소드 누적 보상
    episode_count = 0                  # 에피소드 카운터
    total_steps = 0                    # env.step 호출 카운터

    # 메인 학습 루프: 총 total_frames env.step까지 반복
    while total_steps < total_frames:
        # 1) 행동 선택 (ε-greedy)
        action = agent.select_action(state)

        # 2) 환경에 action 적용 → step()이 (obs, reward, terminated, truncated, info)를 반환
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(next_obs)  # numpy 배열로 변환 (shape=(4,84,84))
        done_flag = terminated or truncated  # 둘 중 하나라도 True면 에피소드 종료

        # 3) transition 저장 및 네트워크 업데이트
        agent.push_transition(state, action, reward, next_state, done_flag)
        agent.update()

        # 4) 상태 및 누적 보상 업데이트
        state = next_state
        episode_reward += reward
        total_steps += 1

        # 5) 에피소드가 끝나면 reset하고 통계 출력
        if done_flag:
            state = env.reset()[0]       # 새로운 에피소드 시작 → reset
            state = np.array(state)
            episode_count += 1
            if episode_count % 10 == 0:
                # 10 에피소드마다 평균 보상 및 ε 출력
                print(
                    f"Episode {episode_count}, Total Frames {total_steps}, "
                    f"Last 10-ep Reward ≈ {episode_reward:.1f}, Epsilon {agent.epsilon:.3f}"
                )
            episode_reward = 0            # 누적 보상 초기화

    # 학습 완료 후 Q-network 가중치 저장
    torch.save(agent.q_net.state_dict(), "dqn_pong_final.pth")
    print("Training completed. Model saved as dqn_pong_final.pth")


if __name__ == "__main__":
    train()  # 스크립트 직접 실행 시 train() 함수 호출
