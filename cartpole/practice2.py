import copy  # 객체를 통째로 복사할 때 사용
from collections import deque  # 고정 크기 큐를 구현하기 위해 사용
import random  # 무작위 샘플링을 위해 사용

import matplotlib.pyplot as plt  # 학습 결과 시각화를 위해 사용
import numpy as np  # 수치 연산 및 배열 처리를 위해 사용
import gym  # OpenAI Gym 환경을 사용하기 위해 임포트

from dezero import Model  # DeZero의 기본 신경망 모델 클래스
from dezero import optimizers  # 다양한 최적화 기법(예: Adam) 사용을 위해 임포트
import dezero.functions as F  # 활성화 함수, 손실 함수 등 각종 함수 모듈
import dezero.layers as L  # 신경망 레이어(L.Linear 등)를 제공하는 모듈

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        # 경험을 저장할 고정 크기 버퍼를 deque로 생성
        self.buffer = deque(maxlen=buffer_size)
        # 학습 시 한 번에 꺼내올 샘플 개수
        self.batch_size = batch_size

    def add(self, s, a, r, ns, done):
        # 상태 s, 행동 a, 보상 r, 다음 상태 ns, 종료 여부 done을 묶어 버퍼에 추가
        self.buffer.append((s, a, r, ns, done))

    def __len__(self):
        # 현재 버퍼에 저장된 경험 개수를 반환
        return len(self.buffer)

    def get_batch(self):
        # 버퍼에서 batch_size만큼 무작위로 경험을 샘플링
        batch = random.sample(self.buffer, self.batch_size)
        # 샘플에서 각각 상태, 행동, 보상, 다음 상태, 종료 여부를 분리
        s, a, r, ns, d = map(np.array, zip(*batch))
        # 종료 여부 d는 정수형으로 변환
        return s, a, r, ns, d.astype(np.int32)

class QNet(Model):
    def __init__(self, action_size):
        super().__init__()  # Model 초기화
        # 첫 번째 은닉 레이어: 출력 크기 128, 입력 크기는 first forward 시 자동 추론
        self.l1 = L.Linear(128)
        # 두 번째 은닉 레이어: 출력 크기 128
        self.l2 = L.Linear(128)
        # 출력 레이어: 행동 개수만큼 Q값 출력
        self.l3 = L.Linear(action_size)

    def forward(self, x):
        # 입력 x에 첫 레이어와 ReLU 활성화를 적용
        x = F.relu(self.l1(x))
        # 두 번째 레이어와 ReLU 활성화 적용
        x = F.relu(self.l2(x))
        # 출력 레이어를 거쳐 Q값 벡터 반환
        return self.l3(x)

class DQNAgent:
    def __init__(self):
        # 할인 계수 γ 설정
        self.gamma = 0.98
        # 학습률 설정
        self.lr = 5e-4
        # 탐험률 ε 초기값 설정
        self.epsilon = 1.0
        # 탐험률 최소값 설정
        self.epsilon_min = 0.01
        # 탐험률 감소 비율(매 update 후 곱해지는 값)
        self.epsilon_decay = 0.999

        # 경험 재생 버퍼 최대 크기
        self.buffer_size = 10000
        # 미니배치 크기
        self.batch_size = 64
        # MountainCar 환경의 행동 수(왼쪽, 유지, 오른쪽)
        self.action_size = 3

        # ReplayBuffer 인스턴스 생성
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        # Q 네트워크(현재 네트워크)
        self.qnet = QNet(self.action_size)
        # 타깃 네트워크(업데이트 대상)
        self.qnet_target = QNet(self.action_size)
        # Adam 옵티마이저 생성
        self.optimizer = optimizers.Adam(self.lr)
        # 옵티마이저에 Q 네트워크 파라미터 등록
        self.optimizer.setup(self.qnet)

    def get_action(self, state):
        # ε 확률로 랜덤 행동 선택(탐험)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        # 그렇지 않으면 Q 네트워크로부터 최대 Q값을 가진 행동 선택(활용)
        s = state[np.newaxis, :]  # 배치 차원 추가
        qs = self.qnet(s).data[0]  # Q값 벡터 계산
        return int(qs.argmax())  # 최대 Q값 인덱스 반환

    def update(self):
        # 버퍼에 충분한 샘플이 없으면 학습 생략
        if len(self.replay_buffer) < self.batch_size:
            return
        # 미니배치 샘플링
        s, a, r, ns, done = self.replay_buffer.get_batch()
        # 현재 Q값 계산
        qs = self.qnet(s)
        q_val = qs[np.arange(self.batch_size), a]

        # 타깃 Q값 계산
        next_qs = self.qnet_target(ns)
        max_next_q = next_qs.max(axis=1).data  # 최대 Q값 추출

        # 벨만 방정식에 따라 목표값 계산
        target = r + (1 - done) * self.gamma * max_next_q
        # MSE 손실 계산
        loss = F.mean_squared_error(q_val, target)

        # 그래디언트 초기화
        self.qnet.cleargrads()
        # 역전파 수행
        loss.backward()
        # 파라미터 업데이트
        self.optimizer.update()

        # ε를 감소시키되 최소값 이하로 내려가지 않도록
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def sync_target(self):
        # 현재 네트워크 파라미터를 타깃 네트워크로 복사
        self.qnet_target = copy.deepcopy(self.qnet)

def train_mountaincar(
    episodes=5000,            # 총 학습 에피소드 수
    sync_interval=50,         # 타깃 네트워크 동기화 주기(에피소드 단위)
    warmup_steps=1000,         # 학습 시작 전 버퍼 워밍업(스텝) 수
    freeze_eps_episodes=200,   # ε 고정 기간(에피소드)
    goal_bonus=100            # 목표 달성 시 추가 보너스
):
    # MountainCar 환경 생성, render_mode는 학습 시 필요 없음
    env = gym.make("MountainCar-v0")
    # 에이전트 초기화
    agent = DQNAgent()
    # 초기 타깃 네트워크 동기화
    agent.sync_target()

    # ε 감소 비율을 에피소드 수에 맞춰 자동 계산
    decay_steps = max(1, episodes - freeze_eps_episodes)
    agent.epsilon_decay = (agent.epsilon_min / 1.0) ** (1.0 / decay_steps)
    print(f"Using ε decay = {agent.epsilon_decay:.6f} over {decay_steps} episodes")

    rewards = []  # 에피소드별 성형 보상 기록
    step_count = 0  # 전체 스텝 카운트
    # 보상 성형 스케일 파라미터
    potential_scale = 100.0
    delta_scale     = 50.0

    for ep in range(episodes):
        # 환경 초기화, 랜덤 시드(ep)로 재현성 보장
        raw_state, _ = env.reset(seed=ep)
        pos, vel     = raw_state
        # 상태 특징 확장: [pos, vel, cos(3*pos)]
        state        = np.array([pos, vel, np.cos(3 * pos)])

        total_r = 0  # 에피소드 누적 보상
        done    = False

        # 초기 freeze_eps_episodes 기간 동안 ε 고정
        if ep < freeze_eps_episodes:
            agent.epsilon = 1.0

        while not done:
            # 행동 선택
            action = agent.get_action(state)
            # 환경에서 한 스텝 진행
            next_raw, r, term, trunc, _ = env.step(action)
            done = term or trunc  # 종료 여부 판단

            npos, nvel = next_raw
            # 다음 상태 특징 확장
            next_state = np.array([npos, nvel, np.cos(3 * npos)])

            # 1) Potential-based 보상 성형
            phi_s  = potential_scale * state[0]
            phi_ns = potential_scale * next_state[0]
            shaped_r = r + agent.gamma * phi_ns - phi_s

            # 2) 위치 변화량 보너스
            shaped_r += delta_scale * (next_state[0] - state[0])

            # 3) 목표 달성 시 큰 보너스
            if done and npos >= 0.5:
                shaped_r += goal_bonus

            # 경험을 리플레이 버퍼에 저장
            agent.replay_buffer.add(state, action, shaped_r, next_state, done)

            step_count += 1
            # 버퍼가 워밍업된 후부터 학습 시작
            if step_count > warmup_steps:
                agent.update()

            # 상태 업데이트 및 보상 누적
            state = next_state
            total_r += shaped_r

        # 에피소드 단위로 타깃 네트워크 동기화
        if ep % sync_interval == 0:
            agent.sync_target()

        rewards.append(total_r)
        print(f"Episode {ep:4d}  Shaped Reward {total_r:7.2f}  ε {agent.epsilon:.4f}")

    # 학습 곡선 시각화
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Shaped Reward")
    plt.show()

    return agent

if __name__ == "__main__":
    # 학습 시작
    trained_agent = train_mountaincar()

    # 평가: 화면에 에이전트 동작 시각화
    eval_env = gym.make("MountainCar-v0", render_mode="human")
    raw_state, _ = eval_env.reset(seed=123)
    pos, vel = raw_state
    state = np.array([pos, vel, np.cos(3 * pos)])
    done = False
    total_eval = 0

    while not done:
        action = trained_agent.get_action(state)  # 탐욕 정책 사용
        next_raw, r, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated

        npos, nvel = next_raw
        state = np.array([npos, nvel, np.cos(3 * npos)])
        total_eval += r  # 원래 보상 누적

        eval_env.render()  # human 렌더링

    print("Evaluation Total Reward:", total_eval)
    eval_env.close()  # 창 닫기
