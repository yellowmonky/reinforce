import numpy as np

# class 정의

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


# main
if __name__=="__main__":
    bandit = Bandit()
    Qs = np.zeros(10) # 각 bandit 머신 가치 추정치
    ns = np.zeros(10) # bandit 머신 몇 번 플레이 했나?

    for n in range(10):
        action = np.random.randint(0, 10)
        reward = bandit.play(action)
        
        ns[action] += 1
        Qs[action] += (reward - Qs[action]) / ns[action]

        formatted = ' '.join(f"{q:5}" for q in Qs)
        print(f"{formatted}")

    print('\n\n\n\n')
    show_bandit = ' '.join(f"{n:5.2f}" for n in ns)
    print(show_bandit, '몇 번 뽑았냐')


