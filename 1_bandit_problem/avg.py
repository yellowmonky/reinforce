import numpy as np

np.random.seed(0)
rewards = []

'''
행동 가치를 얻기 위해 깡으로 다 더하고 나눔

for n in range(1, 11): #10번 플레이
    reward = np.random.rand() #보상
    # print('reward : ',reward)
    rewards.append(reward)
    Q = sum(rewards) / n
    print('Q : ',Q)
'''

Q = 0

for n in range(1, 11): #10번 플레이
    reward = np.random.rand()
    Q += (reward - Q) / n
    print(Q)


