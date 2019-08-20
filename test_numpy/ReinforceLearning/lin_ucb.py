import numpy as np

T = 1000  # T个客人
N = 10  # N道菜

DIM = 5 #特征的维数为5


np.random.seed(0)

true_rewards = np.random.uniform(low=0, high=1, size=N)  # 每道菜好吃的概率
estimated_rewards = np.zeros(N)  # 每道菜好吃的估计概率
chosen_count = np.zeros(N)  # 各个菜被选中的次数,即被尝试的次数
total_reward = 0

# 扰动
def calculate_delta(T, item):
    # 从未选中过,就是1
    if chosen_count[item] == 0:
        return 1
    else:
        return np.sqrt(2 * np.log(T) / chosen_count[item])

def UCB(t, N):
    upper_bound_probs = [estimated_rewards[item] + calculate_delta(t, item) for item in range(N)]
    item = np.argmax(upper_bound_probs)
    #
    reward = np.random.binomial(n=1, p=true_rewards[item]) # reward是一个分布,并不是一个确定的值
    return item, reward

def LinUCB(items, N):
    upper_bound_probs = [items[i].mul(theta) + calculate_delta(X) for i in range(N)]
    item = np.argmax(upper_bound_probs)
    reward = np.random.binomial(n=1, p=true_rewards[item])
    return item, reward

for t in range(1, T): # T个客人依次进入餐馆
   items = [extract_feature(i, t) for i in range(N)]
   item, reward = LinUCB(items, N)
   total_reward += reward # 一共有多少客人接受了推荐

print("ucb avg_reward=" + str(total_reward/T))