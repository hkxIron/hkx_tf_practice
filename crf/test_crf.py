from typing import *
"""
求解最可能的天气
天气状态：Rainy, Sunny, 不可观测(改成心情更合适)
活动状态： walk, shop, clean, 可以观测

已知：
1.初始隐状态概率(数据统计)： {'Rainy': 0.6, 'Sunny': 0.4}  # 初始状态概率
2.隐状态转移概率
transition_probability = {
    'Rainy': {'Rainy': 0.7, 'Sunny': 0.3},
    'Sunny': {'Rainy': 0.4, 'Sunny': 0.6},
}
3.隐状态发射概率
emission_probability = {
    'Rainy': {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
    'Sunny': {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
}
问题：
现观测到活动序列：('walk', 'shop', 'clean')，需求最可能的天气状态序列。

求解最可能的隐状态序列是HMM的三个典型问题之一，通常用维特比算法解决。维特比算法就是求解HMM上的最短路径（-log(prob)，也即是最大概率）的算法。

稍微用中文讲讲思路，很明显，
第一天天晴还是下雨可以算出来：
定义V[时间][今天天气] = 概率，注意今天天气指的是，前几天的天气都确定下来了（概率最大）今天天气是X的概率，这里的概率就是一个累乘的概率了。

因为第一天我的朋友去散步了，所以第一天下雨的概率
V[第一天][下雨] = 初始概率[下雨] * 发射概率[下雨][散步] = 0.6 * 0.1 = 0.06，
同理可得
V[第一天][天晴] = 初始概率[天晴] * 发射概率[天晴][散步]= 0.24 。
从直觉上来看，因为第一天朋友出门了，她一般喜欢在天晴的时候散步，所以第一天天晴的概率比较大，数字与直觉统一了。

从第二天开始，对于每种天气Y，都有前一天天气是X的概率 * X转移到Y的概率 * Y天气下朋友进行这天这种活动的概率。
因为前一天天气X有两种可能，所以Y的概率有两个，选取其中较大一个作为V[第二天][天气Y]的概率，同时将今天的天气加入到结果序列中
比较V[最后一天][下雨]和[最后一天][天晴]的概率，找出较大的哪一个对应的序列，就是最终结果。
"""

# 转移概率
transition_probability = {
    'Rainy': {'Rainy': 0.7, 'Sunny': 0.3},
    'Sunny': {'Rainy': 0.4, 'Sunny': 0.6},
}

# 发射概率
emission_probability = {
    'Rainy': {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
    'Sunny': {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
}

# 打印时间-状态 路径概率表
def print_dptable(V:List[Dict[str, float]]):
    print("    ")
    for i in range(len(V)):
        print("%7d" % i, end='')

    print("")
    for state in V[0].keys():
        print("%s: " % state, end='')
        for t in range(len(V)):
            print("%s " % ("%.4f" % V[t][state]), end='')
        print("")

def viterbi(obs:Tuple[str], states:Tuple[str], start_p:Dict[str, float], trans_p:Dict[str,Dict[str,float]], emit_p:Dict[str,Dict[str,float]]):
    """
    :param obs:观测序列
    :param states:隐状态
    :param start_p:初始概率（隐状态）
    :param trans_p:转移概率（隐状态）
    :param emit_p: 发射概率 （隐状态表现为显状态的概率）
    :return:
    """
    # 时间-隐状态概率表: V[时间][隐状态] = 概率
    # V: [{'Rainy': 0.06, 'Sunny': 0.24},     # t=0
    #     {'Rainy': 0.0384, 'Sunny': 0.043},  # t=1
    #     {'Rainy': 0.01344, 'Sunny': 0.002}] # t=2
    # t=2时，Rainy概率最大，因此最后状态为Rainy
    V:List[Dict[str, float]] = [{}]

    # 到达当前状态是是通过之前哪些隐状态路径来的,也可不保存，后面通过反向解码来计算该概率
    path:Dict[str, List[str]] = {}

    # 初始化初始状态 (t == 0)
    for cur_state in states: # rainy, sunny
        V[0][cur_state] = start_p[cur_state] * emit_p[cur_state][obs[0]] # obs: 'walk', 'shop', 'clean'
        path[cur_state] = [cur_state]

    # 对 t>=1, 前向跑一遍
    for t in range(1, len(obs)):
        V.append({})
        new_path = {}

        for cur_state in states: # rainy, sunny
            # 概率 隐状态 = 前一状态是pre_state的概率 * pre_state转移到state的概率 * state发射当前obs活动的概率
            # 此处可以将最大概率前一状态返回
            # max时会取prob里的最大值
            (prob, prev_state) = max([(V[t - 1][pre_state] * trans_p[pre_state][cur_state] * emit_p[cur_state][obs[t]], pre_state) \
                                      for pre_state in states])
            # 记录最大概率
            V[t][cur_state] = prob
            # 记录最可能的状态路径
            new_path[cur_state] = path[prev_state] + [cur_state] # 相当于append(cur_state)

        # 不需要保留旧路径
        path = new_path

    # 打印各时间-天气状态概率
    print_dptable(V)
    # 选出最后隐状态概率最大的那个
    (prob, prev_state) = max([(V[len(obs) - 1][state], state) for state in states])
    print("V:",V) # V [{'Rainy': 0.06, 'Sunny': 0.24}, {'Rainy': 0.0384, 'Sunny': 0.043}, {'Rainy': 0.01344, 'Sunny': 0.002}]
    print("path:",path) # V [{'Rainy': 0.06, 'Sunny': 0.24}, {'Rainy': 0.0384, 'Sunny': 0.043}, {'Rainy': 0.01344, 'Sunny': 0.002}]
    return prob, path[prev_state]

def example():
    states = ('Rainy', 'Sunny')  # 天气为隐状态，为不可观察
    observations = ('walk', 'shop', 'clean')  # 活动为发射的x，可以观测到
    start_probability = {'Rainy': 0.6, 'Sunny': 0.4}  # 初始状态概率
    return viterbi(observations,  #  ('walk', 'shop', 'clean') # 活动为发射的x，可以观测到
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability)

print("final result:", example())