from random import random    # 随机策略时用到
from gym import Env
import gym
from gridworld import *      # 可以导入各种格子世界环境

class Agent():
    def __init__(self, env: Env):
        self.env = env      # 个体持有环境的引用
        self.Q = {}         # 个体维护一张行为价值表Q
        self.state = None   # 个体当前的观测，最好写成obs.

    def performPolicy(self, state): pass # 执行一个策略

    def act(self, a):       # 执行一个行为
        return self.env.step(a)

    def learning(self): pass   # 学习过程

    def performPolicy(self, s, episode_num, use_epsilon):
        epsilon = 1.00 / (episode_num + 1)
        Q_s = self.Q[s]
        str_act = "unknown"
        rand_value = random()
        action = None
        if use_epsilon and rand_value < epsilon:
            action = self.env.action_space.sample()
        else:
            str_act = max(Q_s, key=Q_s.get)
            action = int(str_act)
        return action

    def learning(self, gamma, alpha, max_episode_num):
        # self.Position_t_name, self.reward_t1 = self.observe(env)
        total_time, time_in_episode, num_episode = 0, 0, 0
        while num_episode < max_episode_num:  # 设置终止条件
            self.state = self.env.reset()  # 环境初始化
            s0 = self._get_state_name(self.state)  # 获取个体对于观测的命名
            self.env.render()  # 显示UI界面
            a0 = self.performPolicy(s0, num_episode, use_epsilon=True)

            time_in_episode = 0
            is_done = False
            while not is_done:  # 针对一个Episode内部
                # a0 = self.performPolicy(s0, num_episode)
                s1, r1, is_done, info = self.act(a0)  # 执行行为
                self.env.render()  # 更新UI界面
                s1 = self._get_state_name(s1)  # 获取个体对于新状态的命名
                self._assert_state_in_Q(s1, randomized=True)
                # 获得A'
                a1 = self.performPolicy(s1, num_episode, use_epsilon=True)
                old_q = self._get_Q(s0, a0)
                q_prime = self._get_Q(s1, a1)
                td_target = r1 + gamma * q_prime
                # alpha = alpha / num_episode
                new_q = old_q + alpha * (td_target - old_q)
                self._set_Q(s0, a0, new_q)

                if num_episode == max_episode_num:  # 终端显示最后Episode的信息
                    print("t:{0:>2}: s:{1}, a:{2:2}, s1:{3}". \
                          format(time_in_episode, s0, a0, s1))

                s0, a0 = s1, a1
                time_in_episode += 1

            print("Episode {0} takes {1} steps.".format(
                num_episode, time_in_episode))  # 显示每一个Episode花费了多少步
            total_time += time_in_episode
            num_episode += 1
        return


def main():
    env = SimpleGridWorld()
    agent = Agent(env)
    print("Learning...")
    agent.learning(gamma=0.9,
                   alpha=0.1,
                   max_episode_num=800)

if __name__ == "__main__":
    main()

