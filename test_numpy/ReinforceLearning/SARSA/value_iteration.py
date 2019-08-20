# https://nbviewer.jupyter.org/github/whitepaper/RL-Zoo/blob/master/value_iteration.ipynb
import numpy as np
from Env import GridWorld


DISCOUNT_FACTOR = 1

class Agent:
    def __init__(self, env):
        self.env = env
        self.V = np.zeros(env.nS)

    def next_best_action(self, s, V):
        action_values = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (reward + DISCOUNT_FACTOR * V[next_state])
        return np.argmax(action_values), np.max(action_values)

    def optimize(self):
        THETA = 0.0001
        delta = float("inf")
        round_num = 0

        while delta > THETA:
            delta = 0
            print("\nValue Iteration: Round " + str(round_num))
            print(np.reshape(self.V, env.shape))
            for s in range(env.nS):
                best_action, best_action_value = self.next_best_action(s, self.V)
                delta = max(delta, np.abs(best_action_value - self.V[s]))
                self.V[s] = best_action_value
            round_num += 1

        policy = np.zeros(env.nS)
        for s in range(env.nS):
            best_action, best_action_value = self.next_best_action(s, self.V)
            policy[s] = best_action

        return policy


env = GridWorld()
agent = Agent(env)
policy = agent.optimize()
print("\nBest Policy")
print(np.reshape([env.get_action_name(entry) for entry in policy], env.shape))

# env = GridWorld(wind_prob=.2)
# agent = Agent(env)
# policy = agent.optimize()
# print("\nBest Policy")
