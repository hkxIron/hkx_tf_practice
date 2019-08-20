"""
https://github.com/dennybritz/reinforcement-learning/tree/master/DP/

Summary
Dynamic Programming (DP) methods assume that we have a perfect model of the environment's Markov Decision Process (MDP). That's usually not the case in practice, but it's important to study DP anyway.
Policy Evaluation: Calculates the state-value function V(s) for a given policy. In DP this is done using a "full backup". At each state, we look ahead one step at each possible action and next state. We can only do this because we have a perfect model of the environment.
Full backups are basically the Bellman equations turned into updates.
Policy Improvement: Given the correct state-value function for a policy we can act greedily with respect to it (i.e. pick the best action at each state). Then we are guaranteed to improve the policy or keep it fixed if it's already optimal.
Policy Iteration: Iteratively perform Policy Evaluation and Policy Improvement until we reach the optimal policy.
Value Iteration: Instead of doing multiple steps of Policy Evaluation to find the "correct" V(s) we only do a single step and improve the policy immediately. In practice, this converges faster.
Generalized Policy Iteration: The process of iteratively doing policy evaluation and improvement. We can pick different algorithms for each of these steps but the basic idea stays the same.
DP methods bootstrap: They update estimates based on other estimates (one step ahead).
"""


import numpy as np
import pprint
import sys
if "./" not in sys.path:
  sys.path.append("./")

from gridworld import GridworldEnv
from gridworld import print_policy

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()


# Taken from Policy Evaluation Exercise!

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
  """
  Evaluate a policy given an environment and a full description of the environment's dynamics.

  Args:
      policy: [S, A] shaped matrix representing the policy.
      env: OpenAI env. env.P represents the transition probabilities of the environment.
          env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
          env.nS is a number of states in the environment.
          env.nA is a number of actions in the environment.
      theta: We stop evaluation once our value function change is less than theta for all states.
      discount_factor: Gamma discount factor.

  Returns:
      Vector of length env.nS representing the value function.
  """
  # Start with a random (all 0) value function
  V = np.zeros(env.nS) # [N_state, 1]
  while True:
    delta = 0
    # For each state, perform a "full backup"
    for s in range(env.nS):
      expected_value = 0
      # Look at the possible next actions
      for action, action_prob in enumerate(policy[s]): # policy里的值为每个action的概率
        # For each action, look at the possible next states...
        for prob, next_state, reward, done in env.P[s][action]: # 转移概率以及reward
          # Calculate the expected value
          expected_value += action_prob * prob * (reward + discount_factor * V[next_state])
      # How much our value function changed (across any states)
      delta = max(delta, np.abs(expected_value - V[s]))
      V[s] = expected_value
    # Stop evaluating once our value function change is below a threshold
    if delta < theta:
      break
  return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
  """
  Policy Improvement Algorithm. Iteratively evaluates and improves a policy
  until an optimal policy is found.

  Args:
      env: The OpenAI envrionment.
      policy_eval_fn: Policy Evaluation function that takes 3 arguments:
          policy, env, discount_factor.
      discount_factor: gamma discount factor.

  Returns:
      A tuple (policy, V).
      policy is the optimal policy, a matrix of shape [S, A] where each state s
      contains a valid probability distribution over actions.
      V is the value function for the optimal policy.

  """

  def one_step_lookahead(state, V):
    """
    Helper function to calculate the value for all action in a given state.

    Args:
        state: The state to consider (int)
        V: The value to use as an estimator, Vector of length env.nS

    Returns:
        A vector of length env.nA containing the expected value of each action.
    """
    A = np.zeros(env.nA)
    for a in range(env.nA):
      for prob, next_state, reward, done in env.P[state][a]:
        A[a] += prob * (reward + discount_factor * V[next_state])
    return A

  # Start with a random policy
  # policy:[n_state, n_action],每个state下采取每个action的概率
  policy = np.ones([env.nS, env.nA]) / env.nA

  while True:
    # Evaluate the current policy
    V = policy_eval_fn(policy, env, discount_factor)

    # Will be set to false if we make any changes to the policy
    policy_stable = True

    # For each state...
    for s in range(env.nS):
      # The best action we would take under the currect policy
      chosen_a = np.argmax(policy[s])

      # Find the best action by one-step lookahead
      # Ties are resolved arbitarily
      action_values = one_step_lookahead(s, V)
      best_a = np.argmax(action_values)

      # 策略改进
      # Greedily update the policy
      if chosen_a != best_a:
        policy_stable = False
      # 将当前state最优的policy记录,并设最优的action的选择概率为1
      policy[s] = np.eye(env.nA)[best_a]

    # If the policy is stable we've found an optimal policy. Return it
    if policy_stable:
      return policy, V

policy, v = policy_improvement(env)
print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print_policy(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
