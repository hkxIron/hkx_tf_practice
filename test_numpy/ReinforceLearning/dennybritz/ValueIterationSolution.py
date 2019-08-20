import numpy as np
import pprint
import sys
if "./" not in sys.path:
  sys.path.append("./")
#from lib.envs.gridworld import GridworldEnv
from gridworld import GridworldEnv
from gridworld import print_policy

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv(shape=(4,4)) # 4*4的方格

print("env.nS:", env.nS," env.nA:", env.nA, ' env.P[][]:',env.P)


def value_iteration(env, theta=0.0001, discount_factor=1.0):
  """
  Value Iteration Algorithm.

  Args:
      env: OpenAI env. env.P represents the transition probabilities of the environment.
          env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
          env.nS is a number of states in the environment.
          env.nA is a number of actions in the environment.
      theta: We stop evaluation once our value function change is less than theta for all states.
      discount_factor: Gamma discount factor.

  Returns:
      A tuple (policy, V) of the optimal policy and the optimal value function.
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

  V = np.zeros(env.nS) # 值函数,每个state一个值
  while True:
    # Stopping condition
    delta = 0
    # Update each state...
    for s in range(env.nS):
      # Do a one-step lookahead to find the best action
      action = one_step_lookahead(s, V)
      best_action_value = np.max(action)
      # Calculate delta across all states seen so far
      delta = max(delta, np.abs(best_action_value - V[s]))
      # Update the value function. Ref: Sutton book eq. 4.10.
      V[s] = best_action_value
      # Check if we can stop
    if delta < theta:
      break

  # Create a deterministic policy using the optimal value function
  policy = np.zeros([env.nS, env.nA])
  for s in range(env.nS):
    # One step lookahead to find the best action for this state
    A = one_step_lookahead(s, V)
    best_action = np.argmax(A)
    # Always take the best action
    policy[s, best_action] = 1.0

  return policy, V

policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print_policy(np.reshape(np.argmax(policy, axis=1), env.shape))
"""
 ^ < < v
 ^ ^ ^ v
 ^ ^ > v
 ^ > > ^
"""
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

# Test the value function
expected_v = np.array([ 0, -1, -2, -3,
                        -1, -2, -3, -2,
                        -2, -3, -2, -1,
                        -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

env._render()
