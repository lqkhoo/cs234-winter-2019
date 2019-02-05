### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
        tuple of the form (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""


def as_tensor(P, nS, nA):
    """
    Return tensor representation of P. Due to how we tend to use P and its
    inconsistent dtype, we return three tensors instead of one higher dimensional one.
    """

    # First call
    if not hasattr(as_tensor, "last_p"):
        as_tensor.last_p = None

    # Memoize result and use it if P is unchanged across calls.
    # This still requires a dict equality check (dict is not hashable) so it's not efficient.
    # Still, it should be better than both stepping through the dictionary and then reinitializing the tensors.
    if P != as_tensor.last_p:
        as_tensor.last_p = P

        prob   = np.zeros((nS, nA, nS), dtype=float)   # p(s'|s,a)
        s_next = np.ones((nS, nA, nS), dtype=int) * -1 # {s': s' reacheable from (s,a)}. Unreacheable s' marked -1, but could be anything bounded, as p(s'|s,a)=0 for those entries
        r      = np.zeros((nS, nA, nS), dtype=float)   # r(s,a). This becomes r(s,a,s'(s,a)) in the stochastic env as the agent doesn't always end up where it wants to go

        for s in range(nS):
            for a in range(nA):
                # P[s][a] is a list of (prob, s_next, r, is_terminal) from taking a from s
                for i in range(len(P[s][a])):
                    s2 = P[s][a][i][1]              # valid state transition
                    s_next[s][a][s2] = s2
                    prob[s][a][s2] += P[s][a][i][0] # probability of that transition. += because sometimes we have the same transition defined twice...
                    r[s][a][s2] += P[s][a][i][2]    # reward of that transition

        as_tensor.prob   = prob    # Tensor: shape (nS, nA, nS)
        as_tensor.s_next = s_next  # Tensor: shape (nS, nA, nS)
        as_tensor.r      = r       # Tensor: shape (nS, nA, nS)

    # Think of this representation as indexing p(s'|s,a), r(s,a,s') and s'(s,a) on indices s,a,s'
    # Think of any state s as being able to transition to any other state s', but some transitions
    # have probability zero, meaning those transitions are invalid. In the deterministic env there
    # is one valid transition per state with p(s'|s,a)=1. In the stochastic env there are 3 with p=1/3 each.
    # s_next could have been binary. It stores next state information only for convenience of operation.

    # A:
    # 0 = left
    # 1 = down
    # 2 = right
    # 3 = up

    # Make sure probabilities of each (s,a) sum to 1
    assert(np.array_equal(np.sum(as_tensor.prob, axis=2), np.ones((nS,nA))))
    # Make sure probability of transitioning to undefined next states are all 0
    assert(np.sum(as_tensor.prob * (as_tensor.s_next==-1).astype(int)) == 0)
    # Make sure each valid state and action has full probability of transitioning
    assert(np.sum(as_tensor.prob * (as_tensor.s_next!=-1).astype(int)) == nA*nS)

    return (as_tensor.prob, as_tensor.s_next, as_tensor.r)



def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """

    value_function = np.zeros(nS)
    ############################
    # YOUR IMPLEMENTATION HERE #

    # Vectorize so it's easier to work with
    (prob, s_next, r) = as_tensor(P, nS, nA) # Tensors: shape (nS, nA, nS)

    V = value_function # Vector: shape (nS,)
    s = np.arange(nS, dtype=int)
    while True:
        
        Q = prob * (r + gamma * V[s_next])  # prob of all undefined state transitions are zero
        Q = np.sum(Q, axis=2)               # Sum across s'. Shape: (nS, nA).
        V_new = Q[s, policy[s]]             # Now that we have the matrix Q(s,a) just pick out the entries for a = pi(s)

        if np.linalg.norm(V - V_new, ord=np.inf) < tol:
            break
        V = V_new.copy()
    
    value_function = V_new
    ############################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """

    new_policy = np.zeros(nS, dtype='int')

    ############################
    # YOUR IMPLEMENTATION HERE #

    V_pi = value_from_policy

    # Vectorize so it's easier to work with
    (prob, s_next, r) = as_tensor(P, nS, nA)  # Tensors: shape (nS, nA, nS)

    Q_pi = prob * (r + gamma * V_pi[s_next])  # prob of all undefined state transitions are zero
    Q_pi = np.sum(Q_pi, axis=2)               # Sum across s'. Shape: (nS, nA).
    new_policy = np.argmax(Q_pi, axis=1)      # Greedy policy: pi(s) = argmax_a[Q_pi]. For each state, pick the action that maximizes Q_pi

    ############################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #

    while True:
        V = policy_evaluation(P, nS, nA, policy, gamma)
        policy_new = policy_improvement(P, nS, nA, V, policy, gamma) # argmax step
        if np.array_equal(policy, policy_new): # Terminate if policy has become stable across iterations
            break
        policy = policy_new.copy()

    value_function = V
    ############################
    return value_function, policy


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #

    # Vectorize so it's easier to work with
    (prob, s_next, r) = as_tensor(P, nS, nA) # Tensors: shape (nS, nA, nS)

    V = value_function
    V_new = V.copy()
    while True:
        
        ### Sutton & Barto eq (4.10)
        ### V(s) := max_a[ p(s',r|s,a) * sum_s'[r(s,a,s') + gamma*V(s')] ]

        Q = prob * (r + gamma * V[s_next])  # prob of all undefined state transitions are zero
        Q = np.sum(Q, axis=2)               # Sum across s'. Shape: (nS, nA)
        np.maximum(V_new, np.max(Q, axis=1), out=V_new) # Best V(s) is just best Q(s,a) among the actions available in each s
        if np.linalg.norm(V - V_new, ord=np.inf) < tol:
            break
        V = V_new.copy()

    policy = policy_improvement(P, nS, nA, V_new, policy, gamma)
    value_function = V_new.copy()
    ############################
    return value_function, policy


def render_single(env, policy, max_steps=100):
    """
        This function does not need to be modified
        Renders policy once on environment. Watch your agent play!

        Parameters
        ----------
        env: gym.core.Environment
        Environment to play on. Must have nS, nA, and P as
        attributes.
        Policy: np.array of shape [env.nS]
        The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    env.render()
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":

    def run_n(env, policy, max_steps=100, iterations=100):
        total_reward = 0
        for i in range(iterations):
            episode_reward = 0
            ob = env.reset()
            for t in range(max_steps):
                a = policy[ob]
                ob, reward, is_done, _ = env.step(a)
                episode_reward += reward
                if is_done:
                    break
            total_reward += episode_reward
        print("Total reward:" + str(total_reward))

    # comment/uncomment these lines to switch between deterministic/stochastic environments
    # env = gym.make("Deterministic-8x8-FrozenLake-v0")
    env = gym.make("Stochastic-4x4-FrozenLake-v0")

    print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    print(V_pi[0])
    # render_single(env, p_pi, 100)
    run_n(env, p_pi, 100, 100)

    print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)

    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    print(V_vi[0])
    # render_single(env, p_vi, 100)
    run_n(env, p_vi, 100, 100)
    