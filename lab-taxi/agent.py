import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state, eps):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment
        - eps (float): epsilon

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if np.random.rand() > eps: # select greedy action with probability epsilon
            return np.argmax(self.Q[state])
        else:                     # otherwise, select an action randomly
            return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done, eps, alpha=1.0, gamma=1.0):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        current = self.Q[state][action]
        policy_s = np.ones(self.nA) * eps / self.nA
        policy_s[np.argmax(self.Q[next_state])] += 1 - eps
        Qsa_next = np.dot(self.Q[next_state], policy_s)
        target = reward + (gamma * Qsa_next)
        self.Q[state][action] += (alpha * (target - current))