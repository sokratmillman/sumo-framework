"""Q-learning Agent class."""
import numpy as np

def create_greedy_policy(epsilon=0.1):
    def policy(q_table, state, action_space):
        probability = np.random.rand()
        if probability <= epsilon:
            action = np.random.randint(0, action_space.n)
        else:
            action = np.argmax(q_table[state])
        return action
    return policy

class CustomAgent:
    def __init__(self, initial_state, state_space, action_space, 
                 alpha=0.5, discount_factor=0.95, exploration_strategy=create_greedy_policy()):
        self.state = initial_state
        self.action_space = action_space
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy
        self.acc_reward = 0
        self.action = None

    def act(self):
        self.action = self.exploration(self.q_table, self.state, self.action_space)
        return self.action

    def learn(self, next_state, reward):
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        current_state = self.state
        last_action = self.action

        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor*self.q_table[next_state][best_next_action]
        td_delta = td_target - self.q_table[current_state][last_action]

        self.q_table[current_state][last_action] += self.alpha * td_delta
        self.state = next_state
        self.acc_reward += reward