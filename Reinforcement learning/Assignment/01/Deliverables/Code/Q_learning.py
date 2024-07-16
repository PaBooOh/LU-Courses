#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

from pydoc import Helper
import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class QLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
            # ---My code---
            # epsilon-greedy
            else:
                if np.random.random() >= epsilon:
                    action = argmax(self.Q_sa[s, ])
                else:
                    action = np.random.choice(range(len(self.Q_sa[s, ])))
                return action
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
            # ---My code---
            p = softmax(self.Q_sa[s, ], temp)
            action = np.random.choice(range(self.n_actions), 1, p=p)[0]
            return action
        
    def update(self, s, a, r, next_s, lr, gamma):
        # ---My code---
        self.Q_sa[s, a] = self.Q_sa[s, a] + lr * (r + gamma * max(self.Q_sa[next_s]) - self.Q_sa[s, a])

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []
    s = env.reset()
    for t in range(n_timesteps):
        # simulate a single episode from start state to terminal state
        a = pi.select_action(s, policy, epsilon, temp)
        next_s, r , done = env.step(a)
        # if plot and t >= 0.95 * n_timesteps:
        #     env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.05)
        rewards.append(r)
        pi.update(s, a, r, next_s, learning_rate, gamma)
        if done:
            s = env.reset()
        else:
            s = next_s
    # print(len(rewards))
    return rewards 

def test():
    
    n_timesteps = 50000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))

if __name__ == '__main__':
    test()
