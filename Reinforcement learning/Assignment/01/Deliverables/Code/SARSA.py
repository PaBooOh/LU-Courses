#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class SarsaAgent:

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
        
    def update(self, s, a, r, next_s, next_a, lr, gamma):
        # ---My code---
        self.Q_sa[s, a] = self.Q_sa[s, a] + lr * (r + gamma * self.Q_sa[next_s, next_a] - self.Q_sa[s, a])
        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []

    # TO DO: Write your SARSA algorithm here!
    s = env.reset()
    a = pi.select_action(s, policy, epsilon, temp) # epsilon-greedy
    for t in range(n_timesteps):
        # simulate a single episode from start state to terminal state
        next_s, r , done = env.step(a)
        # if plot and t >= 0.98 * n_timesteps:
        #     env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.05) 
        rewards.append(r)
        next_a = pi.select_action(next_s, policy, epsilon, temp)
        pi.update(s, a, r, next_s, next_a, learning_rate, gamma)
        if done:
            s = env.reset()
            a = pi.select_action(s, policy, epsilon, temp) # epsilon-greedy
        else:
            s = next_s
            a = next_a

    return rewards 


def test():
    n_timesteps = 100000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))        
    
if __name__ == '__main__':
    test()
