#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

from telnetlib import Telnet
import numpy as np
from sympy import re
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class NstepQLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, n):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n = n
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
        
    def update(self, s, a, Return, lr):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        self.Q_sa[s, a] = self.Q_sa[s, a] + lr * (Return - self.Q_sa[s, a])

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    import math
    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n)
    
    all_rewards = []
    while n_timesteps > 0:
        s = env.reset()
        S = [s]
        A = []
        rewards = []
        # (1) Simulate a single episode from start state to the state determined by 'max_episode_length'
        for t in range(max_episode_length):
            n_timesteps = n_timesteps - 1
            a = pi.select_action(S[t], policy, epsilon, temp) # epsilon-greedy
            next_s, r , done = env.step(a)
            A.append(a)
            S.append(next_s)
            rewards.append(r)
            all_rewards.append(r)
            # if plot and n_timesteps < 100:
            #     env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.01)
            if done or n_timesteps == 0:
                break

        T_ep = t + 1

        # (2) Compute n-step targets and update
        for t in range(T_ep):
            m = min(n, T_ep - t)
            Return = 0
            if np.all(env._state_to_location(S[t + m]) == (7,3)):
                for i in range(m):
                    Return += (gamma ** i) * rewards[t + i]
            else:
                for i in range(m):
                    Return += (gamma ** i) * rewards[t + i]
                Return += (gamma ** m) * max(pi.Q_sa[S[t + m], ])
            
            pi.update(S[t], A[t], Return, learning_rate)
        
    return all_rewards
def test():
    n_timesteps = 50000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    print("Obtained rewards: {}".format(rewards))    
    
if __name__ == '__main__':
    test()
