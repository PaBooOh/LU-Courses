#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class MonteCarloAgent:

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
        
    def update(self, s, a, Return, lr):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        self.Q_sa[s, a] += lr * (Return - self.Q_sa[s, a])

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)

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
            # if plot and episode >= 0.95 * n_timesteps:
            #     env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.01) # Plot the Q-value estimates during Q-learning execution
            if done or n_timesteps == 0:
                break

        Return = 0
        for i in range(t, -1, -1): # inverse: i = T-1, T-2 ,..., 0
            Return = rewards[i] + gamma * Return
            pi.update(S[i], A[i], Return, learning_rate)
        
        # all_rewards.extend(rewards)

    return all_rewards
    
def test():
    n_timesteps = 2000
    max_episode_length = 50
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))  
            
if __name__ == '__main__':
    test()
