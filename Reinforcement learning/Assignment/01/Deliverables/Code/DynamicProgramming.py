#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

from turtle import update
import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions)) # (7x10) x 4
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        # ---My code---
        # Greedy policy
        action = np.argwhere(self.Q_sa[s, ] == np.amax(self.Q_sa[s, ]))
        action = action.flatten()
        if action.shape[0] > 1:
            action = np.random.choice(action.flatten()) # If there are multiple maxima
            return action
        elif action.shape[0] == 1:
            return action[0]
        
    def update(self, env, gamma=1.0):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        # TO DO: Add own code
        # ---My code---
        delta = 0
        for s in range(self.n_states):
            for a in range(self.n_actions):
                p_sas, r_sas = env.model(s, a)
                q_sum = 0
                current_Q_sa = self.Q_sa[s, a] # Store current estimate
                next_states = np.where(p_sas != 0)[0]
                for next_s in next_states:
                    q_sum += p_sas[next_s] * (r_sas[next_s] + gamma * np.max(self.Q_sa[next_s,]))
                self.Q_sa[s, a] = q_sum
                delta = max(np.abs((current_Q_sa - q_sum)), delta)
           
        return delta, self.Q_sa
    
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)    
    
    # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
        
    # Plot current Q-value estimates & print max error
    # env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
    # 
    # ---My code---
    Q_sa_list = list()
    i = 0
    while True:
        delta, Q_sa = QIagent.update(env, gamma)
        Q_sa_list.append(np.array(Q_sa)) # for analysis only
        i += 1
        print("Q-value iteration, iteration {}, max error {}".format(i, delta))
        if delta < threshold:
            break
    
    # for analysis only: beginning, midway and convergence.
    env.render(Q_sa=Q_sa_list[0],plot_optimal_policy=True,step_pause=2)
    env.render(Q_sa=np.array(Q_sa_list[len(Q_sa_list)//2]),plot_optimal_policy=True,step_pause=2)
    env.render(Q_sa=np.array(Q_sa_list[-1]),plot_optimal_policy=True,step_pause=2)
    return QIagent

def experiment():
    # Value iteration using action-value function (i.e., Q(s,a))
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True) # initialize environment
    env.render()
    QIagent = Q_value_iteration(env, gamma, threshold) # update
    
    # View optimal policy
    done = False
    s = env.reset()
    rewards = []
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        rewards.append(r)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.005)
        s = s_next

    mean_reward_per_timestep = np.mean(rewards) #
    # TO DO: Compute mean reward per timestep under the optimal policy
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))

if __name__ == '__main__':
    experiment()
