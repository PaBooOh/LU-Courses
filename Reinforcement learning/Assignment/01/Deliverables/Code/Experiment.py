#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
import time

from Q_learning import q_learning
from SARSA import sarsa
from MonteCarlo import monte_carlo
from Nstep import n_step_Q
from Helper import LearningCurvePlot, smooth

def average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, gamma, policy='egreedy', 
                    epsilon=None, temp=None, smoothing_window=51, plot=False, n=5):

    reward_results = np.empty([n_repetitions,n_timesteps]) # Result array
    now = time.time()
    
    for rep in range(n_repetitions): # Loop over repetitions
        if backup == 'q':
            rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
        elif backup == 'sarsa':
            rewards = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
        elif backup == 'mc':
            rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
        elif backup == 'nstep':
            rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)

        reward_results[rep] = rewards
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))    
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve  

def experiment():
    ####### Settings
    # Experiment    
    n_repetitions = 50
    smoothing_window = 1001

    # MDP    
    n_timesteps = 50000
    max_episode_length = 20
    gamma = 1.0

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.05
    temp = 1.0
    
    # Target and update
    backup = 'q' # 'q' or 'sarsa' or 'mc' or 'nstep'
    learning_rate = 0.25
    n = 5
        
    # Plotting parameters
    plot = True
    
    # Nice labels for plotting
    policy_labels = {'egreedy': '$\epsilon$-greedy policy',
                  'softmax': 'Softmax policy'}

    backup_labels = {'q': 'Q-learning',
                  'sarsa': 'SARSA',
                  'mc': 'Monte Carlo',
                  'nstep': 'n-step Q-learning'}
    
    ####### Experiments
    
    #### Assignment 1: Dynamic Programming
    # Execute this assignment in DynamicProgramming.py
    optimal_average_reward_per_timestep = 1.1176 # set the optimal average reward per timestep you found in the DP assignment here
    
    #### Assignment 2: Effect of exploration
    backup = 'q'
    Plot = LearningCurvePlot(title = 'Q-learning: effect of $\epsilon$-greedy versus softmax exploration')    
    policy = 'egreedy'
    epsilons = [0.01,0.05,0.2]
    for epsilon in epsilons:        
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n)
        Plot.add_curve(learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))    
    policy = 'softmax'
    temps = [0.01,0.1,1.0]
    for temp in temps:
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n)
        Plot.add_curve(learning_curve,label=r'softmax, $ \tau $ = {}'.format(temp))
    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('exploration.png')
    policy = 'egreedy'
    epsilon = 0.05 # set epsilon back to original value 
    temp = 1.0
    
    
    ###### Assignment 3: Q-learning versus SARSA
    backups = ['q','sarsa']
    learning_rates = [0.05,0.2,0.4]
    Plot = LearningCurvePlot(title = 'Q-learning versus SARSA')    
    for backup in backups:
        for learning_rate in learning_rates:
            learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                  gamma, policy, epsilon, temp, smoothing_window, plot, n)
            Plot.add_curve(learning_curve,label=r'{}, $\alpha$ = {} '.format(backup_labels[backup],learning_rate))
    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('on_off_policy.png')
    # Set back to original values
    learning_rate = 0.25
    backup = 'q'


    # ##### Assignment 4: Back-up depth
    backup = 'nstep'
    ns = [1,3,5,10,20,100]
    Plot = LearningCurvePlot(title = 'Effect of target depth')    
    for n in ns:
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n)
        Plot.add_curve(learning_curve,label=r'{}-step Q-learning'.format(n))
    backup = 'mc'
    learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                          gamma, policy, epsilon, temp, smoothing_window, plot, n)
    Plot.add_curve(learning_curve,label='Monte Carlo')        
    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('depth.png')

if __name__ == '__main__':
    experiment()