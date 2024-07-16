# DQN.py
## Requirements
* tensorflow
* numpy
* gym
* signal
* scipy
* matplotlib
## Usage
* set the hyperparameters at the top part of the python script DQN.py
(we already set the hyperparameters, if you want to change the action choosing policy other than
'egreedy', you have to modify the script in "POLICY = 'egreedy' part")
* to run this file, run the following three command(representing 3 kinds of experiment)
** python DQN.py
    This command will use the hyperparameters setted in the script, it would use the replay buffer and target net
** python DQN.py --replay
    This command will remove the replay buffer to run the experiment
** python DQN.py --target
    This command will reomve the target net to run the experiment
** python DQN.py --replay --target
    This command will remove both replay buffer and target net to run the experiment

## Results saving
* After a period of training, you can terminate the program at any time through ctrl+c on the command line, 
and the program will automatically save the training result picture to the same directory as this py file. 
The resulting image contains the model loss curve from training to the current time and 
the curve of the number of steps the agent can persist in each episode. If not manually terminated, 
the model will continue to train until the average persistence steps of the last 30 episodes are greater than 475.

