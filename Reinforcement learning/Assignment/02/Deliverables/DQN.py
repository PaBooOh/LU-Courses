
import numpy as np
import sys
import random
import gym
import time
import signal
from collections import deque
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from scipy.signal import savgol_filter
import math
import matplotlib.pyplot as plt

# (Hyper)parameters
# ------------------------------------------------------Policy
# Policy setting
POLICY = 'egreedy'
# epsilon-greedy
EPSILON = 0.1
# Boltzmann
TEMP = 1
# annealing epsilon-greedy
ANNEALING_EPSILON = 1.0
ANNEALING_EPSILON_MIN = 0.01  
ANNEALING_EPSILON_DECAY = 0.9995
#--------------------------------------------------------Trainning
LR = 0.0001
BATCH_SIZE = 128
EPISODES = 100000
OPTIM = Adam(lr=LR)
TRAINING_LIMITATION = BATCH_SIZE
#--------------------------------------------------------Reinforecement learning
GAMMA = 0.95
REPLAY_CAPACITY = 5000
SET_WEIGHTS_FREQENCY = 50
#--------------------------------------------------------Environment
ENVNAME = 'CartPole-v1'
#--------------------------------------------------------Plot
avg_list = []
step_list = []
if POLICY == 'egreedy':
    FIG_FILE = 'Lr={}, policy={}, epsilon={}, batchsize={}, buffer_size={}, SET_WEIGHTS_FREQENCY={}, GAMMA={}.png'.format(LR, POLICY, EPSILON, BATCH_SIZE, REPLAY_CAPACITY, SET_WEIGHTS_FREQENCY, GAMMA)
elif POLICY == 'annealing_egreedy':
    FIG_FILE = 'Lr={}, policy={}, epsilon={}, epsilon_min={}, epsilon_decay={}, batchsize={}, buffer_size={}, SET_WEIGHTS_FREQENCY={}, GAMMA={}.png'.format(LR, POLICY, ANNEALING_EPSILON, ANNEALING_EPSILON_MIN, ANNEALING_EPSILON_DECAY, BATCH_SIZE, REPLAY_CAPACITY, SET_WEIGHTS_FREQENCY, GAMMA)
elif POLICY == 'boltzmann':
    FIG_FILE = 'Lr={}, policy={}, temp={}, batchsize={}, buffer_size={}, SET_WEIGHTS_FREQENCY={}, GAMMA={}.png'.format(LR, POLICY, TEMP, BATCH_SIZE,  REPLAY_CAPACITY, SET_WEIGHTS_FREQENCY, GAMMA)
#--------------------------------------------------------Command
cmd_arg = sys.argv[1:]
assert len(cmd_arg) in [0, 1, 2], 'The number of args exceeds the limit. Expect less than two.'
if len(cmd_arg) == 0:
    BATCH_SIZE = 1
    TRAINING_LIMITATION = BATCH_SIZE
    REPLAY_CAPACITY = 1
    SET_WEIGHTS_FREQENCY = 1
elif len(cmd_arg) == 1:
    assert cmd_arg[0] in ['--replay', '--target'] , 'The name of Args is unknown. Expect --replay, --target or None.'
    if cmd_arg[0] == '--replay':
        SET_WEIGHTS_FREQENCY = 1
    else:
        BATCH_SIZE = 1
        TRAINING_LIMITATION = BATCH_SIZE
        REPLAY_CAPACITY = 1
        
elif len(cmd_arg) == 2:
    assert cmd_arg[0] in ['--replay', '--target'], 'The name of  Args is unknown. Expect --replay, --target or None.'
    assert cmd_arg[1] in ['--replay', '--target'], 'The name of Args is unknown. Expect --replay, --target or None.'

# other helpful funcs
def softmax(x, temp):
    x = x / temp 
    z = x - max(x) 
    return np.exp(z) / np.sum(np.exp(z))

def handler(signum, frame):
    msg = "Ctrl-C was pressed. Screenshot was saved at the current local folder."
    print()
    print(msg, end="", flush=True)
    plotPerformance(avg_list, step_list)
    exit(1)

def plotPerformance(avg_loss_list, step_counter_list):
    cut = 0
    for i in range(len(avg_loss_list)):
        if not math.isnan(avg_loss_list[i]):
            # print(i)
            cut = i
            break
    step_counter_list = step_counter_list[cut:]
    avg_loss_list = avg_loss_list[cut:]
    x = np.arange(0,len(step_counter_list))
    y1 = step_counter_list
    y2 = avg_loss_list

    y1_smooth = savgol_filter(y1, 51, 3)
    y2_smooth = savgol_filter(y2, 51, 3)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, y1_smooth, 'g-',label='total steps')
    ax2.plot(x, y2_smooth, 'b--',label='average loss')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ax1.set_xlabel('Training episode')
    ax1.set_ylabel('total steps')
    ax2.set_ylabel('average loss')
    plt.savefig(FIG_FILE)

    # print(step_counter_list)
    # print(len(step_counter_list))
    # print(avg_loss_list)
    # print(len(avg_loss_list))

class Network():
    def __init__(self, N_STATES, N_ACTIONS) -> None:
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
    def Qnet_FC(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.N_STATES, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.N_ACTIONS, activation='linear', kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=OPTIM, metrics=['mse'])
        return model

    def Qnet(self):
        X_inputs = Input(shape=(self.N_STATES,))
        # 1st
        X = Conv2D(16, 3, activation='relu')(X_inputs)
        X = MaxPooling2D((2, 2))(X)
        # 2rd
        X = Conv2D(32, 3, activation='relu')(X)
        X = MaxPooling2D((2, 2))(X)
        X = Dropout(0.25)(X)
        X = Flatten()(X)

        X = Dense(32, activation = 'relu')(X)
        X = Dropout(0.25)(X)
        X = Dense(self.N_ACTIONS, activation = 'linear')(X)

        model = Model(inputs = X_inputs, outputs = X)
        return model
    
class DQNAgent():
    def __init__(self, N_STATES, N_ACTIONS):
        nn = Network(N_STATES, N_ACTIONS)
        self.original_model = nn.Qnet_FC()
        self.target_model = nn.Qnet_FC()
        self.replay_buffer = deque(maxlen=REPLAY_CAPACITY)
        self.training_counts = 0
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.policy = POLICY
        self.annealing_egreedy_epsilon = ANNEALING_EPSILON
        self.annealing_egreedy__epsilon_min = ANNEALING_EPSILON_MIN
        self.annealing_egreedy_epsilon_decay = ANNEALING_EPSILON_DECAY
    
    def actEgreedy(self, S):
        S = np.reshape(S, (1, self.N_STATES))
        if np.random.rand() >= EPSILON: # Exploit
            Q = self.original_model.predict(S)
            A = np.argmax(np.squeeze(Q))
        else:
            A = np.random.randint(0, self.N_ACTIONS) # Explore
        return A
    
    def actBoltzmann(self, S):
        S = np.reshape(S, (1, self.N_STATES))
        Q = self.original_model.predict(S)
        probs = softmax(np.squeeze(Q), TEMP)
        A = np.random.choice(range(self.N_ACTIONS), 1, p=np.squeeze(probs))[0]
        return A
    
    def actAnnealingEgreedy(self, S):
        S = np.reshape(S, (1, self.N_STATES))
        if np.random.rand() < self.annealing_egreedy_epsilon: # Explore in the beginning (1.0)
            A = np.random.randint(0, self.N_ACTIONS)
        else:
            Q = self.original_model.predict(S)
            A = np.argmax(np.squeeze(Q))
        return A
    
    def appendExperienceReplay(self, S, A, R, S_next, done):
        transition = (S, A, R, S_next, done)
        self.replay_buffer.append(transition)
    
    def sampleData(self):
        return random.sample(self.replay_buffer, BATCH_SIZE)
    
    def updateTargetModel(self):
        # print('Update target model')
        self.target_model.set_weights(self.original_model.get_weights())
    
    def training(self):
        if self.training_counts % SET_WEIGHTS_FREQENCY == 0:
            self.updateTargetModel()
        self.training_counts += 1
        batch_data = self.sampleData()
        S_batch, A_batch, R_batch, S_next_batch, terminal_batch = [], [], [], [], []
        for data in batch_data:
            S_batch.append(data[0])
            A_batch.append(data[1])
            R_batch.append(data[2])
            S_next_batch.append(data[3])
            terminal_batch.append(data[4])
        
        S_batch = np.array(S_batch)
        S_next_batch = np.array(S_next_batch)
        
        Q_batch = self.original_model(S_batch)
        Q_target_batch = np.array(Q_batch, copy=True)
        Q_next_batch = self.target_model(S_next_batch)
        for i in range(BATCH_SIZE):
            terminal = terminal_batch[i]
            Q_target = R_batch[i] if terminal else R_batch[i] + GAMMA * np.max(Q_next_batch, axis=-1)[i]
            Q_target_batch[i][A_batch[i]] = Q_target
        
        result = self.original_model.fit(x=S_batch, y=Q_target_batch, verbose=0)
        if self.policy == 'annealing_egreedy':
            if self.annealing_egreedy_epsilon > self.annealing_egreedy__epsilon_min:
                self.annealing_egreedy_epsilon = self.annealing_egreedy_epsilon * self.annealing_egreedy_epsilon_decay
        return result.history

def main():
    global avg_list # for plot
    global step_list
    ENV = gym.make(ENVNAME)  # make game env
    N_STATES = ENV.observation_space.shape[0] # 4
    N_ACTIONS = ENV.action_space.n # 2
    Dqn_agent = DQNAgent(N_STATES, N_ACTIONS)
    step_counts_list = []
    scores = []
    avg_loss_list = []
    win_most_recent_30 = deque(maxlen=30)
    # print(SET_WEIGHTS_FREQENCY, REPLAY_CAPACITY, TRAINING_LIMITATION)
    for episode in range(EPISODES):
        S = ENV.reset()
        step_counts = 0
        score = 0
        loss_list = []
        if np.mean(win_most_recent_30) >= 475:
            print('Reach good model')
            break
        while True:
            # ENV.render()
            if Dqn_agent.policy == 'egreedy':
                A = Dqn_agent.actEgreedy(S)
            elif Dqn_agent.policy == 'boltzmann':
                A = Dqn_agent.actBoltzmann(S)
            elif Dqn_agent.policy == 'annealing_egreedy':
                A = Dqn_agent.actAnnealingEgreedy(S)
            else:
                print('Please choose a policy')
                return
            
            S_next, R, terminal, _ = ENV.step(A) # perform an action
            step_counts += 1
            if terminal:
                if step_counts == 500:
                    R = 10
                elif step_counts < 500:
                    R = -R # If fail, suffer punishment
            else:
                R = R
            Dqn_agent.appendExperienceReplay(S, A, R, S_next, terminal) # Experience replay
            current_buffer_size = len(Dqn_agent.replay_buffer)
            if current_buffer_size >= TRAINING_LIMITATION:
                history = Dqn_agent.training() # training if volume is greater than batch size.
                loss_list.append(history['loss'])
            
            S = S_next
            score += R
            # End if there are enough winnings.
            if terminal:
                # Dqn_agent.target_model.set_weights(Dqn_agent.original_model.get_weights()) # Update target model every episode
                win_most_recent_30.append(step_counts)
                step_counts_list.append(step_counts)
                break
            
        scores.append(score)
        average_loss = np.mean(loss_list)
        avg_loss_list.append(average_loss)
        
        avg_list = avg_loss_list
        step_list = step_counts_list
        print("Episode: {}, Total reward: {}, Total step: {}".format(episode, score, step_counts_list[-1]))
    print('Scores: ', scores)
    print('Steps: ', step_counts_list)
    return avg_loss_list, step_counts_list
    
if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    a = time.time()
    avg_loss_list, step_counts_list = main()
    b = time.time()
    print('Total time', b - a)
    plotPerformance(avg_loss_list, step_counts_list)