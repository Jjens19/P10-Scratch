import numpy as np
import os
import time
import subprocess

import random

import matplotlib.pyplot as plt
def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)



# Network selection

from ppo_networks.ppo_agent_LSTM_simple import Agent
#from ppo_networks.ppo_agent_3_dense import Agent



# ------------- CONFIG ------------------------------------

# MODEL NAME <--------------------- 
model_name = "ppo_proto_v1"

# RNN Memory duration (0 or 1 for non-RNN network)
time_steps = 10

# PATHS
# Path to your ns-3 script file
ns3_script = "scratch/q-training_line.cc"

# Path for CWND size file
cwnd_path = "scratch/rate.txt"
cwnd_start = random.randint(512, 512 * 20) #1         * 512 # Multiply with minimum CWND size

# Q-Model folder path
model_folder_path = "models"
os.makedirs(model_folder_path, exist_ok=True)

# Training duration
sample_frequency = 1 # Amount of steps/samples per second
episodes = 2 
steps_max = 400             * sample_frequency # First number indicates training duration in simulated seconds

# Batch size - Amount of steps done before training on them at once
batch_size = 64 

# Number of batches to collect before training on them seperatly
n_epochs = 1

# Learning hyperparameters
learning_rate = 0.05
discount_rate = 0.95

# Exploration
exploration_rate       = 1
exploration_rate_max   = 1
exploration_rate_min   = 0.1
exploration_rate_decay = 0.005



# List of possible actions
actions = [False,             # Nothing
           lambda x: x + 10,  # Increase by 10
           lambda x: x - 1,   # Decrease by 1
           lambda x: x + 1.1, # Increase by 10%
           lambda x: x + 0.9  # Decrease by 10%
           ]

# State sizes
state_size_send    = 12 # Different states based on amount of packets send
state_size_acks    = 12 # Different states based on amount of packets dropped
state_size_rtt     = 12 # Different states based on rtt
states = (state_size_send, state_size_acks, state_size_rtt)




# Reward factors TODO
reward_factor_bytes = 10 # 
reward_factor_rtt   = 1 # 
reward_factor_ack_ratio = 1 #
utility = 0 
utility_threshold = 0.9 # How big should a change be before a reward/penalty is given
utility_reward = 5    # Size of reward/penalty

last_rtt = None

sleep_time = 0.01 # Delay between checking subprocess buffer
cwnd_log = ""
reward_log = ""
state_log = ""

best_score = -np.infty
score_history = []
figure_file = "models/rewards.png"
# ------------- Functions ----------------------------------

def get_next_state(process): 
    
    # read the simulation output or wait for it to finish
    reading = process.stdout.readline()
    temp_list = reading.split('\n')[0].split(',')
    [send, acks, n_bytes, rtt] = [int(xx) for xx in temp_list] 
    
    global state_log
    state_log += reading

    # Divide send and acks over period 
    send, acks = send/sample_frequency, acks/sample_frequency

    # Convert dropped to percent? 
    ack_perc = 1 if send == 0 else acks/send

    
    finished = False
    
    state_array = np.array([send, acks, rtt])

    return np.reshape(state_array, (1,3)) , (n_bytes, ack_perc, finished)
    return (state_send, state_acks, state_rtt), (n_bytes, ack_perc, finished)

def get_reward(n_bytes, ack_perc, rtt): 
    # Declare utility as global... otherwise errors
    global utility
    global last_rtt


    if n_bytes == 0: n_bytes = 1
    if ack_perc == 0: ack_perc = 1
    if rtt == 0: rtt = last_rtt
    
    last_rtt = rtt
    
    utility_new = (np.log2(n_bytes) * reward_factor_bytes - np.log2(rtt) * reward_factor_rtt) #* ack_perc 
    return utility_new
    reward = 0
    
    
    if   utility_new - utility > utility_threshold: reward = utility_reward
    elif utility - utility_new > utility_threshold: reward = -utility_reward
    
    utility = utility_new

    return reward

def perform_action(process, action):
    global cwnd_log  
    # Change CWND in file
    if action != False:
        cwnd = 0
        with open(cwnd_path, 'r') as file:
            cwnd = float(file.read()) // 512 # Divide with minimum size of packets ...
            #print(cwnd, action)
            
            cwnd = actions[action](cwnd) * 512 # Multiply with minimum size of packets ...
            
        cwnd = int(cwnd // 1) # Only whole amounts of bytes
        
        if cwnd < 512: cwnd = 512
        cwnd_log += f"{cwnd}\n"
        with open(cwnd_path, 'w') as file:
            file.write(f"{cwnd}")
            
    else:
        with open(cwnd_path, 'r') as file:
            cwnd = int(file.read())
            cwnd_log += f"{cwnd}\n"
    
    # Inform subprocess new episode is ready 
    process.stdin.write(f"next state\n")
    process.stdin.flush()
    
    # get new state 
    new_state, (n_bytes, ack_perc, finished) = get_next_state(process) 
    
    
    
    # Calculate reward based on new state 
    #reward = get_reward(n_bytes, ack_perc, state[0][2])
    reward = get_reward(state[0][1], ack_perc, state[0][2])
    
    return new_state, reward, finished


# ------------- Initialization -----------------------------


agent = Agent(n_actions = len(actions),     # Action space
              batch_size= batch_size,       # Training batch size
              alpha     = learning_rate,    # Learning rate
              n_epochs  = n_epochs,         # Learning epochs             
              input_dims= len(states),      # State space 
              chkpt_dir = model_folder_path + '/')
# Reward storage
rewards_all_episodes = []


# ------------- Training loops ---------------------------

for episode in range(episodes): #-----------------------------------Episode------------------------------------------
    reward_episode = 0
    # Reset congestion window
    with open(cwnd_path, 'w') as file:
    	
        file.write(f"{cwnd_start}")
	
    # Start NS3 Simulation as subprocess
    process = subprocess.Popen(["./ns3", "run", ns3_script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while "Simulation" not in process.stdout.readline(): 
        time.sleep(sleep_time)
	
    process.stdin.write(f"next state\n")
    process.stdin.flush()
    

    
    # Get first state
    state, (n_bytes, ack_perc, _)  = get_next_state(process)

    get_reward(n_bytes, ack_perc, state[0][2]) # Set utulity value
    
    if time_steps != 0 or time_steps != 1:
        padding = np.zeros((time_steps-1, len(states)))
        state = np.vstack((padding, state))
    
    for step in range(steps_max): # ----------------------------------------STEP-----------------------------------------
    	#print("step",step)
    	
        # Get action
        action, probabilities,  val = agent.choose_action(state)
        
        #print(action, probabilities, val)

        #print(state, action)
        if np.any(np.isnan(probabilities)): print("NAN FOUND")


    	# Perform action and get rewards/info
        new_state, reward, done = perform_action(process, action)
    	
        reward_episode += reward
        reward_log += f"{reward}\n"
        

        agent.store_transition(state, action, probabilities, val, reward, done)
    	# Learn on batch
        if step % batch_size == batch_size - 1 or step == steps_max - 1:
            agent.learn()
        

        if time_steps == 0 or time_steps == 1:
            state = new_state
        elif time_steps > 1:
            state = np.vstack((state, new_state))
            if len(state) > time_steps: 
                state = state[1:]
        else:
            exit("Wrong time_steps variable value")

        # Last part of episode code goes here
        if done or step == steps_max - 1:
            break
        #  ------------------------------------------------STEP END------------------------------------------------ 

    # Close process
    process.terminate()
    
    # Save reward
    rewards_all_episodes.append(reward_episode)
    
    score_history.append(reward_episode)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    
    # Print completed epochs
    print('episode', episode, 'score %.1f' % reward_episode, 'avg score %.1f' % avg_score)
    #print(f"Episode {episode} complete, reward of: {reward_episode}")
    
    with open("scratch/cwnd_log.txt", 'w') as file:
        
        file.write(f"{cwnd_log}")
    
    cwnd_log = ""
    with open("scratch/reward_log.txt", 'w') as file:
        file.write(f"{reward_log}")
    reward_log = ""
    with open("scratch/state_log.txt", 'w') as file:
        file.write(f"{state_log}")
    state_log = ""
    


x = [i+1 for i in range(len(score_history))]
plot_learning_curve(x, score_history, figure_file)


#agent.save_models()

print("Reward over episodes")
for ii, reward in enumerate(rewards_all_episodes): print(ii,reward)

