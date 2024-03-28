import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import ppo_classes

import os
import numpy as np
import random
import time
import subprocess

# ------------- CONFIG ------------------------------------

# MODEL NAME <--------------------- 
model_name = "ppo_proto_v1"

# PATHS
# Path to your ns-3 script file
ns3_script = "scratch/q-training_line.cc"

# Path for CWND size file
cwnd_path = "scratch/rate.txt"
cwnd_start = 1         * 512 # Multiply with minimum CWND size

# Q-Model folder path
model_folder_path = "models"
os.makedirs(model_folder_path, exist_ok=True)

# Training duration
sample_frequency = 1 # Amount of steps/samples per second
episodes = 4000
steps_max = 400             * sample_frequency # First number indicates training duration in simulated seconds

# Batch size - Amount of steps done before training on them at once
batch_size = 64 

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
reward_factor_bytes = 1 # 
reward_factor_rtt   = 2 # 
#reward_factor_ack_ratio = 1
utility = 0 
utility_threshold = 0.9 # How big should a change be before a reward/penalty is given
utility_reward = 5    # Size of reward/penalty

sleep_time = 0.01 # Delay between checking subprocess buffer
cwnd_log = ""
reward_log = ""
state_log = ""
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

    # RTT in ms and average to amount of packets acknowledged
    #rtt = 0 if acks == 0 else -(-rtt // 1) / acks

    # Convert to states, log is used because it sounds like a good idea
    state_send = 0 if send == 0 else     max(0, min(state_size_send-1,    int(np.log2(send)//1))) # From 0 to 2048 (12) 
    
    state_acks = 0 if acks == 0 else     max(0, min(state_size_acks-1,    int(np.log2(acks)//1))) # From 0 to 2048 (12)
    
    state_rtt  = 0 if rtt  == 0 else     max(0, min(state_size_rtt-1,     int(np.log2(rtt)//1)))  # From 0 to 2048 (12)
    cwnd_start
    # Check if simulation should continue (usually should)
    finished = True if state_acks + state_rtt + state_send == 0 else False
    finished = False
    
    state_array = np.array([send, acks, rtt])

    return np.reshape(state_array, (1,3)) , (n_bytes, ack_perc, finished)
    return (state_send, state_acks, state_rtt), (n_bytes, ack_perc, finished)

def get_reward(n_bytes, ack_perc, rtt): 
    # Declare utility as global... otherwise errors
    global utility
    
    if n_bytes == 0: n_bytes = 1
    if ack_perc == 0: ack_perc = 1
    if rtt == 0: rtt = 1
    
    
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
            #print(cwnd)
            
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
    reward = get_reward(n_bytes, ack_perc, state[0][2])
    
    return new_state, reward, finished


# ------------- Initialization -----------------------------

# Create agent
#agent = ((len(states),), len(actions))
agent = ppo_classes.agent()

# Reward storage
rewards_all_episodes = []


# ------------- Training loops ---------------------------

for episode in range(episodes):
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
    

    # Training data list
    training_batch = [[], [], [], [], []] # states, actions, rewards, values, old_probs
    

    # Get first state
    state, (n_bytes, ack_perc, _)  = get_next_state(process)
    #print(state.shape)
    get_reward(n_bytes, ack_perc, state[0][2]) # Set utulity value
    
    for step in range(steps_max):
    	#print("step",step)
    	
        training_batch[0].append(state)
        

        # Get action
        action, probabilities = agent.act(state)
        #print(action, probabilities)

        training_batch[1].append(action)
        training_batch[4].append(probabilities)

        #print(state, action)
        if np.any(np.isnan(probabilities)): print("NAN FOUND")


    	# Perform action and get rewards/info
        new_state, reward, done = perform_action(process, action)
    	
        training_batch[2].append(reward)
        reward_episode += reward
        reward_log += f"{reward}\n"
        
        #print(new_state, reward)


    	# Learn on batch
        if step % batch_size == batch_size - 1 or step == steps_max - 1:
            
            # Calculate value of states, including the latest, but not used state
            values = agent.critic(np.array(training_batch[0] + [new_state]))

            # Calculate advantages                                    Rewards                      Dones (all 0)                     Values            Gamma                    
            distinct_rewards, advantages = ppo_classes.calc_advantage(np.array(training_batch[2]), np.zeros_like(training_batch[1]), np.array(values), 1)
            
            # Learn some amount of times?
            for learns in range(1):
                #           States                      Actions                       Advantages  Probs                        Distinct_rewards
                #print("Learn")
                agent.learn(np.array(training_batch[0]), np.array(training_batch[1]), advantages, np.array(training_batch[4]), np.array(distinct_rewards))
                #print("Learned")

            # Train with - States, actions, advantages, probabilities
            #agent.train(np.array(training_batch[0][-1]), np.array(training_batch[1][-1]), np.array(advantages), np.array(training_batch[3][-1]))
            
            # reset batch variable
            training_batch = [[], [], [], [], []] # states, actions, rewards, old_probs, advantages


        state = new_state

        # Last part of episode code goes here
        if done or step == steps_max - 1:
            break
            

    # Close process
    process.terminate()
    
    # Save reward
    rewards_all_episodes.append(reward_episode)
    
    # Update exploration rate
    exploration_rate = exploration_rate_min + (exploration_rate_max - exploration_rate_min) * np.exp(-exploration_rate_decay * episode)
    
    # Print completed epochs
    print(f"Episode {episode} complete, reward of: {reward_episode}")
    
    with open("scratch/cwnd_log.txt", 'w') as file:
        
        file.write(f"{cwnd_log}")
    
    cwnd_log = ""
    with open("scratch/reward_log.txt", 'w') as file:
        file.write(f"{reward_log}")
    reward_log = ""
    with open("scratch/state_log.txt", 'w') as file:
        file.write(f"{state_log}")
    state_log = ""
    
    if episode % 20 == 19: 
        agent.actor.save(f"{model_folder_path}/{model_name}_actor.keras")
        agent.critic.save(f"{model_folder_path}/{model_name}_critic.keras")
	
#np.save(f"{model_folder_path}/{model_name}.npy", q_matrix)
agent.actor.save(f"{model_folder_path}/{model_name}_actor.keras")
agent.critic.save(f"{model_folder_path}/{model_name}_critic.keras")
print("Reward over episodes")
for ii, reward in enumerate(rewards_all_episodes): print(ii,reward)
