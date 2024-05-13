import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

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
           lambda x: x * 1.1, # Increase by 10%
           lambda x: x * 0.9  # Decrease by 10%
           ]

# State sizes
state_size_send    = 12 # Different states based on amount of packets send
state_size_acks    = 12 # Different states based on amount of packets dropped
state_size_rtt     = 12 # Different states based on rtt
states = (state_size_send, state_size_acks, state_size_rtt)

# Reward factors TODO
reward_factor_bytes = 1 # 
reward_factor_rtt   = 10 # 
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
    
    # Check if simulation should continue (usually should)
    finished = True if state_acks + state_rtt + state_send == 0 else False
    finished = False
    
    state_array = np.array([send, acks, rtt])

    return np.reshape(state_array, (1,3)) , (n_bytes, ack_perc, finished)
    return (state_send, state_acks, state_rtt), (n_bytes, ack_perc, finished)

def get_reward(n_bytes, ack_perc, state_rtt): 
    # Declare utility as global... otherwise errors
    global utility
    
    if n_bytes == 0 and state_rtt == 0: return -50

    if n_bytes == 0: n_bytes = 1
    if ack_perc == 0: ack_perc = 1
    
    
    utility_new = (np.log2(n_bytes) * reward_factor_bytes - state_rtt * reward_factor_rtt) #* ack_perc 

    reward = 0
    
    
    if   utility_new - utility > utility_threshold: reward = utility_reward
    elif utility - utility_new > utility_threshold: reward = -utility_reward
    
    utility = utility_new

    return utility_new #reward

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


# ------------- Tensorflow ---------------------------------


def create_policy_network(input_shape, output_shape):
    model = models.Sequential([
        layers.Dense(32, activation='relu', input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_shape, activation='softmax')
    ])
    return model

def calculate_advantages(rewards, gamma=0.99):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for ii in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[ii]
        advantages[ii] = running_add
    # Standardize advantages
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    return advantages.tolist()

class PPOAgent:
    def __init__(self, input_shape, output_shape, learning_rate=0.01, gamma=0.99):
        self.policy_network = create_policy_network(input_shape, output_shape)
        self.optimizer = optimizers.Adam(learning_rate)
        self.gamma = gamma

    def get_action(self, state):
        # Get action probabilities from the policy network
        action_probs = self.policy_network.predict(state, verbose=0)[0]
        action = np.random.choice(len(action_probs), p=action_probs)
        return action, action_probs[action]

    def train(self, states, actions, advantages, probs):
        with tf.GradientTape() as tape:
            # Calculate new action probabilities
            new_probs = self.policy_network(states)
            action_masks = tf.one_hot(actions, depth=new_probs.shape[1])
            chosen_probs = tf.reduce_sum(action_masks * new_probs, axis=1)
            ratios = chosen_probs / probs

            # Calculate PPO loss
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=0.8, clip_value_max=1.2)
            surrogate1 = ratios * advantages
            surrogate2 = clipped_ratios * advantages
            loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

        # Perform gradient descent
        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))




# ------------- Initialization -----------------------------

# Create agent
agent = PPOAgent((len(states),), len(actions))

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
    training_batch = [[], [], [], [], []] # states, actions, rewards, old_probs
    

    # Get first state
    state, (n_bytes, ack_perc, _)  = get_next_state(process)
    #print(state.shape)
    get_reward(n_bytes, ack_perc, state[0][2]) # Set utulity value
    
    for step in range(steps_max):
    	#print("step",step)
    	
        training_batch[0].append(state)
        
        # Get action
        action, probabilities = agent.get_action(state)

        training_batch[1].append(action)
        training_batch[3].append(probabilities)

        #print(state, action)

    	# Perform action and get rewards/info
        new_state, reward, finished = perform_action(process, action)
    	
        training_batch[2].append(reward)
        reward_episode += reward
        reward_log += f"{reward}\n"
        
        #print(new_state, reward)


    	# Learn on batch
        if step % batch_size == batch_size - 1 or step == steps_max - 1:

            # Calculate advantages
            advantages = calculate_advantages(np.reshape(training_batch[2], (len(training_batch[2]),1)))
            #print(training_batch[2],advantages)
            

            # Train with - States, actions, advantages, probabilities
            agent.train(np.array(training_batch[0][-1]), np.array(training_batch[1][-1]), np.array(advantages), np.array(training_batch[3][-1]))
            
            # reset batch variable
            training_batch = [[], [], [], [], []] # states, actions, rewards, old_probs, advantages

        # Last part of episode code goes here
        if finished or step == steps_max - 1:
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
    
    if episode % 20 == 19: agent.policy_network.save(f"{model_folder_path}/{model_name}.keras")
	
#np.save(f"{model_folder_path}/{model_name}.npy", q_matrix)
agent.policy_network.save(f"{model_folder_path}/{model_name}.keras")
print("Reward over episodes")
for ii, reward in enumerate(rewards_all_episodes): print(ii,reward)
