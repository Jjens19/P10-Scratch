import os
import numpy as np
import random
import time
import subprocess
# ------------- CONFIG ------------------------------------

# MODEL NAME <--------------------- 
model_name = "test4"



# PATHS
# Path to your ns-3 script file
ns3_script = "scratch/test3.cc"

# Path for CWND size file
cwnd_path = "scratch/rate.txt"
cwnd_start = 1         * 512 # Multiply with minimum CWND size

# Q-Model folder path
model_folder_path = "models"
os.makedirs(model_folder_path, exist_ok=True)

# Training duration
sample_frequency = 1 # Amount of steps/samples per second
episodes = 1
steps_max = 400 * sample_frequency

# Learning hyperparameters
discount_rate = 0.95


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
reward_factor_rtt   = 1 # 
#reward_factor_ack_ratio = 1
utility = 0 
utility_threshold = 0.9 # How big should a change be before a reward/penalty is given
utility_reward = 5    # Size of reward/penalty

sleep_time = 0.01 # Delay between checking subprocess buffer
cwnd_list = ""
reward_list = ""
state_list = ""
action_list = ""
# ------------- Functions ----------------------------------

def get_next_state(process): 
    
    # read the simulation output or wait for it to finish
    reading = process.stdout.readline()
    temp_list = reading.split('\n')[0].split(',')
    [send, acks, n_bytes, rtt] = [int(xx) for xx in temp_list] 
    
    global state_list
    state_list += reading

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
    
    return (state_send, state_acks, state_rtt), (n_bytes, ack_perc, finished)

def get_reward(n_bytes, ack_perc, state_rtt): 
    # Declare utility as global... otherwise errors
    global utility
    
    if n_bytes == 0: n_bytes = 1
    if ack_perc == 0: ack_perc = 1
    
    
    utility_new = (np.log2(n_bytes) * reward_factor_bytes - state_rtt * reward_factor_rtt) #* ack_perc 

    reward = 0
    
    
    if   utility_new - utility > utility_threshold: reward = utility_reward
    elif utility - utility_new > utility_threshold: reward = -utility_reward
    
    utility = utility_new

    return reward

def perform_action(process, action):
	global cwnd_list  
	# Change CWND in file
	if action != False:
		cwnd = 0
		with open(cwnd_path, 'r') as file:
			cwnd = float(file.read()) / 512 # Divide with minimum size of packets ...
			
			cwnd = actions[action](cwnd) * 512 # Multiply with minimum size of packets ...
			
		cwnd = int(cwnd // 1) # Only whole amounts of bytes
		if cwnd < 512: cwnd = 512
		cwnd_list += f"{cwnd}\n"
		with open(cwnd_path, 'w') as file:
			file.write(f"{cwnd}")
	else:
		with open(cwnd_path, 'r') as file:
			cwnd = int(file.read())
			cwnd_list += f"{cwnd}\n"

    # Inform subprocess new episode is ready 
	process.stdin.write(f"next state\n")
	process.stdin.flush()
    
    # get new state 
	new_state, (n_bytes, ack_perc, finished) = get_next_state(process) 
    
	
	
    # Calculate reward based on new state 
	reward = get_reward(n_bytes, ack_perc, state[2])

	return new_state, reward, finished


# ------------- Initialization -----------------------------

# Load or create q-matrix
q_matrix = np.load(f"{model_folder_path}/{model_name}.npy")

# Reset congestion window
with open(cwnd_path, 'w') as file:
    file.write(f"{cwnd_start}")


# Reward storage
rewards_all_episodes = []


# ------------- Q-learning loopss ---------------------------

for episode in range(episodes):
    reward_episode = 0
    # Start NS3 Simulation as subprocess	 
    process = subprocess.Popen(["./ns3", "run", ns3_script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while "Simulation" not in process.stdout.readline(): 
    	time.sleep(sleep_time)
    
    process.stdin.write(f"next state\n")
    process.stdin.flush()
    
    # Get first state
    state, (n_bytes, ack_perc, _)  = get_next_state(process)
    get_reward(n_bytes, ack_perc, state[2]) # Set utulity value
    
    for step in range(steps_max):
    	#print("step",step)
    	
    	#Exploit
    	action = np.argmax(q_matrix[state[0],state[1],state[2]])
    	action_list += f"{action}\n"
    	
    	#print(state, action)
    	# Perform action and get rewards/info
    	new_state, reward, finished = perform_action(process, action)
    	#print(new_state, reward)
    	
    	# Update variables
    	
    	state = new_state
    	reward_episode += reward
    	reward_list += f"{reward}\n"
    	
    	if finished:
    		break
    # Close process
    process.terminate()
    
    # Save reward
    rewards_all_episodes.append(reward_episode)
    
    
    # Print completed epochs
    print(f"Episode {episode} complete, reward of: {reward_episode}")
    
    with open("scratch/cwnd_log.txt", 'w') as file:
    	
    	file.write(f"{cwnd_list}")
    	
    cwnd_list = ""
    with open("scratch/reward_log.txt", 'w') as file:
    	file.write(f"{reward_list}")
    reward_list = ""
    with open("scratch/state_log.txt", 'w') as file:
    	file.write(f"{state_list}")
    state_list = ""
    with open("scratch/action_log.txt", 'w') as file:
    	file.write(f"{action_list}")
    action_list = ""
    
	
print(q_matrix)
print("Reward over episodes")
for ii, reward in enumerate(rewards_all_episodes): print(ii,reward)
