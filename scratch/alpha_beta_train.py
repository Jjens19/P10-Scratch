import numpy as np
import os
import time
import subprocess

import random

import matplotlib.pyplot as plt


model_id = 2        # Select network 
training_n = 1      # Select training case

# RNN Memory duration (0 or 1 for non-RNN network)
time_steps = 15

load_existing_model = False

# ------------ Network selection -----------------------------------------------------------------------
match model_id:
    case 0:
        from ppo_networks.debug import Agent
    case 1:
        from ppo_networks.ppo_agent_3_dense import Agent
    case 2:
        from ppo_networks.ppo_agent_LSTM_simple import Agent



# ------------- CONFIG ------------------------------------

# PATHS
# Path to your ns-3 script file
training_network_names = ["line", "4_steps", "4_steps_tcp"]
ns3_script = f"scratch/training_networks/alpha_beta_train/alpha_beta_train.cc"

# Path for CWND size file
cwnd_path = "scratch/rate.txt"

packet_size = 512
cwnd_start = random.randint(packet_size, packet_size * 20) 

# Model folder path
alpha_values = [2.5, 5]# [2.5, 5]
beta_values = [0.05, 0.15] # [0, 0.1, 0.2, 0.3, 0.4, 0.5]
for alpha_value in alpha_values:
    for beta_value in beta_values:
        model_folder_path = f"scratch/models/alpha_beta/alpha{alpha_value}_beta{beta_value}"
        os.makedirs(model_folder_path, exist_ok=True)

        # Training duration
        sample_frequency = 1 # Amount of steps/samples per second
        episodes = 200 #200
        steps_max = 400             * sample_frequency # First number indicates training duration in simulated seconds

        # Batch size - Amount of steps done before training on them at once
        batch_size = 64 

        # Number of batches to collect before training on them seperatly
        n_epochs = 1

        # Learning hyperparameters
        learning_rate = 0.0005
        discount_rate = 0.99

        # Exploration
        exploration_rate       = 0
        exploration_rate_max   = 0
        exploration_rate_min   = 0
        exploration_rate_decay = 0



        # List of possible actions
        actions = [False,             # Nothing
                lambda x: x + 3,   # Increase by 10
                lambda x: x - 1,   # Decrease by 1
                lambda x: x * 1.25, # Increase by 10%
                lambda x: x * 0.75  # Decrease by 10%
                ]

        # State sizes
        states = (0,1,2,3)#(state_size_send, state_size_acks, state_size_rtt, rtt_dev)

        # Custom Max normalization initalial values
        max_norm_initials = [2000, 2000, 200, 200]

        # Reward factors 
        reward_factor_alpha = alpha_value 
        reward_factor_beta  = beta_value

        reward_factor_ack_ratio = 1 #
        utility = 0 
        utility_threshold = 0.9 # How big should a change be before a reward/penalty is given
        utility_reward = 5    # Size of reward/penalty

        # Simulation parameters and information
                                                        # Update value whenever test throughput changes
        throughput_test_list = [lambda x: [120e3] * (x//5) + [60e3] * (x//5) + [90e3] * (x//5) + [30e3] * (x//5) + [120e3] * (x - 4*(x//5)), # Train 1 - 120kBps -> 60kBps -> 90kBps -> 30kBps -> 120kBps
                                lambda x: [60e3] * (x//5) + [30e3] * (x//5) + [45e3] * (x//5) + [15e3] * (x//5) + [60e3] * (x - 4*(x//5)),
                                lambda x: [120e3] * x]
        throughput_available = throughput_test_list[training_n - 1](steps_max)




        last_rtt = None

        sleep_time = 0.01 # Delay between checking subprocess buffer
        cwnd_log = ""
        reward_log = ""
        state_log = ""

        best_score = -np.infty
        score_history = []
        figure_file = model_folder_path + "/rewards.png"
        # ------------- Functions ----------------------------------

        def plot_learning_curve(x, scores, figure_file):
            running_avg = np.zeros(len(scores))
            for i in range(len(running_avg)):
                running_avg[i] = np.mean(scores[max(0, i-30):(i+1)])
            plt.plot(x, running_avg, label='Running Average')
            plt.title('Average Score Over Episodes')
            plt.xlabel('Episodes')
            plt.ylabel('Average Score')
            plt.legend()
            plt.savefig(figure_file, dpi=500)


        def get_next_state(process): 
            
            # read the simulation output or wait for it to finish
            reading = process.stdout.readline()
            temp_list = reading.split('\n')[0].split(',')
            [send, acks, send_bytes, received_bytes, rtt, rtt_dev] = [float(xx) for xx in temp_list] 
            
            global state_log
            state_log += reading

            # Divide send and acks over period 
            send, acks = send/sample_frequency, acks/sample_frequency

            # Convert dropped to percent? 
            ack_perc = 1 if send == 0 else acks/send

            
            finished = False
            
            state_array = np.array([send, acks, rtt, rtt_dev])

            return np.reshape(state_array, (1,4)) , (received_bytes, send_bytes, ack_perc, finished)
            return (state_send, state_acks, state_rtt), (n_bytes, ack_perc, finished)

        def get_reward(received_bytes, send_bytes, rtt): 
            # Declare utility as global... otherwise errors
            global utility
            global last_rtt

            global throughput_available
            global step

            if rtt == 0: rtt = last_rtt
            
            last_rtt = rtt
            
            reward_alpha = 0

            
            # If no data is send with no capacity 
            if send_bytes == 0 and throughput_available == 0:
                return reward_factor_alpha
            elif send_bytes != 0 and throughput_available != 0:
                c_utility = (1 - abs(1 - (send_bytes / throughput_available[step])))
                if c_utility < 0:
                    reward_alpha = reward_factor_alpha * c_utility
                else: 
                    reward_alpha = reward_factor_alpha * (received_bytes / send_bytes) * c_utility
            
            reward_beta = (rtt**0.5) * reward_factor_beta

            return reward_alpha - reward_beta

        def perform_action(process, action):
            global cwnd_log  
            # Change CWND in file
            if action != False:
                cwnd = 0
                with open(cwnd_path, 'r') as file:
                    cwnd = float(file.read()) / packet_size # Divide with minimum size of packets ...
                    #print(cwnd, action)
                    
                    cwnd = actions[action](cwnd) * packet_size # Multiply with minimum size of packets ...
                    
                cwnd = int(cwnd // 1) # Only whole amounts of bytes
                
                if cwnd < packet_size: cwnd = packet_size
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
            new_state, (received_bytes, send_bytes, ack_perc, finished) = get_next_state(process) 

            
            
            # Calculate reward based on new state 
            #reward = get_reward(n_bytes, ack_perc, state[0][2])
            reward = get_reward(received_bytes, send_bytes, new_state[0][2])
            
            return new_state, reward, finished

        # ------------- Normalization struct -----------------------------

        class Normalization():
            def __init__(self, state_len, max_norm_initials=[0,0,0,0]) -> None:
                self.max = [0] * state_len
                if state_len == len(max_norm_initials): self.max = max_norm_initials
                else: print("Custom max normalization not in use, different state size expected")
                self.min = [0] * state_len
            
            def update(self, new_values):
                for ii, new_value in enumerate(new_values[0]):
                    if self.min[ii] > new_value:
                        self.min[ii] = new_value
                    elif self.max[ii] < new_value:
                        self.max[ii] = new_value
                    if ii < 2:
                        if self.min[(ii+1)%2] > new_value:
                            self.min[(ii+1)%2] = new_value
                        elif self.max[(ii+1)%2] < new_value:
                            self.max[(ii+1)%2] = new_value
                    else:
                        if self.min[((ii+1)%2)+2] > new_value:
                            self.min[((ii+1)%2)+2] = new_value
                        elif self.max[((ii+1)%2)+2] < new_value:
                            self.max[((ii+1)%2)+2] = new_value
                
            def normalize(self, values):
                return [[(value - self.min[ii]) / (self.max[ii] - self.min[ii]) for ii, value in enumerate(values[0])]]


        # ------------- Initialization -----------------------------


        agent = Agent(n_actions     = len(actions),     # Action space
                    batch_size    = batch_size,       # Training batch size
                    alpha         = learning_rate,    # Learning rate
                    n_epochs      = n_epochs,         # Learning epochs             
                    input_dims    = len(states),      # State space 
                    gamma         = discount_rate,    # Discount rate
                    gae_lambda    = 0.95,             # GAE advantage parameter
                    chkpt_dir     = model_folder_path + '/')

        if load_existing_model: agent.load_models()

        # Reward storage
        rewards_all_episodes = []

        # Input normalization struct list
        normalizer = Normalization(len(states), max_norm_initials)

        # ------------- Training loops ---------------------------
        print(f"\nStaring training for values alpha: {alpha_value} and beta: {beta_value}")
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
            state, (received_bytes, send_bytes, ack_perc, _)  = get_next_state(process)
            #get_reward(received_bytes, send_bytes, state[0][2]) # Set utulity value

            normalizer.update(state)    
            state = normalizer.normalize(state)

            if time_steps != 0 and time_steps != 1:
                padding = np.zeros((time_steps-1, len(states)))
                state = np.vstack((padding, state))
            
            for step in range(steps_max): # ----------------------------------------STEP-----------------------------------------
                #print("step",step)
                # Get action
                action, probabilities,  val = agent.choose_action(state)
                
                if random.uniform(0, 1) < exploration_rate: action = random.randrange(len(actions))

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
                
                normalizer.update(new_state)
                new_state = normalizer.normalize(new_state)
                # Updata state
                if time_steps == 0 or time_steps == 1:
                    state = new_state # Normalization here
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

            exploration_rate = exploration_rate_min + (exploration_rate_max - exploration_rate_min) * np.exp(-exploration_rate_decay * episode)
            
            # Close process
            process.terminate()
            
            # Save reward
            rewards_all_episodes.append(reward_episode/400)
            
            score_history.append(reward_episode/400)
            moving_avg_score =  np.mean(score_history[-30:])

            if moving_avg_score > best_score:
                best_score = moving_avg_score
                agent.save_models()

            
            # Print completed epochs
            print(f"episode: {episode}, avg score: {reward_episode / 400}")
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
        for ii, reward in enumerate([ii for ii in rewards_all_episodes]): print(ii,reward)

