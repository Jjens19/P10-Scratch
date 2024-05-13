import numpy as np
import os
import time
import subprocess

import random

import matplotlib.pyplot as plt

model_id = 2        # Select network 
training_n = 1      # Select training scenaria
test_id = 4      # Select test case

# RNN Memory duration (0 or 1 for non-RNN network)
time_steps = 15

# ------------ Network selection -----------------------------------------------------------------------
match model_id:
    case 0:
        from ppo_networks.debug import Agent
    case 1:
        from ppo_networks.ppo_agent_3_dense import Agent
    case 2:
        from ppo_networks.ppo_agent_LSTM_simple import Agent



# ------------- CONFIG ------------------------------------
alpha_values = [2.5, 5]
beta_values = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

overall_scores = []
for alpha_value in alpha_values:
    for beta_value in beta_values:

        

        

        # PATHS
        # Path to your ns-3 script file
        ns3_script = f"scratch/tests/{test_id}/"

        # Path for CWND size file
        cwnd_path = "scratch/rate.txt"

        packet_size = 512
        cwnd_start = random.randint(packet_size, packet_size * 20) 

        # Model folder path
        model_names = ["debug", "3dense", "simple_lstm"]
        model_folder_path = f"scratch/models/alpha_beta/alpha{alpha_value}_beta{beta_value}"
        os.makedirs(model_folder_path, exist_ok=True)

        # Training duration
        sample_frequency = 1 # Amount of steps/samples per second
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
        states = (0,1,2,3) # (state_size_send, state_size_acks, state_size_rtt, rtt_dev )

        # Custom Max normalization initalial values
        max_norm_initials = [2000, 2000, 200, 200]

        # Reward factors 
        reward_factor_alpha = 5 # 
        reward_factor_beta   = 0.25 # 

        reward_factor_ack_ratio = 1 #
        utility = 0 
        utility_threshold = 0.9 # How big should a change be before a reward/penalty is given
        utility_reward = 5    # Size of reward/penalty




        test_part = 0
        cwnd_list = [[],[]]
        received_bytes_list = [[],[]]


        last_rtt = None

        sleep_time = 0.01 # Delay between checking subprocess buffer
        cwnd_log = ""
        reward_log = ""
        state_log = ""

        best_score = -np.infty
        score_history = []
        figure_file = "scratch/rewards.png"
        # ------------- Functions ----------------------------------

        def plot_learning_curve(x, scores, figure_file):
            running_avg = np.zeros(len(scores))
            for i in range(len(running_avg)):
                running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
            plt.plot(x, running_avg)
            plt.title('Running average of previous 100 scores')
            plt.savefig(figure_file)


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

            global received_bytes_list
            received_bytes_list[test_part].append(received_bytes)

            return np.reshape(state_array, (1,4)) , (received_bytes, send_bytes, ack_perc, finished)
            return (state_send, state_acks, state_rtt), (n_bytes, ack_perc, finished)


        def perform_action(process, action):
            global cwnd_log
            global cwnd_list

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
                    cwnd_list[test_part].append(cwnd)
            else:
                with open(cwnd_path, 'r') as file:
                    cwnd = int(file.read())
                    cwnd_log += f"{cwnd}\n"
                    cwnd_list[test_part].append(cwnd)
            
            # Inform subprocess new episode is ready 
            process.stdin.write(f"next state\n")
            process.stdin.flush()
            
            # get new state 
            new_state, (received_bytes, send_bytes, ack_perc, finished) = get_next_state(process) 

            
            
            # Calculate reward based on new state 
            #reward = get_reward(n_bytes, ack_perc, state[0][2])
            reward = 1
            
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

        agent.load_models()

        # Reward storage

        # Input normalization struct list
        normalizer = Normalization(len(states), max_norm_initials)

        # ------------- Training loops ---------------------------

        print(f"\nStaring tests for values alpha: {alpha_value} and beta: {beta_value}")
        print("Starting custom CCA test")
        reward_episode = []

        # Reset congestion window
        with open(cwnd_path, 'w') as file:
            
            file.write(f"{cwnd_start}")

        # Start NS3 Simulation as subprocess
        process = subprocess.Popen(["./ns3", "run", ns3_script + "custom/custom.cc"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
            if step % 50 == 0: print("step",step)
            # Get action
            action, probabilities,  val = agent.choose_action(state)
            
            if np.any(np.isnan(probabilities)): print("NAN FOUND")


            # Perform action and get rewards/info
            new_state, reward, done = perform_action(process, action)    
            
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


        # Close process
        process.terminate()


        # Save reward   

        # Print completed epochs
        #print(f"Episode {episode} complete, reward of: {reward_episode}")

        with open("scratch/cwnd_log.txt", 'w') as file:
            
            file.write(f"{cwnd_log}")

        cwnd_log = ""
        with open("scratch/state_log.txt", 'w') as file:
            file.write(f"{state_log}")
        state_log = ""

        print("Custom CCA test completed.")
        #------------------------------------------Cubic Test---------------------------------------------------------------
        print("Starting cubic comparison test.")

        test_part = 1



        # Start NS3 Simulation as subprocess
        process2 = subprocess.Popen(["./ns3", "run", ns3_script + "cubic/cubic.cc"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while "Simulation" not in process2.stdout.readline(): 
            time.sleep(sleep_time)

        process2.stdin.write(f"next state\n")
        process2.stdin.flush()
        state, (received_bytes, send_bytes, ack_perc, _)  = get_next_state(process2)

        for ii in range(400):
            if ii%50 == 0: print(f"Step: {ii}")
            perform_action(process2, 0) 
            


        print("Tests completed.")
        #------------------------------------------GRAPHS-------------------------------------------------------------------
        print("Creating graphs...")

        def plot_data_over_time(data_lists_list, titles, xlabels, ylabels, legend_labels_list, filenames, filepath):
            for data_lists, title, xlabel, ylabel, legend_labels, filename in zip(data_lists_list, titles, xlabels, ylabels, legend_labels_list, filenames):
                for data, label in zip(data_lists, legend_labels):
                    plt.plot(list(range(len(data))), data, label=label)
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.legend()
                plt.savefig(filepath + filename, dpi=500)
                plt.close()

        custom_bytes_list_sum = [sum(received_bytes_list[0][:ii+1]) for ii, _ in enumerate(received_bytes_list[0])]
        cubic_bytes_list_sum = [sum(received_bytes_list[1][:ii+1]) for ii, _ in enumerate(received_bytes_list[1])]

        print(f"Custom sum score: {custom_bytes_list_sum[-1]:.2e}")
        print(f"Cubic sum score:  {cubic_bytes_list_sum[-1]:.2e}")

        overall_scores.append(custom_bytes_list_sum[-1])

        plot_data_over_time(data_lists_list=   [[cwnd_list[0]],[received_bytes_list[0]],[custom_bytes_list_sum],
                                                [cwnd_list[1]],[received_bytes_list[1]],[cubic_bytes_list_sum],
                                                cwnd_list, received_bytes_list, [custom_bytes_list_sum, cubic_bytes_list_sum]], 
                            titles=            ['cWnd over time for custom CCA',               'Bytes over time for custom CCA',               'Cummulative bytes over time for custom CCA', 
                                                'cWnd over time for TCP Cubic',                'Bytes over time for TCP Cubic',                'Cummulative bytes over time for TCP Cubic',
                                                'cWnd over time for Custom CCA and TCP Cubic', 'Bytes over time for Custom CCA and TCP Cubic', 'Cummulative bytes over time for Custom CCA and TCP Cubic'],
                            xlabels=           ['Time [s]'] * 9, 
                            ylabels=           ['cWnd Value [B]', 'Bytes [B]', 'Bytes [B]'] * 3, 
                            legend_labels_list=[['Custum CCA']] * 3 + [['TCP Cubic']] * 3 + [['Custum CCA', 'TCP Cubic']] * 3, 
                            filenames=         ['custom_cWnd.png',  'custom_bytes.png',  'custom_bytes_summed.png',
                                                'cubic_cWnd.png',   'cubic_bytes.png',   'cubic_bytes_summed.png',
                                                'combined_cWnd.png','combined_bytes.png','combined_bytes_summed.png'],
                            filepath=          f"{model_folder_path}/")


        print("Graphs saved.")

print("\nALL TEST COMPLETED \nResults:")
for ii in range(len(alpha_values)):
    for jj in range(len(beta_values)):
        print(f"Alpha {alpha_values[ii]}, Beta {beta_values[jj]}: {overall_scores[ii * len(beta_values) + jj]}")
print(overall_scores)

best_model = np.argmax(overall_scores)

print(f"Best model: Alpha {alpha_values[best_model//len(beta_values)]}, Beta {beta_values[best_model%(len(beta_values))]}")



plt.plot(beta_values, overall_scores[:len(overall_scores)//2], label=f"Alpha: {alpha_values[0]}")
plt.plot(beta_values, overall_scores[len(overall_scores)//2:], label=f"Alpha: {alpha_values[1]}")

plt.title("Alpha/Beta test scores")
plt.xlabel("Beta value")
plt.ylabel("Score")
plt.legend()
plt.savefig(f"scratch/models/alpha_beta/test_scores.png", dpi=500)
plt.close()