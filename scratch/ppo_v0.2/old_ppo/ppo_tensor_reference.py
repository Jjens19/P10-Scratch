import tensorflow as tf
import gym
from tensorflow.keras import layers, models, optimizers

import numpy as np

num_episodes = 100


def create_policy_network(input_shape, output_shape):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_shape, activation='softmax')
    ])
    return model


class PPOAgent:
    def __init__(self, input_shape, output_shape, learning_rate=0.001, gamma=0.99):
        self.policy_network = create_policy_network(input_shape, output_shape)
        self.optimizer = optimizers.Adam(learning_rate)
        self.gamma = gamma

    def get_action(self, state):
        # Get action probabilities from the policy network
        action_probs = self.policy_network.predict(state)[0]
        action = np.random.choice(len(action_probs), p=action_probs)
        return action, action_probs[action]

    def train(self, states, actions, advantages, old_probs):
        with tf.GradientTape() as tape:
            # Calculate new action probabilities
            new_probs = self.policy_network(states)
            action_masks = tf.one_hot(actions, depth=new_probs.shape[1])
            chosen_probs = tf.reduce_sum(action_masks * new_probs, axis=1)
            ratios = chosen_probs / old_probs

            # Calculate PPO loss
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=0.8, clip_value_max=1.2)
            surrogate1 = ratios * advantages
            surrogate2 = clipped_ratios * advantages
            loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

        # Perform gradient descent
        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))

def calculate_advantages(rewards, gamma=0.99):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        advantages[t] = running_add
    # Standardize advantages
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    return advantages.tolist()



env = gym.make('CartPole-v1')
input_shape = env.observation_space.shape
output_shape = env.action_space.n
agent = PPOAgent(input_shape, output_shape)

for episode in range(num_episodes):
    state = env.reset()
    states, actions, rewards, old_probs = [], [], [], []

    while True:
        # Collect experiences
        states.append(state)
        action, prob = agent.get_action(state)
        actions.append(action)
        old_probs.append(prob)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        state = next_state

        if done:
            # Calculate advantages
            advantages = calculate_advantages(rewards)

            # Convert lists to arrays
            states = np.array(states)
            actions = np.array(actions)
            advantages = np.array(advantages)
            old_probs = np.array(old_probs)

            # Train the agent
            agent.train(states, actions, advantages, old_probs)
            break
