import tensorflow as tf
import numpy as np
# Define your probabilities array
probs = [0.1, 0.3, 0.6]  # Example probabilities


sums = [0,0,0]

for ii in range(100000):
    # Create a categorical distribution manually
    sampled_value = tf.random.categorical(tf.math.log([probs]), 1)

    # Print the sampled value
    sums[sampled_value.numpy()[0][0]] += 1

print(sums[0], sums[1], sums[2])

print(np.reshape(probs, (len(probs),2)))
print(probs)