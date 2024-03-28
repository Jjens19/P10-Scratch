import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses



# ------------- PPO Actor ---------------------------------
class actor(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.fc1 = layers.Dense(64,activation='relu')
    self.fc2 = layers.Dense(128,activation='relu')
    self.fc3 = layers.Dense(64,activation='relu')
    self.a = layers.Dense(5,activation='softmax')

  def call(self, input_data):
    x = self.fc1(input_data)
    x = self.fc2(input_data)
    x = self.fc3(input_data)
    a = self.a(x)
    return a

# ------------- PPO Critic ---------------------------------

class critic(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.fc1 = layers.Dense(64,activation='relu')
    self.fc2 = layers.Dense(128,activation='relu')
    self.fc3 = layers.Dense(64,activation='relu')
    self.v = layers.Dense(1,activation=None)

  def call(self, input_data):
    x = self.fc1(input_data)
    x = self.fc2(input_data)
    x = self.fc3(input_data)
    v = self.v(x)
    return v
    



# ------------- PPO Agent ---------------------------------

class agent():
    def __init__(self):
        self.a_opt = optimizers.Adam(learning_rate=7e-3)
        self.c_opt = optimizers.Adam(learning_rate=7e-3)
        self.actor = actor()
        self.critic = critic()
        self.clip_pram = 0.2

          
    def act(self,state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        sample = tf.random.categorical(tf.math.log(prob[0]), 1)
        action = sample.numpy()[0][0]
        return action, prob[0][0]
    

    # Learn
    def actor_loss(self, probs, actions, adv, old_probs, closs):    
        probability = probs

        # Remove 0% probabilities as log can't handle them
        probability = tf.where(tf.equal(probability, 0), tf.fill(tf.shape(probability), 1e-10), probability)

        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,tf.math.log(probability))))
        #print("Ent", probability,entropy)
        
        sur1 = []
        sur2 = []
        
        for pb, t, op, a in zip(probability, adv, old_probs, actions):
                        t =  tf.constant(t)
                        #op =  tf.constant(op)
                        #print(f"t{t}")
                        #ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
                        ratio = tf.math.divide(pb[0][a],op[a])
                        #print(f"ratio{ratio}")
                        s1 = tf.math.multiply(ratio,t)
                        #print(f"s1{s1}")
                        s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram),t)
                        #print(f"s2{s2}")
                        sur1.append(s1)
                        sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        
        #closs = tf.reduce_mean(tf.math.square(td))
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        return loss


    def learn(self, states, actions,  adv , old_probs, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),len(discnt_rewards[0][0])))
        adv = tf.reshape(adv, (len(adv),len(adv[0][0])))

        old_p = old_probs

        #print(old_p.shape)
        #old_p = tf.reshape(old_p, (len(old_p),2))
        #print(old_p.shape)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v =  self.critic(states,training=True)
            v = tf.reshape(v, (len(v),))
            #td = tf.math.subtract(discnt_rewards, v)


            #print("Calc Loss")
            c_loss = 0.5 * losses.mean_squared_error(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)
            #print(a_loss)

            #print("Calc gradients")
            grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
            grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
            
            #print("Actor gradients")
            #for ii in zip(grads1, self.actor.trainable_variables): print(ii)

            #print("Critic gradients")
            #for ii in zip(grads2, self.actor.trainable_variables): print(ii)

            #print("Apply gradients")
            self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
            self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
            #print("Done")
        return a_loss, c_loss


# ------------- PPO Advantage ---------------------------------

def calc_advantage(rewards, dones, values, gamma):
    g = 0
    lmbda = 0.95
    returns = []
    for i in reversed(range(len(rewards))):
       delta = rewards[i] + gamma * values[i + 1] * dones[i] - values[i]
       g = delta + gamma * lmbda * dones[i] * g
       returns.append(g + values[i])

    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - values[:-1]
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    returns = np.array(returns, dtype=np.float32)
    return returns, adv 