#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gym
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras import layers, Input

tf.config.experimental_run_functions_eagerly(True)
tf.keras.backend.set_floatx('float64')
#%%
# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA =  0.9 # discount factor
INITIAL_EPSILON =  0.9 # starting value of epsilon
FINAL_EPSILON =  0.1 # final value of epsilon
EPSILON_DECAY_STEPS = 100 # decay period

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# TODO: Define Network Graph
REWARD_DIM = 1
DONE_DIM = 1
learning_rate = 0.001
hidden_units = 20
rate_sam = 0.06
refresh_target = 25
ReplayMemory_size = 10000
ReplayMemory = np.zeros((1, STATE_DIM + ACTION_DIM + REWARD_DIM + STATE_DIM + DONE_DIM)) # just for experience replay.

#%%
def Store_State(ReplayMemory, ReplayMemory_size, s, a, r, s_, done):
    elements = np.expand_dims(np.hstack((s, a, r, s_, done)), axis = 0)
    ReplayMemory = np.concatenate((ReplayMemory, elements), 0)
    if not any(ReplayMemory[0, :]) or len(ReplayMemory) > ReplayMemory_size:
        ReplayMemory = np.delete(ReplayMemory, 0, axis=0)
    full = any(ReplayMemory[0, :])
    return ReplayMemory, full

def Sample_State(ReplayMemory, sample_percent, replace = False):
    sample_num = round(len(ReplayMemory) * sample_percent)
    ind = np.random.choice(range(len(ReplayMemory)), sample_num, replace = replace)
    Sample_batch = ReplayMemory[ind, :]
    s = Sample_batch[:, : STATE_DIM]
    a = Sample_batch[:, STATE_DIM: STATE_DIM + ACTION_DIM]
    r = Sample_batch[:, STATE_DIM + ACTION_DIM : STATE_DIM + ACTION_DIM + REWARD_DIM]
    s_ = Sample_batch[:, STATE_DIM + ACTION_DIM + REWARD_DIM : STATE_DIM + ACTION_DIM + REWARD_DIM + STATE_DIM]
    done = Sample_batch[:, -1:]

    return s, a, r, s_, done

#%%

def NGraph(STATE_DIM = STATE_DIM, ACTION_DIM = ACTION_DIM, hidden_units = 14):
    inputs = Input(name="state", shape=(STATE_DIM,))
    linear = layers.Dense(hidden_units, activation="relu")(inputs)
    outputs = layers.Dense(ACTION_DIM)(linear)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

#%%
# TODO: Network outputs
q_values = NGraph()
q_target = tf.keras.models.clone_model(q_values)
double_q = tf.keras.models.clone_model(q_values)
#%%
# @tf.function
def Loss(pred, label, action_in):
    q_action = tf.math.reduce_sum(pred*action_in, axis = 1)
    loss = tf.reduce_mean(tf.square(label - q_action))

    return loss
#%%
opt = tf.keras.optimizers.Adam(learning_rate)
#%%
def train(opt, Loss_fun, model, inputs, label, action_in):
    def loss_graph():
        # if tf.config.list_physical_devices
        with tf.GradientTape():
            closs = Loss_fun(model(inputs, training=True), label, action_in)
            tf.print(closs)
        return closs
    opt.minimize(loss_graph, model.trainable_variables)
    # return Loss_fun(model(inputs), label, action_in)
    return 

# 1. change the struchture of NN  Q(s, a) when prediction set a = ones(1, action_dim)
# 2. NN: Q(s) = a, change loss

# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.predict(np.array([state]), batch_size = 1)
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))

        # store the state as tuple (s, a, r, s_, done)
        ReplayMemory, full = Store_State(
                                ReplayMemory,
                                ReplayMemory_size,
                                state,
                                action,
                                reward,
                                next_state,
                                int(done)
                            )

        if len(ReplayMemory) > round(1/rate_sam):#and episode < 120:
            s_batch, a_batch, r_batch, ns_batch, done_batch = Sample_State(ReplayMemory, rate_sam)


            # Q1 -> r + Q2()
            nextstate_q_values = q_target.predict(ns_batch)

            Q_values = double_q.predict(ns_batch)

            action_index = np.argmax(nextstate_q_values, axis = -1)

            action_n = np.zeros_like(a_batch)
            action_n[[j for j in range(nextstate_q_values.shape[0])], action_index] = 1

            Q_double = action_n * Q_values

            target_batch = r_batch + GAMMA * (1 - done_batch) * np.max(Q_double, axis=1, keepdims=1) # need axis = 1

            target = target_batch.squeeze() if target_batch.shape != (1, 1) else [target_batch.squeeze()]

            # Do one training step
            train(opt, Loss, q_values, s_batch, target, a_batch)

            if step % refresh_target == 0:
                q_target.set_weights(q_values.get_weights())
                # print(f"episode: {episode} - step: {step}, refresh Q_target.")
            if step % refresh_target == 7:
                double_q.set_weights(q_values.get_weights())
                # print(f"episode: {episode} - step: {step}, refresh Double_Q_target.")

        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.predict(np.array([state])))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        with open("out.txt", 'w') as fp:
            print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward, file=fp)

env.close()


# %%
