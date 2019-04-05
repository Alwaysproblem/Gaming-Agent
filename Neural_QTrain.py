#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gym
import tensorflow as tf
import numpy as np
import random

from tensorflow.layers import Dense, Dropout

from queue import PriorityQueue as PQ

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA =  0.95 # discount factor
INITIAL_EPSILON =  0.9 # starting value of epsilon
FINAL_EPSILON =  0.1 # final value of epsilon
EPSILON_DECAY_STEPS = 100 # decay period

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph
# np.random.seed(100)
# tf.set_random_seed(100)
REWARD_DIM = 1
DONE_DIM = 1
learning_rate = 0.001
hidden_units = 20
rate_sam = 0.8
batch_size = 256
refresh_target = 25
capacity = 2 ** 14
dropout = 0.4

#%%
class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """

    def __init__(self, capacity):
        self.data_pointer = 0
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root

    def __len__(self):
        return len([i for i in self.data if not isinstance(i, int)])

class ReplayMemory():
    def __init__(self, capacity, epsilon = 0.01, alpha = 0.6, beta = 0.4, beta_increment = 0.001, abs_err_upper = 1.):
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = 0.4
        self.beta_increment = beta_increment
        self.abs_err_upper = abs_err_upper
        self.tree = SumTree(capacity)

    def _store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def Store(self, state, action, reward, next_state, done):
        transition = np.hstack((state, action, reward, next_state, done))
        self._store(transition)

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight

        if min_prob == 0:
            min_prob = self.epsilon

        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            # print(data)
            b_idx[i], b_memory[i] = idx, data
        
        return b_idx, b_memory, ISWeights

    def transform(self, memory):
        s = memory[:, : STATE_DIM]
        a = memory[:, STATE_DIM: STATE_DIM + ACTION_DIM]
        r = memory[:, STATE_DIM + ACTION_DIM : STATE_DIM + ACTION_DIM + REWARD_DIM]
        s_ = memory[:, STATE_DIM + ACTION_DIM + REWARD_DIM : STATE_DIM + ACTION_DIM + REWARD_DIM + STATE_DIM]
        done = memory[:, -1:]
        return s, a, r, s_, done

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self):
        return len(self.tree)


#%%
def NGraph(state_in, STATE_DIM = STATE_DIM, ACTION_DIM = ACTION_DIM, hidden_units = 100):
    """
    the input state is row vector.
    the output is also a vector.
    """

    pred = Dense(hidden_units, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.5, seed = 1), bias_initializer=tf.constant_initializer(0.1))(state_in)
    pred = Dropout(rate=dropout)(pred)
    output = Dense(ACTION_DIM, kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.5, seed = 1), bias_initializer=tf.constant_initializer(0.1))(pred)

    return output

# TODO: Network outputs
q_values = NGraph(state_in)
q_target = tf.identity(q_values)
double_q_target = tf.identity(q_values)
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

Weight = tf.placeholder(dtype = tf.float32, shape=(None, 1), name="ISWeight")

# TODO: Loss/Optimizer Definition
with tf.variable_scope("loss"):
    loss = tf.reduce_sum(Weight * tf.square(target_in - q_action))
with tf.variable_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


RM = ReplayMemory(capacity)

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

        RM.Store(state, action, reward, next_state, int(done))

        # store the state as tuple (s, a, r, s_, done)
        # ReplayMemory, full = Store_State(
        #                         ReplayMemory,
        #                         ReplayMemory_size,
        #                         state,
        #                         action,
        #                         reward,
        #                         next_state,
        #                         int(done)
        #                     )
        if len(RM) > round(1/rate_sam):

            b_idx, b_memory, ISWeights = RM.sample(n = round(len(RM) * rate_sam) if len(RM) * rate_sam < batch_size else batch_size )

            s_batch, a_batch, r_batch, ns_batch, done_batch = RM.transform(b_memory)

            nextstate_q_values = q_target.eval(feed_dict={
                state_in: ns_batch
            })

            q_target_nn = double_q_target.eval(feed_dict={
                state_in: ns_batch
            })

            # TODO: tansform action_t into one-hot coding.
            action_index = np.argmax(nextstate_q_values, axis=1)
            
            action_n = np.zeros_like(a_batch)
            action_n[[j for j in range(nextstate_q_values.shape[0])], action_index] = 1

            target_batch = r_batch + GAMMA * (1 - done_batch) * np.max(q_target_nn * action_n, axis=1, keepdims=1) # need axis = 1

            RM.batch_update(b_idx, np.absolute(target_batch))

            target = target_batch.squeeze() if target_batch.shape != (1, 1) else [target_batch.squeeze()]

            # Do one training step
            loss_ , _ = session.run([loss, optimizer], feed_dict={
                target_in: target,
                action_in: a_batch,
                state_in: s_batch,
                Weight: ISWeights
            })

            if step % refresh_target == 0:
                q_target = tf.identity(q_values)

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
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)
    
    print("processing:{0}%".format(round((episode % TEST_FREQUENCY + 1) * 100 / TEST_FREQUENCY)), end="\r")

env.close()
