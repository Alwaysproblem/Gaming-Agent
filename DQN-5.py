from collections import deque
from random import randint
import random
import tensorflow as tf
import numpy as np
import gym

class network:
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def inference_graph(self):
        self.inputs = tf.placeholder(tf.float32, shape=[None, 4])
        self.W_dense1 = self.weight_variable([4 , 32])
        self.W_dense2 = self.weight_variable([32 , 32])
        self.W_outputs = self.weight_variable([32 , 2])
        self.b_dense1 = self.bias_variable([32])
        self.b_dense2 = self.bias_variable([32])
        self.b_outputs = self.bias_variable([2])

        self.dense1 = tf.nn.relu(tf.matmul(self.inputs,self.W_dense1) + self.b_dense1)
        self.dense2 = tf.nn.relu(tf.matmul(self.dense1,self.W_dense2) + self.b_dense2)
        self.outputs = tf.nn.relu(tf.matmul(self.dense2,self.W_outputs) + self.b_outputs)

        self.max_q = tf.reduce_max(self.outputs, axis = 1)

    def learning_graph(self):
        self.rewards = tf.placeholder(tf.float32, shape=[None, 1])
        self.discounted_q = tf.placeholder(tf.float32, shape=[None, 1])
        self.target_q = tf.add(self.rewards, self.discounted_q)
        self.actions = tf.placeholder(tf.int32, shape=[None, 1])
        self.pred = tf.reduce_sum(tf.multiply(self.outputs, tf.reshape(tf.one_hot(self.actions, 2, dtype=tf.float32), [-1, 2])), 1, keepdims=True)
        self.loss = tf.losses.mean_squared_error(self.target_q, self.pred)
        self.train = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

    def sync_graphs(self, target_network, policy_network):
        self.a1 = tf.assign(target_network.W_dense1, policy_network.W_dense1)
        self.a2 = tf.assign(target_network.W_dense2, policy_network.W_dense2)
        self.a3 = tf.assign(target_network.W_outputs, policy_network.W_outputs)
        self.a4 = tf.assign(target_network.b_dense1, policy_network.b_dense1)
        self.a5 = tf.assign(target_network.b_dense2, policy_network.b_dense2)
        self.a6 = tf.assign(target_network.b_outputs, policy_network.b_outputs)

policy_network = network()
policy_network.inference_graph()
policy_network.learning_graph()

target_network = network()
target_network.inference_graph()

sync_network = network()
sync_network.sync_graphs(target_network, policy_network)

initialize = tf.global_variables_initializer()

sess = tf.Session()
sess.run(initialize)

buffer_size = 2048
sample_size = 32
M = 8
epsilon = 1.0

env = gym.make('CartPole-v0')
replay_buffer = deque(maxlen=buffer_size)
current_observation = env.reset()

def get_policy_action(current_observation, epsilon):
    q_values = sess.run(policy_network.outputs, feed_dict={policy_network.inputs: [current_observation]})[0]
    if np.random.uniform() < epsilon:
        action = randint(0, 1)
    else:
        action = np.argmax(q_values)
    return action

def train(batch):
    s = []
    a = []
    s_n = []
    r = []
    for i in batch:
        s.append(i[0])
        a.append([i[1]])
        s_n.append(i[2])
        r.append([i[3]])
    max_q = sess.run(target_network.max_q, feed_dict={target_network.inputs: s_n})
    discounted_q = np.split(0.95*max_q, max_q.shape[0])
    _, loss = sess.run([policy_network.train, policy_network.loss], feed_dict={policy_network.inputs: s,
                                                                               policy_network.actions: a,
                                                                               policy_network.rewards: r,
                                                                               policy_network.discounted_q: discounted_q})
    return loss

def sync_networks():
    sess.run(sync_network.a1)
    sess.run(sync_network.a2)
    sess.run(sync_network.a3)
    sess.run(sync_network.a4)
    sess.run(sync_network.a5)
    sess.run(sync_network.a6)

num_samples = 0
for episode in range(100000):
    episode_reward = 0
    epsilon = max(0.01,epsilon * 0.995)
    while True:
        action = get_policy_action(current_observation, epsilon)
        next_observation, reward, done, info = env.step(action)
        if done == True and episode_reward != 199:
            reward = -1
        replay_buffer.append([current_observation, action, next_observation, reward])
        num_samples += 1
        episode_reward += reward
        if num_samples > sample_size:
            batch = random.sample(replay_buffer, sample_size)
            loss = train(batch)
        if num_samples % (M) == 0:
            sync_networks()
        if done:
            current_observation = env.reset()
            if episode % 10 == 0 and num_samples > sample_size:
                print ("Episode",episode,"complete. Reward =", episode_reward, "steps =",num_samples,"loss =",loss, "epsilon =",epsilon)
            break
        else:
            current_observation = next_observation