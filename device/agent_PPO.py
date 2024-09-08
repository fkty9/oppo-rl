import warnings
import os
from threading import Thread
import time
import numpy as np
import random
import socket
import struct
import math
import sys
sys.path.append('D:/fkty/Desktop/oppoooo/new/oppo-RL-master/oppo-RL-master/')
import utils.tools as tools
from datetime import datetime

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.compat.v1.enable_eager_execution()
    tf.enable_eager_execution()
    # from keras import backend as K
    from tensorflow.compat.v1.keras import backend as K
    from collections import defaultdict
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.layers import Dense
    from collections import deque
    # import tensorflow as tf
    import matplotlib.pyplot as plt

import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import socket

import tensorflow.compat.v1 as tf
tf.compat.v1.enable_eager_execution()
tf.disable_v2_behavior()
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import Dense, Input
from tensorflow.compat.v1.keras.optimizers import Adam, RMSprop
import keras_tuner as kt
from keras_tuner import RandomSearch

def send_socket_data(message, host='192.168.0.7', port=8888):
    try:
        # 创建一个 socket 对象
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 连接到服务器
        client_socket.connect((host, port))
        # 发送数据
        client_socket.sendall(message.encode('utf-8'))
        # 接收来自服务器的响应
        response = client_socket.recv(1024)
        return response.decode('utf-8')
        
    except socket.error as e:
        print(f"Socket error: {e}")
    finally:
        # 关闭连接
        client_socket.close()


experiment_time = 1000
clock_change_time = 30
cpu_power_limit = 1000
gpu_power_limit = 1600
action_space = 9
target_fps = 60
target_temp = 65
beta = 2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# 超参数搜索空间，包括层数、每层单元数和学习率
class HyperModel(kt.HyperModel):
    def __init__(self, state_size, action_size):
        super(HyperModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

    def build(self, hp):
        model = Sequential()
        for i in range(hp.Int('num_layers', min_value=3, max_value=20, step=1)):
            model.add(Dense(units=hp.Int(f'units_{i}', min_value=8, max_value=128, step=8),
                            activation='relu', kernel_initializer='normal'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='normal'))
        model.compile(
            optimizer=Adam(hp.Float('lr', min_value=1e-5, max_value=1e-2, sampling='log')),
            loss='mse'
        )
        return model

class PPOAgent:
    def __init__(self, state_size, action_size, using_nas):
        self.state_size = state_size
        self.action_size = action_size
        self.using_nas = using_nas
        self.actions = list(range(9))
        self.clk_action_list = []
        for i in range(3):
            for j in range(3):
                clk_action = (3 * i + 2, j + 1)
                self.clk_action_list.append(clk_action)

        # Hyperparameters
        self.learning_rate = 0.0003
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.batch_size = 64
        self.epochs = 10
        self.memory = deque(maxlen=2000)

        # Build models
        self.policy_model = self.build_nas_model() if using_nas else self.build_model()
        self.value_model = self.build_value_model()
        self.optimizer = Adam(lr=self.learning_rate)
        # 构建优化器以适应模型的可训练变量
        # self.optimizer.build(self.policy_model.trainable_variables + self.value_model.trainable_variables)

        # 记录指标
        self.losses = []
        self.value_losses = []
    def build_nas_model(self):
        print("victory")
        hypermodel = HyperModel(self.state_size, self.action_size)
        tuner = RandomSearch(
            hypermodel,
            objective='val_loss',
            max_trials=10,
            executions_per_trial=1,
            directory='my_dir',
            project_name='dqn_tuner'
        )
        tuner.search_space_summary()
        tuner.search(np.random.random((1, self.state_size)), np.array([0]),
                     validation_data=(np.random.random((1, self.state_size)), np.array([0])))
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = hypermodel.build(best_hyperparameters)

        return model

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_normal'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
        return model

    def build_value_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(1, activation='linear', kernel_initializer='he_normal'))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        return model
    def get_action(self, state):
        state = np.array([state], dtype=np.float32)
        probs = self.policy_model.predict(state)[0]
        # 归一化概率分布，确保和为 1
        probs = np.clip(probs, 1e-10, 1.0)  # 防止概率为 0
        probs = probs / np.sum(probs)
        action = np.random.choice(self.action_size, p=probs)
        return action, probs

    def compute_advantage(self, rewards, values, next_values, dones):
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = np.zeros_like(deltas)
        advantage = 0
        for t in reversed(range(len(deltas))):
            advantage = deltas[t] + self.gamma * self.lam * advantage
            advantages[t] = advantage
        return advantages

    def train_model(self):
        states, actions, rewards, next_states, dones = zip(*self.memory)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        values = self.value_model.predict(states).flatten()
        next_values = self.value_model.predict(next_states).flatten()
        advantages = self.compute_advantage(rewards, values, next_values, dones)

        epoch_losses = []
        value_losses = []

        for epoch in range(self.epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_rewards = rewards[batch_indices]
                batch_dones = dones[batch_indices]

                # Update policy
                with tf.GradientTape() as tape:
                    probs = self.policy_model(batch_states)
                    action_probs = tf.reduce_sum(probs * tf.one_hot(batch_actions, self.action_size), axis=1)
                    old_probs = tf.reduce_sum(probs * tf.one_hot(batch_actions, self.action_size), axis=1)
                    ratio = action_probs / (old_probs + 1e-10)
                    clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    policy_loss = -tf.reduce_mean(
                        tf.minimum(ratio * batch_advantages, clipped_ratio * batch_advantages))

                grads = tape.gradient(policy_loss, self.policy_model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables))

                # Update value function
                with tf.GradientTape() as tape:
                    values = self.value_model(batch_states)
                    value_loss = tf.reduce_mean(tf.square(batch_rewards - values))

                grads = tape.gradient(value_loss, self.value_model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.value_model.trainable_variables))

                policy_loss_value = sess.run(policy_loss)
                epoch_losses.append(policy_loss_value)
                value_loss_value = sess.run(value_loss)
                value_losses.append(value_loss_value)
                # epoch_losses.append(policy_loss.numpy())
                # value_losses.append(value_loss.numpy())
        self.losses.append(np.mean(epoch_losses))
        self.value_losses.append(np.mean(value_losses))
        if self.using_nas:
            self.policy_model.save_weights('./save_model/nas_model.h5')

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


def normalization(big_cpu_freq, little_cpu_freq,big_util, little_util, mem, fps):
    big_cpu_freq = int(big_cpu_freq) / int(cpu_freq_list[1][-1])
    little_cpu_freq = int(little_cpu_freq) / int(cpu_freq_list[0][-1])
    mem = int(mem) / 1e6
    fps = int(fps) / 60
    return big_cpu_freq, little_cpu_freq, float(big_util), float(little_util),int(mem), fps

def get_reward(fps, target_fps, big_clock, little_clock,):
    reward = (fps - target_fps) * 20 +  ( big_clock + little_clock  ) *  (-100) 
    return reward

def process_action(action):
    action1, action2 = action % 3 , action // 3
    return [0 , action1, action2, 0]
    
if __name__=="__main__":

    # s_dim: 状态维度，即输入的状态向量的维度。
    # h_dim: 隐藏层的维度，即神经网络中间层的神经元数量。
    # branches : 每个分支的action的数量

    agent = PPOAgent(6, 9,True)

    cpu_freq_list, gpu_freq_list = tools.get_freq_list('pixel3')
    little_cpu_clock_list = tools.uniformly_select_elements(5, cpu_freq_list[0])
    big_cpu_clock_list = tools.uniformly_select_elements(5, cpu_freq_list[1])
    super_big_cpu_clock_list = tools.uniformly_select_elements(5, cpu_freq_list[2])   # 若是没有超大核，则全部为0

    state=(0,0,0,0,0,0)
    action=0
    loss = 0
    experiment_time=1000
    target_fps=25
    reward = 0
    closs,aloss=0,0

    f = open("output.csv", "w")
    f.write(f'episode,big_cpu_freq,little_cpu_freq,big_util,little_util,ipc,cache_miss,fps,action,aloss,closs,reward\n')

    t=1
    try:
        while t < experiment_time:
            t1 = datetime.now()
            temp = send_socket_data('0').split(',')
            big_cpu_freq = temp[0]
            little_cpu_freq =temp[1]
            fps = temp[2]
            mem = temp[3]
            little_util = temp[4]
            big_util = temp[5]
            ipc = temp[6]
            cache_miss = temp[7]
            print(temp)

            f.write(f'{t},{big_cpu_freq},{little_cpu_freq},{big_util},{little_util},{ipc},{cache_miss},{fps},{action},{aloss},{closs},{reward}\n')
            f.flush() 
            # print('[{}] state:{} action:{} fps:{}'.format(t, state,action,fps))
            # print(losses)

            big_cpu_freq, little_cpu_freq, big_util, little_util, mem, fps = normalization(big_cpu_freq, little_cpu_freq, big_util, little_util, mem, fps)

            # 解析数据
            # next_state=(underlying_data[0], underlying_data[1], underlying_data[2], underlying_data[3], underlying_data[4] ,fps)
            next_state = (big_cpu_freq, little_cpu_freq, big_util, little_util, mem, fps)
            
            # reward 
            reward = get_reward(fps, target_fps,big_cpu_freq, little_cpu_freq)
            # print(state, action, next_state, reward)

            # 获得action
            action = agent.get_action(state)
            action = action[0] 
            processed_action = process_action(action)
           
            done =1
            # replay memory
            agent.append_sample(state, action, reward, next_state, done)

            print(f'1,{big_cpu_clock_list[processed_action[1]]},{little_cpu_clock_list[processed_action[2]]}')
            res = send_socket_data(f'1,{big_cpu_clock_list[processed_action[1]]},{little_cpu_clock_list[processed_action[2]]}')
            # closs, aloss = agent.train(state, action, reward, next_state, done)
            aloss=0
            if(int(res) == -1):
                print('freq set error')
                break
            
            # update some state
            state = next_state
            t += 1

            print(len(agent.memory))
            if len(agent.memory) >= agent.batch_size:
                agent.train_model()

            if t % 60 == 0:
                agent.actor_learning_rate = 0.1
                agent.critic_learning_rate = 0.1
                print('[Reset learning_rate]')

            # 下面这段是记录进文件，供绘图使用的 
            # f.write(f'episode,big_cpu_freq,little_cpu_freq,gpu_freq,cpu_temp,fps\n')


    finally:
        f.close()
 