import warnings
import os
from threading import Thread
import time
import numpy as np
import random
import socket
import struct
import math
from copy import deepcopy
from torch import nn, optim
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import sys
sys.path.append('/home/wakaba/Desktop/zTT/')
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
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import train_utils as train_utils
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import Dense, Input
from tensorflow.compat.v1.keras.optimizers import Adam, RMSprop


def send_socket_data(message, host='192.168.2.108', port=8888):
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


experiment_time = 10000
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

class Config:
    def __init__(self):
        self.batch_size = 20
        self.memory_capacity = 10000
        self.lr_a = 2e-3
        self.lr_c = 5e-3
        self.gamma = 0.9
        self.tau = 0.005
        self.policy_freq = 2
        self.noise_clip = 0.5
        self.policy_noise = 0.2
        self.seed = random.randint(0, 100)
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256
        self.n_states = None
        self.n_actions = None
        self.action_bound = None
        self.device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')

class ReplayBuffer:
    def __init__(self, cfg):
        self.buffer = np.empty(cfg.memory_capacity, dtype=object)
        self.size = 0
        self.pointer = 0
        self.capacity = cfg.memory_capacity
        self.batch_size = cfg.batch_size
        self.device = cfg.device

    def push(self, transitions):
        self.buffer[self.pointer] = transitions
        self.size = min(self.size + 1, self.capacity)
        self.pointer = (self.pointer + 1) % self.capacity

    def clear(self):
        self.buffer = np.empty(self.capacity, dtype=object)
        self.size = 0
        self.pointer = 0

    def sample(self):
        batch_size = min(self.batch_size, self.size)
        indices = np.random.choice(self.size, batch_size, replace=False)
        print(self.buffer[indices])
        samples = map(lambda x: torch.tensor(np.array(x), dtype=torch.float32,
                                             device=self.device), zip(*self.buffer[indices]))
        return samples


class Actor(nn.Module):
    def __init__(self, cfg):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(cfg.n_states, cfg.actor_hidden_dim)
        self.fc2 = nn.Linear(cfg.actor_hidden_dim, cfg.actor_hidden_dim)
        self.fc3 = nn.Linear(cfg.actor_hidden_dim, cfg.n_actions)
        self.action_bound = torch.tensor(cfg.action_bound, dtype=torch.float32, device=cfg.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.action_bound
        return action


class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(cfg.n_states + cfg.n_actions, cfg.critic_hidden_dim)
        self.fc2 = nn.Linear(cfg.critic_hidden_dim, cfg.critic_hidden_dim)
        self.fc3 = nn.Linear(cfg.critic_hidden_dim, 1)

        self.fc4 = nn.Linear(cfg.n_states + cfg.n_actions, cfg.critic_hidden_dim)
        self.fc5 = nn.Linear(cfg.critic_hidden_dim, cfg.critic_hidden_dim)
        self.fc6 = nn.Linear(cfg.critic_hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)

        q1 = F.relu(self.fc1(cat))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(cat))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2

    @torch.jit.export
    def Q1(self, x, a):
        cat = torch.cat([x, a], dim=1)
        q1 = F.relu(self.fc1(cat))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1


class TD3:
    def __init__(self, cfg):
        self.cfg = cfg
        self.memory = ReplayBuffer(cfg)
        self.total_up = 0
        self.scaler = GradScaler()

        self.actor = Actor(cfg).to(cfg.device)
        self.actor = torch.jit.script(self.actor)
        self.actor_target = deepcopy(self.actor)
        self.actor_target = torch.jit.script(self.actor_target)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.lr_a)

        self.critic = Critic(cfg).to(cfg.device)
        self.critic = torch.jit.script(self.critic)
        self.critic_target = deepcopy(self.critic)
        self.critic_target = torch.jit.script(self.critic_target)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.lr_c)
        self.epsilon = 0.9
        self.epsilon_decay = 0.99
    

    @torch.no_grad()
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            self.epsilon *= self.epsilon_decay
            return [random.randrange(0, self.cfg.action_bound)]

        state = torch.tensor(state, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
        action = self.actor(state).squeeze(0).cpu().numpy()
        return action

    def update(self):
        if self.memory.size < self.cfg.batch_size:
            return 0, 0
        self.total_up += 1
        states, actions, rewards, next_states, dones = self.memory.sample()
        actions, rewards, dones = actions.view(-1, 1), rewards.view(-1, 1), dones.view(-1, 1)

        with autocast():
            with torch.no_grad():
                noise = (torch.randn_like(actions, device=self.cfg.device) *
                        self.cfg.policy_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
                next_actions = (self.actor_target(next_states) + noise).clamp(-self.cfg.action_bound, self.cfg.action_bound)
                target_q1, target_q2 = self.critic_target(next_states, next_actions)
                target_q = rewards + (1 - dones) * self.cfg.gamma * torch.min(target_q1, target_q2)

            q1, q2 = self.critic(states, actions)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            
        self.critic_optim.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_optim)
        self.scaler.update()

        actor_loss = torch.tensor(0.0, device=self.cfg.device)

        if self.total_up % self.cfg.policy_freq == 0:
            for params in self.critic.parameters():
                params.requires_grad = False
            with autocast():
                actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            self.actor_optim.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.step(self.actor_optim)
            self.scaler.update()

            for params in self.critic.parameters():
                params.requires_grad = True
            self.update_params()

        print('here in loss ============')

        return critic_loss.item(), actor_loss.item()

    def update_params(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data +
                                    (1. - self.cfg.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data +
                                    (1. - self.cfg.tau) * target_param.data)



def normalization(big_cpu_freq, little_cpu_freq,big_util, little_util, mem, fps):
    big_cpu_freq = int(big_cpu_freq) / int(cpu_freq_list[1][-1])
    little_cpu_freq = int(little_cpu_freq) / int(cpu_freq_list[0][-1])
    mem = int(mem) / 1e6
    fps = int(fps) / 60
    return big_cpu_freq, little_cpu_freq, float(big_util), float(little_util),int(mem), fps

def get_reward(fps, target_fps, big_clock, little_clock):
    reward = (fps - target_fps) * 20 +  ( big_clock + little_clock  ) *  (-100) 
    return reward

def process_action(action):
    print(action)
    action = abs(action[0])
    action = round(action)
    action1, action2 = action % 3 , action // 3
    return [0 , action1, action2, 0]
    
from datetime import datetime

if __name__=="__main__":
    cfg = Config()
    cfg.n_states = 6
    cfg.n_actions = 1
    cfg.action_bound = 9
    agent = TD3(cfg)

    cpu_freq_list, gpu_freq_list = tools.get_freq_list('k20p')
    little_cpu_clock_list = tools.uniformly_select_elements(8, cpu_freq_list[0])
    big_cpu_clock_list = tools.uniformly_select_elements(8, cpu_freq_list[1])
    super_big_cpu_clock_list = tools.uniformly_select_elements(6, cpu_freq_list[2])   # 若是没有超大核，则全部为0

    state=(0,0,0,0,0,0)
    action=[0]
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
            start_time = datetime.now()

            # 打点1：开始处理socket数据
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
            end_t1 = datetime.now()
            print(f"[Time] Socket data processing took: {end_t1 - t1}")

            f.write(f'{t},{big_cpu_freq},{little_cpu_freq},{big_util},{little_util},{ipc},{cache_miss},{fps},{action[0]},{aloss},{closs},{reward}\n')
            f.flush() 
            
            # 打点2：开始数据归一化
            t2 = datetime.now()
            big_cpu_freq, little_cpu_freq, big_util, little_util, mem, fps = normalization(big_cpu_freq, little_cpu_freq, big_util, little_util, mem, fps)
            end_t2 = datetime.now()
            print(f"[Time] Normalization took: {end_t2 - t2}")

            # 解析数据
            next_state = (big_cpu_freq, little_cpu_freq, big_util, little_util, mem, fps)
            
            # 打点3：开始计算reward
            t3 = datetime.now()
            reward = get_reward(fps, target_fps, big_cpu_freq, little_cpu_freq)
            end_t3 = datetime.now()
            print(f"[Time] Reward calculation took: {end_t3 - t3}")

            agent.memory.push((state, action[0], reward, next_state, 0))


            # 获得action
            action = agent.choose_action(state)
            processed_action = process_action(action)
           
            done = 1

            # 打点4：开始发送socket数据
            t5 = datetime.now()
            res = send_socket_data(f'1,{big_cpu_clock_list[processed_action[1]]},{little_cpu_clock_list[processed_action[2]]}')
            end_t5 = datetime.now()
            print(f"[Time] Sending socket data took: {end_t5 - t5}")

            # 打点5：开始训练模型
            if (t > 5):
                closs, aloss = agent.update()

            if int(res) == -1:
                print('freq set error')
                break
            
            # update some state
            state = next_state
            t += 1

            time.sleep(0.01)

    finally:
        f.close()
