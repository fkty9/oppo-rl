#!/usr/bin/env python3

import warnings
import os
from threading import Thread
import time
import random
import socket
import struct
import math
from collections import namedtuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import train_utils as train_utils
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append('/home/wakaba/Desktop/zTT/')
import utils.tools as tools
import subprocess
from torchsummary import summary
import torch
from torch import nn, optim
from torch.nn import functional as F
from collections import deque
from torch.cuda.amp import GradScaler, autocast
import socket
import pickle


def execute(cmd):
    # print(cmd)
    cmds = [ 'su',cmd, 'exit']
    # cmds = [cmd, 'exit']
    obj = subprocess.Popen("adb shell", shell= True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info = obj.communicate(("\n".join(cmds) + "\n").encode('utf-8'))
    return info[0].decode('utf-8')


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

class Config:
    def __init__(self):
        self.algo_name = 'DQN'
        self.train_eps = 500
        self.test_eps = 5
        self.max_steps = 100000
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 800
        self.lr = 0.001
        self.gamma = 0.9
        self.seed = random.randint(0, 100)
        self.batch_size = 32
        self.memory_capacity = 100000
        self.hidden_dim = 256
        self.target_update = 4
        self.n_states = None
        self.n_actions = None
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def show(self):
        print('-' * 30 + '参数列表' + '-' * 30)
        for k, v in vars(self).items():
            print(k, '=', v)
        print('-' * 60)

class MLP(nn.Module):
    def __init__(self, n_states, n_actions, n_dims=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_states, n_dims)
        self.fc2 = nn.Linear(n_dims, n_dims)
        self.fc3 = nn.Linear(n_dims, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, transitions):
        self.buffer.append(transitions)

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size, sequential = False):
        batch_size = min(batch_size, len(self.buffer))
        if sequential: # 获得有序的buffer数据，最近的若干个数据
            rand_index = random.randint(0, len(self.buffer) - batch_size + 1)
            batch = [self.buffer[i] for i in range(rand_index, rand_index + batch_size)]
        else:
            batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def size(self):
        return len(self.buffer)

    def dump(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.buffer, f)
        print(f"Replay buffer dumped to {filename}")

    def load(self, filename):
        """从文件加载数据到缓冲区"""
        with open(filename, 'rb') as f:
            self.buffer = pickle.load(f)
        print(f"Replay buffer loaded from {filename}")


class DQN:
    def __init__(self, policy_net, target_net, memory, cfg):
        self.sample_count = 0
        self.memory = memory
        self.policy_net = policy_net
        self.target_net = target_net
        self.cfg = cfg
        self.epsilon = cfg.epsilon_start
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.scaler = GradScaler()

    @torch.no_grad()
    def choose_action(self, state):
        self.sample_count += 1
        self.epsilon = self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * \
                       np.exp(-1. * self.sample_count / self.cfg.epsilon_decay)
        if random.uniform(0, 1) > self.epsilon:
            state = torch.tensor(np.array(state), device=self.cfg.device, dtype=torch.float32).unsqueeze(0)
            q_value = self.policy_net(state)
            action = q_value.argmax(dim=1).item()
        else:
            action = random.randrange(self.cfg.n_actions)
        return action

    @torch.no_grad()
    def predict_action(self, state):
        state = torch.tensor(np.array(state), device=self.cfg.device, dtype=torch.float32).unsqueeze(0)
        q_value = self.policy_net(state)
        action = q_value.argmax(dim=1).item()
        return action

    def update(self):
        if self.memory.size() < self.cfg.batch_size:
            return 0
        state_batch, action_batch, reward_batch, next_state_batch = self.memory.sample(self.cfg.batch_size)
        state_batch = torch.tensor(np.array(state_batch), device=self.cfg.device, dtype=torch.float32)
        action_batch = torch.tensor(np.array(action_batch), device=self.cfg.device, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(np.array(reward_batch), device=self.cfg.device, dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.cfg.device, dtype=torch.float32)

        with autocast():
            q_value = self.policy_net(state_batch).gather(1, action_batch)
            next_q_value = self.target_net(next_state_batch).max(dim=1)[0].detach()
            expect_q_value = reward_batch + self.cfg.gamma * next_q_value

            # 检查是否存在 NaN
            if torch.isnan(q_value).any():
                print('nan in q_value')
                print(state_batch)
                exit(0)
            
            if torch.isnan(expect_q_value).any():
                print('nan in expect_q_value')
                print(state_batch)
                exit(0)


            loss = F.mse_loss(q_value, expect_q_value.unsqueeze(1))

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        # 取消梯度缩放后进行梯度裁剪
        self.scaler.unscale_(self.optimizer)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss

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
    # print(action)
    action1, action2 = action % 8 , action // 8
    return [0 , action1, action2, 0]
    
if __name__=="__main__":

    # s_dim: 状态维度，即输入的状态向量的维度。
    # h_dim: 隐藏层的维度，即神经网络中间层的神经元数量。
    # branches : 每个分支的action的数量
    cfg = Config()
    n_states = 6
    n_actions = 64
    cfg.n_actions = n_actions
    cfg.n_states = n_states
    
    policy_net = torch.jit.script(MLP(n_states, n_actions, cfg.hidden_dim).to(cfg.device)) 
    target_net = torch.jit.script(MLP(n_states, n_actions, cfg.hidden_dim).to(cfg.device)) 
    memory = ReplayBuffer(cfg.memory_capacity)
    agent = DQN(policy_net, target_net, memory, cfg)

    cpu_freq_list, gpu_freq_list = tools.get_freq_list('k20p')
    little_cpu_clock_list = tools.uniformly_select_elements(8, cpu_freq_list[0])
    big_cpu_clock_list = tools.uniformly_select_elements(8, cpu_freq_list[1])
    super_big_cpu_clock_list = tools.uniformly_select_elements(6, cpu_freq_list[2])   # 若是没有超大核，则全部为0

    state=(0,0,0,0,0,0)
    action=0
    loss = 0
    experiment_time=10000
    target_fps=25
    reward = 0

    f = open("output.csv", "w")
    f.write(f'episode,big_cpu_freq,little_cpu_freq,big_util,little_util,ipc,cache_miss,fps,action,loss,reward\n')

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

            f.write(f'{t},{big_cpu_freq},{little_cpu_freq},{big_util}, {little_util}, {ipc}, {cache_miss}, {fps},{action}, {loss},{reward}\n')
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

            # replay memory
            if t != 1:
                agent.memory.push((np.array(state), action, reward, np.array(next_state)))

            # 获得action
            action = agent.choose_action(state)
            print(action)
            
            processed_action = process_action(action)

            # print(processed_action)

            # 设置action
            # tools.set_gpu_freq(gpu_freq_list[action[0]], action[0])
            # tools.set_cpu_freq([little_cpu_clock_list[processed_action[1]], big_cpu_clock_list[processed_action[2]], super_big_cpu_clock_list[processed_action[3]]]) # 若没有超大核，实际超大核不会设置
            res = send_socket_data(f'1,{big_cpu_clock_list[processed_action[1]]},{little_cpu_clock_list[processed_action[2]]}')
            if(int(res) == -1):
                print('freq set error')
                break

            if (t > 5):
                loss = agent.update()
                print('here loss is ', loss)
            
            # update some state
            state = next_state
            t += 1

            if t % 100 == 0:
                agent.memory.dump('replay_buffer.pkl')

            time.sleep(0.01)
             # 下面这段是记录进文件，供绘图使用的 
            # f.write(f'episode,big_cpu_freq,little_cpu_freq,gpu_freq,cpu_temp,fps\n')


    finally:
        f.close()
 