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
sys.path.append('D:/fkty/Desktop/oppoooo/new/oppo-RL-master/oppo-RL-master/')
import utils.tools as tools
# import Pixel_3a.PMU.pmu as pmu
# import Pixel_3a.PowerLogger.powerlogger as powerlogger
# from SurfaceFlinger.get_fps import SurfaceFlingerFPS
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
    cmds = ['su',cmd, 'exit']
    obj = subprocess.Popen("adb shell", shell= True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info = obj.communicate(("\n".join(cmds) + "\n").encode('utf-8'))
    return info[0].decode('utf-8')

def send_socket_data(message, host='192.168.0.7', port=8888): # 192.168.0.8:8888
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

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(torch.utils.data.Dataset):
    """
    Basic ReplayMemory class. 
    Note: Memory should be filled before load.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __getitem__(self, idx):        
        return self.memory[idx] 

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            # to avoid index out of range
            self.memory.append(None)
        transition = Transition(*args)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity


# RL Controller with action branching
class DQN_AB(nn.Module):
    def __init__(self, s_dim=10, h_dim=25, branches=[1,2,3]):
        super(DQN_AB, self).__init__()
        self.s_dim, self.h_dim = s_dim, h_dim
        self.branches = branches
        self.shared = nn.Sequential(nn.Linear(self.s_dim, self.h_dim), nn.ReLU())
        self.shared_state = nn.Sequential(nn.Linear(self.h_dim, self.h_dim), nn.ReLU())
        self.domains, self.outputs = [], []
        for i in range(len(branches)):
            layer = nn.Sequential(nn.Linear(self.h_dim, self.h_dim), nn.ReLU())
            self.domains.append(layer)
            layer_out = nn.Sequential(nn.Linear(self.h_dim*2, branches[i]))
            self.outputs.append(layer_out)

    def forward(self, x):
        # return list of tensors, each element is Q-Values of a domain
        f = self.shared(x)
        s = self.shared_state(f)
        outputs = []
        for i in range(len(self.branches)):
            branch = self.domains[i](f)
            branch = torch.cat([branch,s],dim=1)
            outputs.append(self.outputs[i](branch))
        return outputs

# Agent with action branching without time context
class DQN_AGENT_AB():
    def __init__(self, s_dim, h_dim, branches, buffer_size, params, nas_model):
        self.eps = 0.8
        # 2D action space
        self.actions = [np.arange(i) for i in branches]
        # Experience Replay(requires belief state and observations)
        self.mem = ReplayMemory(buffer_size)
        self.state_size = s_dim
        self.action_size = len(branches)
        self.nas_model = nas_model # bool值：是否使用nas模型

        if self.nas_model:
            # Initialize DARTS model
            pass
        else:
            # Initialize traditional DQN model
            self.policy_net = DQN_AB(s_dim, h_dim, branches)
            self.target_net = DQN_AB(s_dim, h_dim, branches)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        
    def max_action(self, state):
        # actions for multidomains
        max_actions = []
        with torch.no_grad():
            # Inference using policy_net given (domain, batch, dim)
            q_values = self.policy_net(state)
            for i in range(len(q_values)):
                domain = q_values[i].max(dim=1).indices
                max_actions.append(self.actions[i][domain])
        return max_actions

    def e_gready_action(self, actions, eps):
        # Epsilon-Gready for exploration
        final_actions = []
        for i in range(len(actions)):
            p = np.random.random()
            if isinstance(actions[i],np.ndarray):
                if p < 1- eps:
                    final_actions.append(actions[i])
                else:
                    # randint in (0, domain_num), for batchsize
                    final_actions.append(np.random.randint(len(self.actions[i]),size=len(actions[i])))
            else:
                if p < 1- eps:
                    final_actions.append(actions[i])
                else:
                    final_actions.append(np.random.choice(self.actions[i]))

        return final_actions

    def select_action(self, state):
        return self.e_gready_action(self.max_action(state),self.eps)

    def train(self, n_round, n_update, n_batch):
        # Train on policy_net
        losses = []
        GAMMA = 0.99  # Use a more standard value for GAMMA

        self.target_net.train()
        train_loader = torch.utils.data.DataLoader(self.mem, shuffle=True, batch_size=n_batch, drop_last=True)
        length = len(train_loader.dataset)
        # Calculate loss for each branch and then simply sum up
        for i, trans in enumerate(train_loader):
            loss = 0.0  # initialize loss at the beginning of each batch
            states, actions, next_states, rewards = trans
            with torch.no_grad():
                target_result = self.target_net(next_states)
            policy_result = self.policy_net(states)
            # Loop through each action domain
            for j in range(len(self.actions)):
                next_state_values = target_result[j].max(dim=1)[0]
                expected_state_action_values = rewards.float() + (next_state_values * GAMMA)
                # Gather action-values that have been taken
                branch_actions = actions[:, j].long()
                state_action_values = policy_result[j].gather(1, branch_actions.unsqueeze(1))

                # Check for NaN in state_action_values and expected_state_action_values
                if torch.isnan(state_action_values).any() or torch.isnan(expected_state_action_values).any():
                    print("NaN detected in state_action_values or expected_state_action_values")
                    continue

                loss += self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)

            if i > n_update:
                break
            self.optimizer.step()
        return losses


    def save_model(self, n_round, savepath):
        train_utils.save_checkpoint({'epoch': n_round, 'model_state_dict':self.target_net.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict()}, savepath)


    def load_model(self, loadpath):
        if not os.path.isdir(loadpath): os.makedirs(loadpath)
        checkpoint = train_utils.load_checkpoint(loadpath)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.eval()

    def sync_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


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
    action1, action2 = action[0], action[1]
    return [0 , action1, action2, 0]
    
if __name__=="__main__":

    # s_dim: 状态维度，即输入的状态向量的维度。
    # h_dim: 隐藏层的维度，即神经网络中间层的神经元数量。
    # branches : 每个分支的action的数量
    agent = DQN_AGENT_AB(6, 15, [3,3], 200, None, True)

    cpu_freq_list, gpu_freq_list = tools.get_freq_list('pixel3') # k20p
    little_cpu_clock_list = tools.uniformly_select_elements(8, cpu_freq_list[0])
    big_cpu_clock_list = tools.uniformly_select_elements(8, cpu_freq_list[1])
    super_big_cpu_clock_list = tools.uniformly_select_elements(6, cpu_freq_list[2])   # 若是没有超大核，则全部为0

    state=(0,0,0,0,0,0)
    action=[0,0]
    losses = 0
    experiment_time=1000
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

            f.write(f'{t},{big_cpu_freq},{little_cpu_freq},{big_util}, {little_util}, {ipc}, {cache_miss}, {fps},{action}, {losses},{reward}\n')
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
            agent.mem.push(torch.tensor(state).float(), torch.tensor(action).float(), torch.tensor(next_state).float(), torch.tensor(reward).float())

            # 获得action
            with torch.no_grad():
                action = agent.select_action(agent.mem[-1].state.unsqueeze(0))

            # 获得action
            processed_action = process_action(action)

            print(processed_action)

            # 设置action
            # tools.set_gpu_freq(gpu_freq_list[action[0]], action[0])
            # tools.set_cpu_freq([little_cpu_clock_list[processed_action[1]], big_cpu_clock_list[processed_action[2]], super_big_cpu_clock_list[processed_action[3]]]) # 若没有超大核，实际超大核不会设置
            res = send_socket_data(f'1,{big_cpu_clock_list[processed_action[1]]},{little_cpu_clock_list[processed_action[2]]}')
            if(int(res) == -1):
                print('freq set error')
                break

            if (t > 5):
                losses = agent.train(1,1,1) 
                losses = np.sum(losses)
            
            # update some state
            state = next_state
            t += 1



    finally:
        f.close()
 