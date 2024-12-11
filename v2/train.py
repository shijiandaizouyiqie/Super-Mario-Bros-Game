import pickle
import random
from tqdm import tqdm

from collections import deque

import gym_super_mario_bros
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrappers import *



def arrange(s):
    if not type(s) == "numpy.ndarray":
        s = np.array(s)       # 确保为numpy数组       
    assert len(s.shape) == 3                # 确保为3维数组[84,84,4]
    ret = np.transpose(s, (2, 0, 1))        # 转置为[4,84,84]
    ret=np.expand_dims(ret, 0)           # 增加一维[1,4,84,84]
    return ret

# 经验池
class replay_memory(object):
    def __init__(self, N):
        self.memory = deque(maxlen=N)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, n):
        return random.sample(self.memory, n)

    def len(self):
        return len(self.memory)

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

## Q网络
class VAnet(nn.Module):
    def __init__(self,n_frame,n_action,device):
        super(VAnet,self).__init__()
        self.device=device
        # AV共享部分
        self.layer1=nn.Conv2d(n_frame,32,8,4)       
        self.layer2=nn.Conv2d(32,64,3,1)
        self.fc=nn.Linear(20736, 512)

        self.fc_A=nn.Linear(512, n_action)
        self.fc_V=nn.Linear(512, 1)
        self.seq = nn.Sequential(self.layer1, self.layer2, self.fc, self.fc_A, self.fc_V)

        self.seq.apply(init_weights)

    def forward(self,x):        # x:[batch_size, n_frame, height, width]
        # 共同处理部分

        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x).to(self.device)
        x = torch.relu(self.layer1(x))      # 卷积+激活
        x = torch.relu(self.layer2(x))
        x = x.view(-1, 20736)
        x = torch.relu(self.fc(x))     

        A = self.fc_A(x)
        V=self.fc_V(x)
        Q=V+(A-1/A.shape[-1]*A.max(-1,True)[0])

        return Q


class Dueling_DQN:
    def __init__(self, n_state, n_action, device, epsilon, gamma, learning_rate, target_network_update,env):
        self.device = device
        self.q_net = VAnet(n_frame=n_state, n_action=n_action, device=device).to(device)
        self.q_target_net = VAnet(n_frame=n_state, n_action=n_action, device=device).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self.epsilon = epsilon
        self.gamma = gamma
        self.target_network_update = target_network_update
        self.cnt = 0
        self.env=env

    # 选择动作
    def take_action(self, state):
        state = torch.FloatTensor(state).to(self.device)  
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():  
                action = self.q_net(state).argmax().item()  
        return action

    def update_target_net(self):
        self.q_target_net.load_state_dict(self.q_net.state_dict())

    # 更新Q网络
    def update_q_net(self, replay_buffer, batch_size):
        states, rewards, actions, next_states, dones = list(map(np.array, zip(*replay_buffer.sample(batch_size))))    
        ## 数据类型转化，用于后续送入网络：numpy.ndarray->torch.Tensor 
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        states = states.cpu().numpy().squeeze(1)  # 去掉维度1，使形状变为 (256, 4, 84, 84)
        next_states= next_states.cpu().numpy().squeeze(1)  # 同样去掉维度1
           

        with torch.no_grad():
            next_actions = self.q_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.q_target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + (self.gamma * next_q_values * dones)

        q_values = self.q_net(states).gather(1, actions)

        loss = F.smooth_l1_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标Q网络
        if self.cnt % self.target_network_update == 0:
            self.update_target_net()
        self.cnt += 1
        return loss.item()


def train(env, agent, replay_buffer, epochs, batch_size, minimal_size):
    buffer_capacity = replay_buffer.memory.maxlen  # 经验池的最大容量

    for idx in range(10):
        with tqdm(total=int(epochs / 10), desc="Epochs %d" % idx) as pbar:
            # 新增replay_buffer的进度条
            replay_pbar = tqdm(total=buffer_capacity, desc="Replay Buffer Fill", position=1, leave=False)
            
            for i_episode in range(int(epochs / 10)):
                state = arrange(env.reset())  # 将state从[84,84,4]转化为[1,4,84,84]的格式
                done = False
                total_score = 0

                while not done:
                    action = agent.take_action(state)  # eg: action: 3
                    next_state, reward, done, info = env.step(action)

                    next_state = arrange(next_state)
                    total_score += reward
                    reward = np.sign(reward) * (np.sqrt(abs(reward) + 1) - 1) + 0.001 * reward
                    replay_buffer.push((state, float(reward), int(action), next_state, int(1 - done)))
                    state = next_state  # 更新状态
                    replay_pbar.update(1)

                    if replay_buffer.len() > minimal_size:
                        agent.update_q_net(replay_buffer, batch_size)

                if i_episode % 10 == 0:
                    pbar.set_postfix({
                        'epochs': i_episode,
                        'total_score': total_score
                    })
                pbar.update(1)
            
            # 重置 replay_buffer 进度条用于下一个 epoch
            replay_pbar.close()
        # 每隔10个epoch保存一次模型，避免训练中断模型丢失
        torch.save(agent.q_net.state_dict(), model_path)
        torch.save(agent.q_target_net.state_dict(), target_model_path)

current_file_path = os.path.abspath(__file__)
file_path= os.path.dirname(current_file_path)
model_dir=os.path.join(file_path,"models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path=os.path.join(model_dir,"mario_q.pth")
target_model_path=os.path.join(model_dir,"mario_q_target.pth")

def main():
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("+++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Device: {device}.")
    print("+++++++++++++++++++++++++++++++++++++++++++++++")
    
    Dueling_DQN_parms={
        "n_state":4,                        # 输入图像的通道数
        "n_action":env.action_space.n,      # 动作空间大小
        "device":device,                    # 设备  
        "epsilon":0.001,                    # 随机动作概率
        "gamma":0.99,                       # 奖励衰减因子
        "learning_rate" : 0.0001,           # 学习率
        "target_network_update":1000,       # 目标网络更新频率
        "env":env,
    }
    train_parms={
        "epochs":2000000,                   # 训练轮数
        "batch_size":256,                   # 每次训练的样本数
        "minimal_size":2000,                # 经验回放池最小样本数
    }

    # 经验回放池
    buffer_size=50000       # 经验池大小
    replay_buffer = replay_memory(buffer_size)

    agent = Dueling_DQN(**Dueling_DQN_parms)        # 初始化智能体
    train(env,agent,replay_buffer,**train_parms)      # 训练


if __name__ == "__main__":
    main()