import sys
import time
import gym_super_mario_bros
import torch
import torch.nn as nn
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrappers import *
import argparse
import keyboard


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
        # print("++++++++++++++++++++++++++++++++++++")
        # print(f"state shape in network: {x.shape}")        #[1,4,84,84],对应1个样本，4帧图片叠加，84x84像素的图像
        # print("++++++++++++++++++++++++++++++++++++")
        x = torch.relu(self.layer1(x))      # 卷积+激活
        x = torch.relu(self.layer2(x))
        x = x.view(-1, 20736)
        x = torch.relu(self.fc(x))     

        A = self.fc_A(x)
        V=self.fc_V(x)
        Q=V+(A-1/A.shape[-1]*A.max(-1,True)[0])

        return Q



def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def arange(s):
    if not type(s) == "numpy.ndarray":
        s = np.array(s)
    assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1))
    ret=np.expand_dims(ret, 0)
    return ret


def parse_opt(known=False):
    parser = argparse.ArgumentParser(description="Super Mario RL Model Loader")
    parser.add_argument("--model_name", type=str, default="mario_q_target.pth", help="Path to the model file (default: mario_q_target.pth)")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt):
    current_file_path = os.path.abspath(__file__)
    file_path = os.path.dirname(current_file_path)
    model_dir=os.path.join(file_path,"models")
    model_path = os.path.join(model_dir,opt.model_name)
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Load model from {model_path}.")
    print("++++++++++++++++++++++++++++++++++++++++++++++++")

    n_frame = 4
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q_net = VAnet(n_frame, env.action_space.n, device).to(device)

    q_net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    total_score = 0.0
    done = False
    state = arange(env.reset())
    while not done:
        if done:
            state = arange(env.reset())
        env.render()
        action = q_net(state).argmax().item()
        next_state, reward, done, _ = env.step(action)
        next_state = arange(next_state)
        total_score += reward
        state = next_state
        time.sleep(0.01)
        if keyboard.is_pressed('q'):
            env.close()
            break

    stage = env.unwrapped._stage
    print("Total score : %f | stage : %d" % (total_score, stage))

if __name__ == "__main__":
    opt=parse_opt()
    main(opt)