from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time
from matplotlib import pyplot as plt
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
import os
from stable_baselines3 import PPO
import keyboard
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


# path
monitor_dir = os.path.join(os.getcwd(),"Mario_me" ,"monitor_dir")

if not os.path.exists(monitor_dir):
    os.makedirs(monitor_dir)

tensorboard_log = os.path.join(os.getcwd(),"Mario_me" , "tensorboard_log")

if not os.path.exists(tensorboard_log):
    os.makedirs(tensorboard_log)

model_path = os.path.join(os.getcwd(), "Mario_me" ,"models", "model_3000000.zip")
if not os.path.exists(model_path):
    print("完蛋噜,你的模型不见了呜呜呜呜呜")

print('——————————————————————————————————————————————————————————————————————')
print(f"monitor_dir_path:{monitor_dir}")
print(f"tensorboard_log_path:{tensorboard_log}")
print(f"model_path:{model_path}")
print('——————————————————————————————————————————————————————————————————————')


model = PPO.load(model_path)
## env init
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = Monitor(env, monitor_dir)
env=GrayScaleObservation(env,keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4)

## test
obs = env.reset()
obs=obs.copy()
done = True
while True:
    if done:
        state = env.reset()
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    obs=obs.copy()
    env.render()
    time.sleep(0.001)
    if keyboard.is_pressed('q'):
        env.close()
        break


