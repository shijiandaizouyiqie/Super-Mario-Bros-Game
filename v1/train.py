from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback



# path
# path
monitor_dir = os.path.join(os.getcwd(), "monitor_dir")
if not os.path.exists(monitor_dir):
    os.makedirs(monitor_dir)

tensorboard_log = os.path.join(os.getcwd(), "tensorboard_log")
if not os.path.exists(tensorboard_log):
    os.makedirs(tensorboard_log)


# env init
env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = Monitor(env, monitor_dir)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4)

# Callback Function
class SaveBestModelCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir=log_dir
        self.model_dir = os.path.join(self.log_dir, 'models')
        self.best_mean_reward = -np.inf
        os.makedirs(self.model_dir, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Save current model in every check_freq steps
            print(f"Saving model at step {self.n_calls}")
            self.model.save(os.path.join(self.model_dir, f'model_{self.n_calls}.zip'))

            # Load results and compute mean reward
        
        try:    # Save best model
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print(f"New best mean reward! {self.best_mean_reward:.2f}")
                    self.model.save(os.path.join(self.model_dir, 'best_model.zip'))  
        except Exception as e:
            print(f"Error loading results: {e}")

        return True
            
# Train model
train_params = {
    'learning_rate': 1e-6,
    'n_steps': 128
}

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=tensorboard_log, **train_params)

callback_f = SaveBestModelCallback(check_freq=100, log_dir=monitor_dir)
model.learn(total_timesteps=1000, callback=callback_f)
