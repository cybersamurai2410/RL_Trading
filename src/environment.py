import gym
import gym_anytrading
from gym import spaces

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import HerReplayBuffer, A2C, PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_df(path):
  # Load dataset for stock 
  df = pd.read_csv(path)
  
  # Convert Date column to datetime type
  df['Date'] = pd.to_datetime(df['Date'])
  df = df.sort_values('Date')
  
  # Run if commas in prices for each column
  df['Open'] = df['Open'].str.replace(',', '').astype(float)
  df['High'] = df['High'].str.replace(',', '').astype(float)
  df['Low'] = df['Low'].str.replace(',', '').astype(float)
  df['Close'] = df['Close'].str.replace(',', '').astype(float)
  df['Volume'] = df['Volume'].str.replace(',', '').astype(float)

  return df

# Build environment
def build_env(path):
  df = load_df(path)
  env = gym.make('stocks-v0', df=df, frame_bound=(5,100), window_size=5)
  env = DummyVecEnv([lambda: env])
  
  wrapped_env = env.envs[0]
  print("Environment: ", wrapped_env.spec.id)
  print("Observation Space: ", wrapped_env.observation_space)
  print("Action Space: ", wrapped_env.action_space) 
  print("Data Shape: ", wrapped_env.shape) 
  print('\n')
  print("Signal Features: \n", wrapped_env.signal_features)
  print("Prices: \n", wrapped_env.prices) 
  print("Max Possible Profit: \n", wrapped_env.max_possible_profit()) 

  return env
