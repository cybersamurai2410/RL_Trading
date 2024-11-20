import torch as th
from torch import nn
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gym import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
from torch.nn import Conv1d, LSTM

class CustomNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 8,
        last_layer_dim_vf: int = 8,
        conv_filters: int = 32,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1
    ):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Convolutional1D Layer for policy_net
        self.policy_conv = nn.Conv1d(in_channels=1, out_channels=conv_filters, kernel_size=3)

        # LSTM Layer for policy_net
        self.policy_lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )

        # Linear Layer for policy_net after LSTM
        self.policy_linear = nn.Linear(lstm_hidden_size, last_layer_dim_pi)

        # Convolutional1D Layer for value_net
        self.value_conv = nn.Conv1d(in_channels=1, out_channels=conv_filters, kernel_size=3)

        # LSTM Layer for value_net
        self.value_lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )

        # Linear Layer for value_net after LSTM
        self.value_linear = nn.Linear(lstm_hidden_size, last_layer_dim_vf)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        x = features.unsqueeze(1)  # Add an extra dimension for Conv1D input
        x = self.policy_conv(x)
        x = x.squeeze(2)  # Remove the extra dimension added by Conv1D
        x = x.permute(0, 2, 1)  # Permute dimensions for LSTM input
        x, _ = self.policy_lstm(x)
        x = self.policy_linear(x[:, -1, :])  # Use only the last LSTM output

        return x

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        x = features.unsqueeze(1)  # Add an extra dimension for Conv1D input
        x = self.value_conv(x)
        x = x.squeeze(2)  # Remove the extra dimension added by Conv1D
        x = x.permute(0, 2, 1)  # Permute dimensions for LSTM input
        x, _ = self.value_lstm(x)
        x = self.value_linear(x[:, -1, :])  # Use only the last LSTM output

        return x

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule=lr_schedule,
            *args,
            **kwargs,
        )
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

tb_log = "./rlcustom_ac/"
env = gym.make('stocks-v0', df=df, frame_bound=(5, 100), window_size=5)
env = DummyVecEnv([lambda: env])

stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=100, verbose=1)
eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)

# Create a PPO agent with the custom policy
best_model_ppo = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log=tb_log)

# best_model_ppo = PPO(policy=CustomActorCriticPolicy, env=env, learning_rate=best_params_ppo['learning_rate'],
#                     n_steps=best_params_ppo['n_steps'], batch_size=best_params_ppo['batch_size'],
#                     gamma=best_params_ppo['gamma'], gae_lambda=best_params_ppo['gae_lambda'], verbose=1, tensorboard_log=tb_log)

best_model_ppo.learn(total_timesteps=int(1e6), callback=eval_callback)

# Evaluate the trained agent
eval_env = gym.make('stocks-v0', df=df, frame_bound=(90, 110), window_size=5)
obs = eval_env.reset()
while True:
    obs = obs[np.newaxis, ...]  # Reshape for non-vectorized environment
    action, _states = best_model_ppo.predict(obs)
    obs, rewards, done, info = eval_env.step(action)

    if done:
        print("info", info)

        # Render the environment
        plt.figure(figsize=(15, 6))
        plt.cla()
        eval_env.render_all()

        plt.xlabel('Trading Days')
        plt.ylabel('Closing Price')

        plt.show()
        break

# Create a A2C agent with the custom policy
env = gym.make('stocks-v0', df=df, frame_bound=(5, 100), window_size=5)
env = DummyVecEnv([lambda: env])

stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=100, verbose=1)
eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)

best_model_a2c = A2C(CustomActorCriticPolicy, env, verbose=1, tensorboard_log=tb_log)

# best_model_a2c = A2C(CustomActorCriticPolicy, env, learning_rate=best_params_a2c['learning_rate'],
#                     n_steps=best_params_a2c['n_steps'], gamma=best_params_a2c['gamma'], verbose=1, tensorboard_log=tb_log)
best_model_a2c.learn(total_timesteps=int(1e6), callback=eval_callback)

eval_env = gym.make('stocks-v0', df=df, frame_bound=(90, 110), window_size=5)
obs = eval_env.reset()
while True:
    obs = obs[np.newaxis, ...]
    action, _states = best_model_a2c.predict(obs)
    obs, rewards, done, info = eval_env.step(action)

    if done:
        print("info", info)

        plt.figure(figsize=(15, 6))
        plt.cla()
        eval_env.render_all()

        plt.xlabel('Trading Days')
        plt.ylabel('Closing Price')

        plt.show()
        break
