from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import HerReplayBuffer, A2C, PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy

import optuna
from environment import build_environment 

def optimize_hyperparameters(trial: optuna.Trial, env, model_cls) -> float:

    if model_cls is DQN:
      learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
      buffer_size = trial.suggest_categorical('buffer_size', [int(1e3), int(1e4), int(1e5)])
      batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
      gamma = trial.suggest_float('gamma', 0.9, 0.999)
      train_freq = trial.suggest_categorical('train_freq', [1, 4, 8])

      model = model_cls("MlpPolicy", env, learning_rate=learning_rate, buffer_size=buffer_size,
                  batch_size=batch_size, gamma=gamma, train_freq=train_freq)

    elif model_cls is PPO:
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        n_steps = trial.suggest_categorical('n_steps', [64, 128, 256])
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        gamma = trial.suggest_float('gamma', 0.9, 0.999)
        gae_lambda = trial.suggest_float('gae_lambda', 0.8, 0.99)

        model = model_cls("MlpPolicy", env, learning_rate=learning_rate, n_steps=n_steps,
                          batch_size=batch_size, gamma=gamma, gae_lambda=gae_lambda)

    elif model_cls is A2C:
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        n_steps = trial.suggest_categorical('n_steps', [64, 128, 256])
        gamma = trial.suggest_float('gamma', 0.9, 0.999)

        model = model_cls("MlpPolicy", env, learning_rate=learning_rate, n_steps=n_steps,
                          gamma=gamma)

    model.learn(total_timesteps=int(1e5), log_interval=1000)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

    env.close()

    return mean_reward

env = build_environment('apple.csv')

# Perform hyperparameter optimization for DQN
study_dqn = optuna.create_study(direction='maximize')
study_dqn.optimize(lambda trial: optimize_hyperparameters(trial, env, DQN), n_trials=100)

# Perform hyperparameter optimization for PPO
study_ppo = optuna.create_study(direction='maximize')
study_ppo.optimize(lambda trial: optimize_hyperparameters(trial, env, PPO), n_trials=100)

# Perform hyperparameter optimization for A2C
study_a2c = optuna.create_study(direction='maximize')
study_a2c.optimize(lambda trial: optimize_hyperparameters(trial, env, A2C), n_trials=100)

# Get the best hyperparameters for each model
best_params_dqn = study_dqn.best_params
best_params_ppo = study_ppo.best_params
best_params_a2c = study_a2c.best_params

print("Best Hyperparameters for DQN:", best_params_dqn)
print("Best Hyperparameters for PPO:", best_params_ppo)
print("Best Hyperparameters for A2C:", best_params_a2c)

# Train the models with the best hyperparameters
tb_log = "./rlmodel_optimised/"
print("Training optimised DQN...")
best_model_dqn = DQN("MlpPolicy", env, learning_rate=best_params_dqn['learning_rate'],
                    buffer_size=best_params_dqn['buffer_size'], batch_size=best_params_dqn['batch_size'],
                    gamma=best_params_dqn['gamma'], train_freq=best_params_dqn['train_freq'], verbose=1, tensorboard_log=tb_log)
best_model_dqn.learn(total_timesteps=int(1e5))

print("Training optimised PPO...")
best_model_ppo = PPO("MlpPolicy", env, learning_rate=best_params_ppo['learning_rate'],
                    n_steps=best_params_ppo['n_steps'], batch_size=best_params_ppo['batch_size'],
                    gamma=best_params_ppo['gamma'], gae_lambda=best_params_ppo['gae_lambda'], verbose=1, tensorboard_log=tb_log)
best_model_ppo.learn(total_timesteps=int(1e5))

print("Training optimised A2C...")
best_model_a2c = A2C("MlpPolicy", env, learning_rate=best_params_a2c['learning_rate'],
                    n_steps=best_params_a2c['n_steps'], gamma=best_params_a2c['gamma'], verbose=1, tensorboard_log=tb_log)
best_model_a2c.learn(total_timesteps=int(1e5))
