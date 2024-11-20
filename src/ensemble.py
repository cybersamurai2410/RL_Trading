model_name = ['PPO', 'A2C', 'DQN']
model_index = 0

actions_dict = {}
actions = []

prices_dict = {}
prices = []

history = []

for model in [best_model_ppo, best_model_a2c, best_model_dqn]:
  env = gym.make('stocks-v0', df=df, frame_bound=(90,110), window_size=5)
  obs = env.reset()

  while True:
      obs = obs[np.newaxis, ...] # Reshape for non-vectorized environment
      action, _states = model.predict(obs)
      obs, rewards, done, info = env.step(action)
      actions.append(action)
      prices.append(obs[4][0])
      # print(obs)

      if done:
        print(model_name[model_index])
        actions_dict[model_name[model_index]] = actions
        prices_dict[model_name[model_index]] = prices
        actions = []
        prices = []
        model_index += 1

        history.append(env.history)
        print("info", info)
        plt.figure(figsize=(15,6))
        plt.cla()
        env.render_all()

        plt.xlabel('Trading Days')
        plt.ylabel('Closing Price')

        plt.show()
        break

eval_env = gym.make('stocks-v0', df=df, frame_bound=(90, 110), window_size=5)

for model in [best_model_ppo, best_model_a2c, best_model_dqn]:
  # Evaluate the policy of the trained models
  print(model)
  mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

  print("Mean Reward:", mean_reward)
  print("Std Reward:", std_reward)
  print()

print("Actions")
print('PPO: ', actions_dict['PPO'])
print('A2C: ', actions_dict['A2C'])
print('DQN: ', actions_dict['DQN'])

print("\nPrices")
print('PPO: ', prices_dict['PPO'])
print('A2C: ', prices_dict['A2C'])
print('DQN: ', prices_dict['DQN'])

# Combine actions using majority voting
voting_actions = np.array([actions_dict['PPO'], actions_dict['A2C'], actions_dict['DQN']])

# Take the majority vote for each timestep
ensemble_actions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=voting_actions)
print("Ensemble Actions: ", ensemble_actions)

# Build environment applying the voted actions
env = gym.make('stocks-v0', df=df, frame_bound=(90,110), window_size=5)
obs = env.reset()

for action in ensemble_actions:
      obs = obs[np.newaxis, ...]
      obs, rewards, done, info = env.step(action)

      if done:
        print("info", info)
        break

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()
