from finta import TA
import pyfolio as pf

df_appl = pd.read_csv('/content/drive/MyDrive/Advanced Computing/Individual Project/data/STOCK_US_XNAS_AAPL.csv')
df_amzn = pd.read_csv('/content/drive/MyDrive/Advanced Computing/Individual Project/data/STOCK_US_XNAS_AMZN.csv')
df_msft = pd.read_csv('/content/drive/MyDrive/Advanced Computing/Individual Project/data/STOCK_US_XNAS_MSFT.csv')

# Preprocess datasets of each stock
dfs = [df_appl, df_amzn, df_msft]
for df in dfs:
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df['Volume'] = df['Volume'].str.replace(',', '').astype(float)

    # Technical Indicators
    df['SMA'] = TA.SMA(df, 12)
    df['RSI'] = TA.RSI(df)
    df['OBV'] = TA.OBV(df)

# Add stock name as prefix for their respecting columns before concatenating into a single dataset.
df_appl.columns = ['AAPL_' + col for col in df_appl.columns]
df_amzn.columns = ['AMZN_' + col for col in df_amzn.columns]
df_msft.columns = ['MSFT_' + col for col in df_msft.columns]

df = pd.concat([df_appl, df_amzn, df_msft], axis=1)
df.dropna(inplace=True)

class MultiStockTradingEnv(gym.Env):
    def __init__(self, df, stocks):
        super(MultiStockTradingEnv, self).__init__()

        self.df = df
        self.stocks = stocks
        self.num_stocks = len(stocks)

        # Define a continuous action space range between -1 and 1 for each stock
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_stocks,))

        # Observations are the close, low, volume, SMA, RSI, OBV, and our current holdings for each stock
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7 * self.num_stocks,))

        # Initialize state
        self.reset()

    def step(self, actions):
        self.current_step += 1
        rewards = []
        for i, action in enumerate(actions):
            stock = self.stocks[i]
            action = np.clip(action, -1, 1)  # Normalize the action to the range of [0, 2], then shift it to [-1, 1] so that it can represent selling, holding, and buying
            previous_portfolio_value = self.balance[i] + self.holdings[i] * self.df[f'{stock}_Close'].iloc[self.current_step-1]
            self.holdings[i] += action * self.balance[i] / self.df[f'{stock}_Close'].iloc[self.current_step]
            self.balance[i] -= action * self.balance[i]
            portfolio_value = self.balance[i] + self.holdings[i] * self.df[f'{stock}_Close'].iloc[self.current_step]
            reward = portfolio_value - previous_portfolio_value
            rewards.append(reward)

            # Calculate profit for each stock
            profit = portfolio_value - self.initial_balances[i]
            self.profits[i].append(profit)

        done = self.current_step >= len(self.df.index) - 1
        obs = self._get_obs()

        return obs, np.mean(rewards), done, {}

    def reset(self):
        self.current_step = 0
        self.balance = [10000 for _ in range(self.num_stocks)]  # Initial balance for each stock
        self.initial_balances = self.balance.copy()  # Store the initial balance for profit calculation
        self.holdings = [0 for _ in range(self.num_stocks)]  # Initial holdings for each stock
        self.profits = [[] for _ in range(self.num_stocks)]  # Initialize profits for each stock as empty lists

        return self._get_obs()

    def render(self, mode='human'):
        for i, stock in enumerate(self.stocks):
            print(f'Stock: {stock}, Step: {self.current_step}, Balance: {self.balance[i]}, Holdings: {self.holdings[i]}')

    def _get_obs(self):
        obs = []
        for i, stock in enumerate(self.stocks):
            obs.extend([
                self.df[f'{stock}_Close'].iloc[self.current_step],
                self.df[f'{stock}_Low'].iloc[self.current_step],
                self.df[f'{stock}_Volume'].iloc[self.current_step],
                self.df[f'{stock}_SMA'].iloc[self.current_step],
                self.df[f'{stock}_RSI'].iloc[self.current_step],
                self.df[f'{stock}_OBV'].iloc[self.current_step],
                self.holdings[i]
            ])

        return np.array(obs, dtype=np.float32)

class EnsembleAgent:
    def __init__(self, a2c_model, ppo_model):
        self.a2c_model = a2c_model
        self.ppo_model = ppo_model

    def predict(self, obs):
        a2c_action, _ = self.a2c_model.predict(obs)
        ppo_action, _ = self.ppo_model.predict(obs)

        # Averaging the actions for each stock
        ensemble_action = (a2c_action + ppo_action) / 2.0

        return ensemble_action

# Define the stocks
stocks = ['AAPL', 'AMZN', 'MSFT']

# Split the data
train_size = int(len(df) * 0.8)  # 80% for training
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Initialize the environments
train_env = MultiStockTradingEnv(train_df, stocks)
test_env = MultiStockTradingEnv(test_df, stocks)

# Define the models
models = [A2C, PPO]
model_names = ['A2C', 'PPO']

agents = {}  # Store the trained agents
results = {}

# Train the models and evaluate them
for model, model_name in zip(models, model_names):
    print(f'Training {model_name}...')
    agent = model('MlpPolicy', train_env, verbose=1)
    agent.learn(total_timesteps=int(1e5))
    agents[model_name] = agent  # Store the agent

    # Evaluate the model and store the results
    print(f'Evaluating {model_name}...')
    obs = test_env.reset()
    balances, profits, actions = [], [], []
    done = False
    while not done:
        action, _ = agent.predict(obs)
        obs, reward, done, info = test_env.step(action)
        balances.append(test_env.balance.copy())
        profits.append([profit[-1] for profit in test_env.profits])  # Store the most recent profit for each stock
        actions.append(action.copy())

        print(f'Observation: {obs}, Action: {action}, Reward: {reward}, Done: {done}, Info: {info}')

    # Store the results
    results[model_name] = {
        'balances': balances,
        'profits': profits,
        'actions': actions
    }

# Evaluate the Ensemble Agent
ensemble_agent = EnsembleAgent(agents['A2C'], agents['PPO'])
print("Evaluating Ensemble Agent...")
obs = test_env.reset()
balances, profits, actions = [], [], []
done = False
while not done:
    action = ensemble_agent.predict(obs)
    print('ensemble: ', action)
    obs, reward, done, info = test_env.step(action)
    balances.append(test_env.balance.copy())
    profits.append([profit[-1] for profit in test_env.profits])
    actions.append(action.copy())

    print(f'Observation: {obs}, Action: {action}, Reward: {reward}, Done: {done}, Info: {info}')

results['Ensemble'] = {
    'balances': balances,
    'profits': profits,
    'actions': actions
}

# Visualisations
print(results)

# Plot the balance on test data
for j, stock in enumerate(stocks):
    plt.figure(figsize=(10, 6))
    for model_name, data in results.items():
        plt.plot([balance[j] for balance in data['balances']], label=model_name)
    plt.title(f'Agent Balance over time on test data for {stock}')
    plt.xlabel('Timesteps')
    plt.ylabel('Balance')
    plt.legend()
    plt.show()

# Plot the profit on test data
for j, stock in enumerate(stocks):
    plt.figure(figsize=(10, 6))
    for model_name, data in results.items():
        plt.plot([profit[j] for profit in data['profits']], label=model_name)
    plt.title(f'Agent Profit over time on test data for {stock}')
    plt.xlabel('Timesteps')
    plt.ylabel('Profit')
    plt.legend()
    plt.show()

# Extract closing prices from the dataframe for visualization
for model_name, data in results.items():
    for j, stock in enumerate(stocks):
        closing_prices = test_df[f'{stock}_Close'].values[:len(data['actions'])]  # adjust to ensure length matches

        # Plot the closing prices on test data with markers for actions of the current model
        plt.figure(figsize=(10, 6))
        plt.plot(closing_prices)

        actions = data['actions']
        buy_points = [i for i, action in enumerate(actions) if action[j] > 0]
        sell_points = [i for i, action in enumerate(actions) if action[j] < 0]
        hold_points = [i for i, action in enumerate(actions) if action[j] == 0]

        plt.scatter(buy_points, [closing_prices[i] for i in buy_points], label=f'{model_name} Buy', marker='^')
        plt.scatter(sell_points, [closing_prices[i] for i in sell_points], label=f'{model_name} Sell', marker='v')
        plt.scatter(hold_points, [closing_prices[i] for i in hold_points], label=f'{model_name} Hold', marker='o')

        plt.title(f'Agent Trading Decisions for {stock} using {model_name}')
        plt.xlabel('Timesteps')
        plt.ylabel('Closing price')
        plt.legend()
        plt.show()

print("Actions")
print('PPO: ', actions_dict['PPO'])
print('A2C: ', actions_dict['A2C'])
print('DQN: ', actions_dict['DQN'])

print("\nPrices")
print('PPO: ', prices_dict['PPO'])
print('A2C: ', prices_dict['A2C'])
print('DQN: ', prices_dict['DQN'])

ppo_actions = actions_dict['PPO']
ppo_actions = [int(action[0]) for action in ppo_actions]
ppo_prices = prices_dict['PPO']

a2c_actions = actions_dict['A2C']
a2c_actions = [int(action[0]) for action in a2c_actions]
a2c_prices = prices_dict['A2C']

dqn_actions = actions_dict['DQN']
dqn_actions = [int(action[0]) for action in dqn_actions]
dqn_prices = prices_dict['DQN']

# Prepare the returns data for each model
ppo_returns = pd.Series(ppo_prices).pct_change().fillna(0)
a2c_returns = pd.Series(a2c_prices).pct_change().fillna(0)
dqn_returns = pd.Series(dqn_prices).pct_change().fillna(0)

# Create the returns DataFrame with a column for each model
returns_df = pd.DataFrame({
    'PPO': ppo_returns,
    'A2C': a2c_returns,
    'DQN': dqn_returns
})

print("Returns")
print(returns_df)
print()

# Generate pyfolio performance analysis for each model
for column in returns_df.columns:
    stats = pf.timeseries.perf_stats(returns_df[column])
    print(f"Performance statistics for {column}:")
    print(stats)
    print()

