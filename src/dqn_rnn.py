import torch
import torch.nn as nn
import torch.optim as optim
import gym_anytrading
import gym
import numpy as np

# Define the RNN-based Q-Network for Deep Q-Learning
class RNNQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        super(RNNQNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Define a GRU based recurrent layer
        self.rnn = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        # Define a fully connected layer for Q-value prediction
        self.fc = nn.Linear(hidden_dim, output_dim)

    # Define forward propagation
    def forward(self, x, h):
        out, h = self.rnn(x, h)
        out = self.fc(out[:, -1, :])  # Use the last timestep to predict Q-values
        return out, h

    # Helper function to initialize hidden states
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden

# Deep Q-Learning Agent definition
class DeepQLearning:
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, lr, epsilon_decay):
        # Initialize Q-Network
        self.q_network = RNNQNetwork(input_dim, output_dim, hidden_dim, n_layers)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        # Parameters for epsilon-greedy action selection
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01

    # Get action using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() < self.epsilon:  # Explore: take a random action
            return np.random.choice([0, 1])
        state = torch.FloatTensor(state).unsqueeze(0)
        hidden = self.q_network.init_hidden(1)
        # Exploit: take the action with the highest Q-value
        q_values, _ = self.q_network(state, hidden)
        action = torch.argmax(q_values).item()
        return action

    # Train the Q-Network using Bellman equation
    def update(self, state, action, reward, next_state, done):
        # Convert all to tensors
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.LongTensor([action])
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])

        # Compute current Q-value
        current_q, _ = self.q_network(state, self.q_network.init_hidden(1))
        current_q = current_q[0][action]

        # Compute next Q-value
        next_q, _ = self.q_network(next_state, self.q_network.init_hidden(1))
        max_next_q = torch.max(next_q)

        # Calculate target Q-value
        expected_q = reward + (1 - done) * 0.99 * max_next_q

        # Compute and backpropagate the loss
        loss = self.criterion(current_q, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

# Initialize environment
env = gym.make('stocks-v0', df=df, frame_bound=(5, 100), window_size=5)
agent = DeepQLearning(2, 2, 128, 2, 0.001, 0.995)

# Training loop
for episode in range(10):
    print('Episode:', episode)
    state = env.reset()
    done = False
    total_loss = 0
    step_count = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        loss = agent.update(state, action, reward, next_state, done)

        print()
        print(info)
        print(f'Episode: {episode}, Step: {step_count}, Loss: {loss}')
        print()

        state = next_state
        total_loss += loss
        step_count += 1

    avg_loss = total_loss / step_count
    print(f'Episode: {episode}, Average Loss: {avg_loss}')
    print()

    # Decay epsilon for epsilon-greedy policy
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

def evaluate(agent, env):
    total_reward = 0
    obs = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs, epsilon=0)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(info)

    return info

# Assuming the agent has been trained
eval_env = gym.make('stocks-v0', df=df, frame_bound=(90, 110), window_size=5)
info = evaluate(agent, eval_env)

print()
print(f"Total Reward: {info['total_reward']}")
print(f"Total Profit: {info['total_profit']}")

