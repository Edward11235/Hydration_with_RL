import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the neural network for the policy and value functions
class PolicyValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Directly outputting action probabilities
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        value = self.value_head(x)
        return action_probs, value

class WateringPlantEnv:
    def __init__(self):
        self.moisture_target = 0.7
        self.moisture_decay_rate = 0.01
        self.moisture_increase = 0.03
        self.moisture = self.moisture_target
        self.time_elapsed = 0
        self.history = []

    def reset(self):
        self.moisture = self.moisture_target
        self.time_elapsed = 0
        self.history = []
        return torch.tensor([self.time_elapsed], dtype=torch.float32)

    def step(self, action):
        if action == 1:
            self.moisture += self.moisture_increase
            self.time_elapsed = 0
        else:
            self.moisture -= self.moisture_decay_rate
            self.time_elapsed += 1

        self.history.append(self.moisture)
        if len(self.history) > 100:
            self.history.pop(0)
        
        reward = -10.0 * abs(self.moisture - self.moisture_target)
        done = len(self.history) == 100 and all(0.65 <= m <= 0.75 for m in self.history)
        return torch.tensor([self.time_elapsed], dtype=torch.float32), reward, done

class PPO:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3, gamma=0.99, epsilon=0.9, epsilon_decay=0.995):
        self.network = PolicyValueNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def get_action(self, state):
        action_probs, _ = self.network(state)
        if np.random.rand() > self.epsilon:
            action = torch.argmax(action_probs).item()
        else:
            action = torch.multinomial(action_probs, 1).item()
        return action

    def compute_loss(self, states, actions, rewards, next_states, old_probs):
        probs, values = self.network(states)
        _, next_values = self.network(next_states)
        td_targets = rewards + self.gamma * next_values.squeeze()
        advantages = td_targets - values.squeeze()
        new_probs = probs.gather(1, actions.unsqueeze(-1)).squeeze()
        ratio = new_probs / (old_probs + 1e-10)
        policy_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 0.8, 1.2) * advantages).mean()
        value_loss = 0.5 * (td_targets - values.squeeze()).pow(2).mean()
        return policy_loss + value_loss

    def update(self, states, actions, rewards, next_states, old_probs):
        loss = self.compute_loss(states, actions, rewards, next_states, old_probs)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        self.epsilon *= self.epsilon_decay

# Parameters
num_epochs = 5000
max_timesteps = 100
batch_size = 256

env = WateringPlantEnv()
ppo = PPO(1, 64, 2)

for epoch in range(num_epochs):
    states, actions, rewards, next_states, old_probs = [], [], [], [], []
    state = env.reset()
    for t in range(max_timesteps):
        action = ppo.get_action(state.unsqueeze(0))
        next_state, reward, done = env.step(action)
        action_probs, _ = ppo.network(state.unsqueeze(0))
        prob = action_probs[0][action].item()

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        old_probs.append(prob)

        state = next_state
        if done:
            break

    # Convert collected data into tensors
    states = torch.stack(states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    next_states = torch.stack(next_states)
    old_probs = torch.tensor(old_probs)

    # Update PPO in mini-batches
    for i in range(0, len(states), batch_size):
        ppo.update(states[i:i+batch_size], actions[i:i+batch_size], rewards[i:i+batch_size],
                   next_states[i:i+batch_size], old_probs[i:i+batch_size])

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Average Reward: {rewards.mean().item()}")

# Test the policy after training
max_test_timesteps = 10000
state = env.reset()
total_reward = 0
for t in range(max_test_timesteps):
    action = ppo.get_action(state.unsqueeze(0))
    next_state, reward, done = env.step(action)
    total_reward += reward
    state = next_state
    if done:
        print(f"Great! The model is an hydration expert now")
        break
print("Unfortunately, policy does not pass the 100 iteration hydration test after 10000 timestamp, which means the policy does not grasp how to water a plant.")