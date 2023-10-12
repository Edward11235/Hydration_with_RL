import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pygame

# Initialize pygame
pygame.init()
window_width, window_height = 400, 600
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('Watering Visualization')
font = pygame.font.Font(None, 36)
recent_moistures = []

def draw_water_level(moisture, avg_moisture):
    screen.fill((255, 255, 255))
    water_height = int(avg_moisture * window_height)
    pygame.draw.rect(screen, (0, 0, 255), (0, window_height - water_height, window_width, water_height))
    
    # Display average moisture in the upper right corner
    moisture_text = font.render(f"Moisture: {avg_moisture:.2f}", True, (0, 0, 0))
    screen.blit(moisture_text, (window_width - moisture_text.get_width() - 10, 10))
    
    pygame.display.flip()

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

    def compute_loss(self, states, actions, rewards, next_states=None, old_probs=None):
        probs, values = self.network(states)
        if next_states is not None:
            _, next_values = self.network(next_states)
        else:
            next_values = values

        # Calculate advantages
        td_targets = rewards + self.gamma * next_values.squeeze()
        advantages = td_targets - values.squeeze()

        if old_probs is not None:
            new_probs = probs.gather(1, actions.unsqueeze(-1)).squeeze()
            ratio = new_probs / old_probs
        else:
            ratio = torch.ones_like(rewards)
        policy_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 0.8, 1.2) * advantages).mean()
        value_loss = 0.5 * (td_targets - values.squeeze()).pow(2).mean()
        return policy_loss + value_loss

    def update(self, states, actions, rewards, next_states=None, old_probs=None):
        if next_states is None or old_probs is None:  # Feedback training without next_states and old_probs
            self.optimizer.zero_grad()
            value_loss = ((rewards - self.network(states)[1].squeeze())**2).mean()
            value_loss.backward()
            self.optimizer.step()
            return

# Modify the training loop to incorporate human feedback and pygame visualization
num_epochs = 5000
max_timesteps = 100
batch_size = 256

env = WateringPlantEnv()
ppo = PPO(1, 64, 2)

feedback_storage = {'states': [], 'actions': [], 'rewards': []}
last_feedback_iteration = 0

for epoch in range(num_epochs):
    state = env.reset()
    consecutive_target_count = 0

    for t in range(max_timesteps):
        action = ppo.get_action(state.unsqueeze(0))
        next_state, reward, done = env.step(action)
        
        # Update recent moisture values and compute average
        recent_moistures.append(env.moisture)
        if len(recent_moistures) > 10:
            recent_moistures.pop(0)
        avg_moisture = sum(recent_moistures) / len(recent_moistures)
        
        # Draw averaged moisture level using pygame
        draw_water_level(env.moisture, avg_moisture)
        
        pygame.time.wait(100)  # Introduces a delay of 100 milliseconds (0.1 seconds)
        
        # Check for target moisture range and update counter
        if 0.65 <= env.moisture <= 0.75:
            consecutive_target_count += 1
            if consecutive_target_count >= 500:
                print("Now the policy learns to water plant")
                pygame.quit()
                quit()
        else:
            consecutive_target_count = 0
        
        # Store actions, states, rewards for potential future feedback
        feedback_storage['states'].append(state)
        feedback_storage['actions'].append(action)
        feedback_storage['rewards'].append(reward)

        state = next_state
        should_update = False
        
        # Handle pygame events (like keypresses and quitting)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    feedback_storage['rewards'] = [-r for r in feedback_storage['rewards']]
                    should_update = True
                elif event.key == pygame.K_RETURN:
                    should_update = True

            if should_update:
                # Use feedback to train PPO
                states_tensor = torch.stack(feedback_storage['states'])
                actions_tensor = torch.tensor(feedback_storage['actions'])
                rewards_tensor = torch.tensor(feedback_storage['rewards'])
                for i in range(0, len(states_tensor), batch_size):
                    ppo.update(states_tensor[i:i+batch_size], actions_tensor[i:i+batch_size], rewards_tensor[i:i+batch_size], None, None)
                feedback_storage = {'states': [], 'actions': [], 'rewards': []}
                should_update = False

        if done:
            break

print("Training complete!")