import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import matplotlib.pyplot as plt

random.seed(42)

# Data Collection
data = []
moisture = 0.7  # Starting point
trend = 1
period = 10

PLOT_MOISTURE = False

for _ in range(1000):
    data.append((moisture, None))  # Append moisture with placeholder label
    
    # Check every 5th measurement and label accordingly
    if len(data) % period == 0:
        if 0.65 <= data[-period][0] <= 0.75:
            for i in range(-period, 0):
                data[i] = (data[i][0], 1)  # label as "good"
        else:
            for i in range(-period, 0):
                data[i] = (data[i][0], 0)  # label as "bad"
    
    # Update moisture, ensuring it remains within [0, 1]
    if moisture == 1:  
        # If moisture is at max, we want to decrease it
        direction = -1  
        trend = -1
    elif moisture == 0:  
        # If moisture is at min, we want to increase it
        direction = 1  
        trend = 1
    else:
        if trend == 1:
            direction = 1 if random.uniform(0, 1) < 0.67 else -1  # 67%:33% ratio for +1:-1
        elif trend == -1:
            direction = 1 if random.uniform(0, 1) >= 0.67 else -1  # 67%:33% ratio for -1:+1
    # Update moisture, ensuring it remains within [0, 1]
    moisture = min(max(0, moisture + direction * 0.01), 1)

assert all(label is not None for _, label in data), "Unlabeled data exists"

if PLOT_MOISTURE:
    # Extract the moisture values (first column of `data`)
    moisture_values = [moisture for moisture, _ in data]

    # Create a histogram
    plt.hist(moisture_values, bins=10, edgecolor='black', alpha=0.7)

    # Add a title and labels
    plt.title("Moisture Distribution")
    plt.xlabel("Moisture")
    plt.ylabel("Frequency")

    # Show the plot
    plt.show()

# Make sure all samples are labeled


# --- Model Definition ---
class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)


# --- Data Preparation ---
X = torch.tensor([x[0] for x in data], dtype=torch.float32).view(-1, 1)
y = torch.tensor([x[1] for x in data], dtype=torch.float32).view(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = map(torch.tensor, (X_train, X_test, y_train, y_test))


# --- Model Setup ---
model = RewardModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# --- Training Loop ---
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# --- Evaluation ---
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    binary_predictions = (test_predictions > 0.5).float()

    accuracy = accuracy_score(y_test, binary_predictions)
    precision = precision_score(y_test, binary_predictions)
    recall = recall_score(y_test, binary_predictions)
    f1 = f1_score(y_test, binary_predictions)
    
    print(f"\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}")

    # Output model predictions for specific moisture values
    for moisture in [i/20 for i in range(21)]:
        moisture_tensor = torch.tensor([[moisture]], dtype=torch.float32)
        reward_pred = model(moisture_tensor).item()
        print(f"Moisture: {moisture:.2f}, Reward: {reward_pred:.4f}")
