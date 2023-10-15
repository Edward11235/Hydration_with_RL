import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

# --- Data Collection ---
data = []
for _ in range(1000):
    moisture = random.uniform(0, 1)
    label = 1 if 0.65 <= moisture <= 0.75 else 0
    data.append((moisture, label))


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
