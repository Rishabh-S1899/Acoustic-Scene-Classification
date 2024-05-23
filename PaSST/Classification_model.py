import torch
import torch.nn as nn
import torch.functional as F

class SceneClassifier(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(SceneClassifier, self).__init__()
    self.fc1 = nn.Linear(input_dim, int(input_dim/2))  # Hidden layer with half the input dimension
    self.fc2 = nn.Linear(int(input_dim/2), 128)        # New hidden layer with 128 neurons
    self.fc3 = nn.Linear(128, output_dim)           # Output layer

  def forward(self, x):
    x = F.relu(self.fc1(x))  # Apply activation function (ReLU) to first hidden layer output
    x = F.relu(self.fc2(x))  # Apply activation function (ReLU) to new hidden layer output
    x = F.log_softmax(self.fc3(x), dim=1)  # Apply log_softmax for classification
    return x