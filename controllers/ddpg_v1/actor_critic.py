import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, output_dim: int):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))  # Add activation here
        x = torch.relu(self.fc2(x))      # Add activation here
        x = self.sigmoid(self.fc3(x))    # Sigmoid to restrict values between 0 and 1
        return x * 576.0                 # Scale to [0, 576]
    
    def save(self, filename):
        torch.save(self.state_dict(), filename)
        
    def load(self, filename):
        self.load_state_dict(torch.load(filename, weights_only=True))
        
class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, output_dim: int):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def save(self, filename):
        torch.save(self.state_dict(), filename)
        
    def load(self, filename):
        self.load_state_dict(torch.load(filename, weights_only=True))