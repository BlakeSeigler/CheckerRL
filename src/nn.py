import torch
import numpy as np

class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.hidden_size = 64
        self.input_size = 64 # 8x8 board 
        self.encoder = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.value_head = torch.nn.Linear(self.hidden_size, 1)
        self.selection_head = torch.nn.Linear(self.hidden_size, 4096)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = torch.tanh(self.value_head(x))         # tanh bounds between -1 and 1
        priors = torch.softmax(self.selection_head(x))      # softmax makes these a probability distribution
        return value, priors
