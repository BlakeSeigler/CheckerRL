import torch

class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.hidden_size = 64
        self.input_size = 200 # 10x10 board + 2 for turn and value
        self.encoder = torch.nn.Linear(self.input_size, 64)
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 8)

        self.value_head = torch.nn.Linear(8, 1)
        self.selection_head = torch.nn.Linear(8, 1)

    def value_forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.value_head(x)

    def selection_forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.selection_head(x)
