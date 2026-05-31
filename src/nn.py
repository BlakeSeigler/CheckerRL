import torch

class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.hidden_size = 64
        self.input_size = 64 # 8x8 board 
        self.encoder = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.value_head = torch.nn.Linear(self.hidden_size, 1024)  # 32 x 32 board. Not perfect but good enough
        self.selection_head = torch.nn.Linear(self.hidden_size, 1024)

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
