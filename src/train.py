from self_play_game import self_play_game
from collections import deque
import multiprocessing as mp
from nn import NN
import torch
import signal

def train_model():
    training = True
    BUFFER_SIZE_THRESHOLD = 1000
    training_buffer = deque(maxlen=BUFFER_SIZE_THRESHOLD)
    network = NN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001) # the standard fr
    value_loss_fn = torch.nn.MSELoss()  # mean squared error loss 
    policy_loss_fn = torch.nn.CrossEntropyLoss()  # cross entropy loss 

    i = 0 
    while training:
        # For every one game played, we will train once. Not ideal but good for now.
        network.eval()
        with torch.no_grad():  # this disables gradient tracking
            training_data = self_play_game(network)
            training_buffer.extend(training_data)

            # Collect the data
            states, dists, turns, values = zip(*training_buffer)
            num_samples = len(states)

            states = torch.tensor(states).to(device)
            dists = torch.tensor(dists).to(device)
            turns = torch.tensor(turns).to(device)
            values = torch.tensor(values).to(device)

        if len(training_buffer) < BUFFER_SIZE_THRESHOLD: # skip training until buffer is filled
            continue

        # Train
        network.train()
        policy_losses = []
        value_losses = []
        for i in range(num_samples):
            pred_value = network.value_forward(states[i])
            pred_dist = network.selection_forward(states[i])

            value_loss = value_loss_fn(pred_value, values[i])
            policy_loss = policy_loss_fn(pred_dist, dists[i])

            loss = value_loss + policy_loss
            optimizer.zero_grad()     # set to 0
            loss.backward()        # calculate new gradients
            optimizer.step()         # step with the gradients

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        # Every 20 games/training loops, we will save the model
        if i % 1000 == 0:
            torch.save(network.state_dict(), f"models/network_{i}.pth")

    # save one last time
    return

train_model()