from self_play_game import self_play_game
from collections import deque
import multiprocessing as mp
from nn import NN

def train_model(training_data):
    training = True
    BUFFER_SIZE_THRESHOLD = 1000
    TRAINING_STEPS_PER_BUFFER
    training_buffer = deque(maxlen=BUFFER_SIZE_THRESHOLD)
    network = NN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001) # the standard fr
    loss_fn = torch.nn.MSELoss()  # mean squared error loss -- basic fr

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

        if len(buffer) < BUFFER_SIZE_THRESHOLD: # skip training until buffer is filled
            continue

        # Train
        network.train()
        policy_losses = []
        value_losses = []
        for i in range(num_samples):
            pred_value = network.value_forward(states[i])
            pred_dist = network.selection_forward(states[i])

            value_loss = loss_fn(pred_value, values[i])
            policy_loss = loss_fn(pred_dist, dists[i])          #TODO this isn't right, this should be softmax plus some other stuff -- understand why

            loss = value_loss + policy_loss
            optimizer.zero_grad()     # set to 0
            loss.backward()        # calculate new gradients
            optimizer.step()         # step with the gradients

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        # Every 20 games/training loops, we will save the model
        if i % 20 == 0:
            ... # save the model here

        # need a way to stop training without deleteting shit.
        if ...: # or ctrl+c lmao
            break
    
    # save one last time
    return

def save_model(network, i):
    ...