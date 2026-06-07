from self_play_game import self_play_game
from collections import deque
from nn import NN
from tqdm import tqdm
import torch
import csv
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training_log.csv')

def train_model():
    os.makedirs(MODELS_DIR, exist_ok=True)

    log_exists = os.path.exists(LOG_PATH)
    log_file = open(LOG_PATH, 'a', newline='')
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(['game', 'avg_policy_loss', 'avg_value_loss'])

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
        for j in tqdm(range(num_samples), desc=f"Training on game {i}", leave=False):
            pred_value, pred_dist = network.forward(states[j])

            value_loss = value_loss_fn(pred_value, values[j])
            policy_loss = policy_loss_fn(pred_dist, dists[j])

            loss = value_loss + policy_loss
            optimizer.zero_grad()     # set to 0
            loss.backward()        # calculate new gradients
            optimizer.step()         # step with the gradients

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        avg_policy_loss = sum(policy_losses) / len(policy_losses)
        avg_value_loss = sum(value_losses) / len(value_losses)
        log_writer.writerow([i, avg_policy_loss, avg_value_loss])
        log_file.flush()

        # Every 1000 games/training loops, we will save the model
        if i % 1000 == 0:
            torch.save(network.state_dict(), os.path.join(MODELS_DIR, f"network_{i}.pth"))

        print(f"Finished Game {i} | policy_loss={avg_policy_loss:.4f} value_loss={avg_value_loss:.4f}")
        i += 1

    # save one last time
    torch.save(network.state_dict(), os.path.join(MODELS_DIR, "network_final.pth"))
    return

if __name__ == "__main__":
    train_model()