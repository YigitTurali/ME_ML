import copy
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from lstm_autoencoders import LSTMAutoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # if gpu is available use gpu


def add_more_noise(data, noise_factor=0.25):
    noise = torch.randn_like(data) * noise_factor
    return data + noise


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def train_model(model, train_dataloader, val_dataloader, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.L1Loss().to(device)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = torch.inf

    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for seq_true in pbar:
            seq_true = seq_true.to(device)
            seq_noisy = add_more_noise(seq_true).to(device)
            optimizer.zero_grad()
            seq_pred = model(seq_noisy)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dataloader:
                seq_true = seq_true.to(device)
                seq_noisy = add_more_noise(seq_true).to(device)
                seq_pred = model(seq_noisy)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')


X_train = np.load("X_train.npy")
X_train = torch.from_numpy(X_train).float()

y_train = np.load("y_train.npy")
y_train = torch.from_numpy(y_train).int()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=False)

train_dataset = TimeSeriesDataset(X_train)
del X_train, y_train
val_dataset = TimeSeriesDataset(X_val)
del X_val, y_val
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

model = LSTMAutoencoder(3, 64).to(device)
train_model(model, train_dataloader, val_dataloader, 100)


all_train_errors = []
for batch_data in train_dataloader:
    with torch.no_grad():
        outputs = model(batch_data)
        batch_errors = torch.mean((batch_data - outputs) ** 2, dim=[1, 2])
        all_train_errors.extend(batch_errors.cpu().numpy())

# Threshold for anomaly detection
threshold = np.percentile(all_train_errors, 95)

# Anomaly detection
def detect_anomaly(data_loader):
    anomalies = []
    for batch_data in data_loader:
        with torch.no_grad():
            outputs = model(batch_data)
            batch_errors = torch.mean((batch_data - outputs) ** 2, dim=[1, 2])
            batch_anomalies = batch_errors > threshold
            anomalies.extend(batch_anomalies.cpu().numpy())
    return anomalies

# Detect anomalies in test data


X_test = np.load("X_test.npy")
X_test = torch.from_numpy(X_test).float()

test_dataset = TimeSeriesDataset(X_test)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

test_anomalies = detect_anomaly(test_dataloader)

# y_test = np.load("y_test.npy")
# y_test = torch.from_numpy(y_test).int()
