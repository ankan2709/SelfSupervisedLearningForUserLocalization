import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Set the seed for the random number generator
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load your data
H = np.load('data/unlabelled_magnitude.npy')
snr = np.load('data/unlabelled_snr.npy')

# Assuming you have loaded and preprocessed your unlabeled data into 'H' and 'snr'
class SelfSupervisedDataset(Dataset):
    def __init__(self, unlabeled_data, snr_data):
        self.data = torch.tensor(unlabeled_data, dtype=torch.float32)
        self.snr = torch.tensor(snr_data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        snr_sample = self.snr[idx]
        return sample, snr_sample

# Split your dataset into training and validation sets
total_samples = len(H)
train_samples = int(0.8 * total_samples)  # 80% for training, adjust as needed
val_samples = total_samples - train_samples



train_dataset, val_dataset = random_split(SelfSupervisedDataset(H, snr), [train_samples, val_samples])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64)

class SelfSupervisedModel(nn.Module):
    def __init__(self):
        super(SelfSupervisedModel, self).__init__()
        # Adjust the input size to match your data
        self.encoder = nn.Sequential(
            nn.Linear(56 * 924, 256),  # Adjust the input size here
            nn.ReLU(),
            nn.Linear(256, 128),  # Adjust the input size here
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 56 * 924)  # Adjust the output size here
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

self_supervised_model = SelfSupervisedModel().to(device)

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(self_supervised_model.parameters(), lr=0.001)

best_val_loss = float('inf')  # Initialize with a high value
best_model_state_dict = None

num_epochs = 100  # Adjust as needed

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    self_supervised_model.train()  # Set the model to training mode
    total_train_loss = 0.0
    for data, snr_data in train_dataloader:
        optimizer.zero_grad()
        # Flatten your data to match the input size
        inputs = data.view(data.size(0), -1).to(device)
        
        encoded, decoded = self_supervised_model(inputs)
        loss = criterion(decoded, inputs)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}')

    # Validation loop
    self_supervised_model.eval()  # Set the model to evaluation mode
    total_val_loss = 0.0
    with torch.no_grad():
        for val_data, val_snr_data in val_dataloader:
            val_inputs = val_data.view(val_data.size(0), -1).to(device)
            val_encoded, val_decoded = self_supervised_model(val_inputs)
            v_loss = criterion(val_decoded, val_inputs)
            total_val_loss += v_loss.item()

    # Calculate the average validation loss
    avg_val_loss = total_val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    # Check if the current model has a lower validation loss than the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # Save the state dictionary of the current best model
        best_model_state_dict = self_supervised_model.state_dict()
        print('saving model')

    print(f'Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {avg_val_loss:.4f}')

# After training, you can save the best model's state dictionary to a file
torch.save(best_model_state_dict, 'results/unlabeled/mlp/best_self_supervised_model.pth')


plt.figure(figsize=(12, 6))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('results/unlabeled/mlp/loss_curvesAEunlabelled.png')
plt.show()
