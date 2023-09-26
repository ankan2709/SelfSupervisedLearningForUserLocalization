import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# Create a DataLoader for your labeled dataset

magnitude = 'data/magnitude_meanOF5.npy'
magnitude = np.load(magnitude)

labels = 'data/pos_cord.npy'
labels = np.load(labels)

# Assuming you have loaded and preprocessed your unlabeled data into 'H' and 'snr'
class SelfSupervisedDataset(Dataset):
    def __init__(self, labeled_data, positions):
        self.data = torch.tensor(labeled_data, dtype=torch.float32)
        self.pos = torch.tensor(positions, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        coords = self.pos[idx]
        return sample, coords

train_ratio = 0.9
val_ratio = 0.05
test_ratio = 0.05

# Calculate the sizes of each split
total_samples = len(labels)
train_samples = int(train_ratio * total_samples)
val_samples = int(val_ratio * total_samples)
test_samples = total_samples - train_samples - val_samples

# Use random_split to split your dataset into train, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(
    SelfSupervisedDataset(magnitude, labels),
    [train_samples, val_samples, test_samples]
)

# Create DataLoader instances for each split
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

input_channels = 1
input_height = 56
input_width = 924
output_size = 3  

# Define your CNN-based position estimation model
class PositionEstimationCNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(PositionEstimationCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (input_height // 4) * (input_width // 4), 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Create the CNN-based position estimation model
position_estimation_cnn = PositionEstimationCNN(input_channels, output_size).to(device)

# Define your loss function and optimizer for position estimation
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(position_estimation_cnn.parameters(), lr=0.001)

# Initialize variables to keep track of the best validation loss and the corresponding model weights
best_val_loss = float('inf')
best_model_weights = None

# Lists to store training and validation loss and accuracy
train_losses = []
val_losses = []

# Training loop
num_epochs = 200  # Adjust as needed
for epoch in range(num_epochs):
    position_estimation_cnn.train()  # Set the model to training mode
    total_train_loss = 0.0
    for data, labels in train_dataloader:
        optimizer.zero_grad()
        data = data.unsqueeze(1)
        data = data.to(device)
        labels = labels.to(device)
        predictions = position_estimation_cnn(data)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # Calculate average training loss for this epoch
    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}')

    # Validation loop
    position_estimation_cnn.eval()  # Set the model to evaluation mode
    total_val_loss = 0.0
    with torch.no_grad():
        for val_data, val_labels in val_dataloader:
            val_data = val_data.unsqueeze(1)
            val_data = val_data.to(device)
            val_labels = val_labels.to(device)
            val_predictions = position_estimation_cnn(val_data)
            val_loss = criterion(val_predictions, val_labels)
            total_val_loss += val_loss.item()

    # Calculate average validation loss for this epoch
    avg_val_loss = total_val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_weights = position_estimation_cnn.state_dict()
        print(f'better weights for model')

    print(f'Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {avg_val_loss:.4f}')

# Save the best model
if best_model_weights is not None:
    torch.save(best_model_weights, 'results/labeled/cnn/position_estimation_model_CNN_label_only.pth')

# Plot the training and validation loss curves
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('results/labeled/cnn/cnn_loss_curves_labelled_only_CNN.png')
plt.show()


############################ testing  ########################

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def calculate_mse(predictions, labels):
    return mean_squared_error(labels, predictions)

def calculate_mae(predictions, labels):
    return mean_absolute_error(labels, predictions)

def calculate_rmse(predictions, labels):
    return np.sqrt(mean_squared_error(labels, predictions))

def calculate_mape(predictions, labels):
    absolute_percentage_errors = np.abs((labels - predictions) / labels)
    return np.mean(absolute_percentage_errors) 

def calculate_rmspe(predictions, labels):
    percentage_errors = ((labels - predictions) / labels) ** 2
    return np.sqrt(np.mean(percentage_errors)) 



model_weights_path = "results/labeled/cnn/position_estimation_model_CNN_label_only.pth"
position_estimation_cnn.load_state_dict(torch.load(model_weights_path))

test_losses = []
mse_values = []
mae_values = []
rmse_values = []
norm_mae_values = []
norm_rmse_values = []

position_estimation_cnn.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for test_data, test_labels in test_dataloader:
        test_data = test_data.unsqueeze(1)
        test_data = test_data.to(device)
        test_labels = test_labels.to(device)
        test_predictions = position_estimation_cnn(test_data)
        test_loss = criterion(test_predictions, test_labels)
        test_losses.append(test_loss.item())

        # Convert predictions and labels back to CPU if necessary
        test_predictions = test_predictions.cpu().numpy()
        test_labels = test_labels.cpu().numpy()

        mse = calculate_mse(test_predictions, test_labels)
        mae = calculate_mae(test_predictions, test_labels)
        rmse = calculate_rmse(test_predictions, test_labels)
        norm_mae = calculate_mape(test_predictions, test_labels)
        norm_rmse = calculate_rmspe(test_predictions, test_labels)


        mse_values.append(mse)
        mae_values.append(mae)
        rmse_values.append(rmse)
        norm_mae_values.append(norm_mae)
        norm_rmse_values.append(norm_rmse)

avg_test_loss = np.mean(test_losses)
avg_mse = np.mean(mse_values)
avg_mae = np.mean(mae_values)
avg_rmse = np.mean(rmse_values)
avg_norm_mae_values = np.mean(norm_mae_values)
avg_norm_rmse_values = np.mean(norm_rmse_values)

print(f'Test Loss: {avg_test_loss:.4f}')
print(f'MSE: {avg_mse:.4f}')
print(f'MAE: {avg_mae:.4f}')
print(f'RMSE: {avg_rmse:.4f}')
print(f'MAPE: {avg_norm_mae_values:.4f}')
print(f'RMSPE: {avg_norm_rmse_values:.4f}')
