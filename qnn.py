import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader


class QuantumLayer(nn.Module):
    """
    A placeholder to mimic a quantum circuit layer.

    In this simplified example, we use a parameterized linear transformation followed by a non-linear activation.
    Replace this layer with an actual quantum simulation using frameworks such as Pennylane or TorchQuantum for a true quantum computation.
    """
    def __init__(self, input_dim, output_dim):
        super(QuantumLayer, self).__init__()
        # These parameters could represent the variational parameters of a quantum circuit.
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
        self.bias = nn.Parameter(torch.randn(output_dim) * 0.1)

    def forward(self, x):
        # A linear transformation followed by a non-linear activation simulates quantum circuit processing.
        x = torch.matmul(x, self.weight) + self.bias
        x = torch.sigmoid(x)  # Activation can mimic the probabilistic nature of quantum measurements
        return x

#qnn
class QuantumNeuralNetwork(nn.Module):
    def __init__(self, input_shape, q_output_dim=128):
        """
        input_shape: tuple (time_window, num_features)
        q_output_dim: dimension of the output from the quantum layer
        """
        super(QuantumNeuralNetwork, self).__init__()
        # Compute flattened input dimension (e.g., time_window * features)
        input_dim = np.prod(input_shape)

        # Classical feature extraction layers
        self.fc1 = nn.Linear(input_dim, 256)

        # The quantum layer
        self.q_layer = QuantumLayer(256, q_output_dim)

        # Final fully-connected layer to predict outputs for each of 5 horizons and 3 classes per horizon
        # 5 horizons * 3 classes = output dimension of 15
        self.fc2 = nn.Linear(q_output_dim, 5 * 3)

    def forward(self, x):
        # x shape: (batch_size, time_window, num_features)
        # Flatten the input features
        x = x.view(x.size(0), -1)
        # Extract features using a classical dense layer
        x = torch.relu(self.fc1(x))
        # Process features with the quantum layer
        x = self.q_layer(x)
        # Map to predictions for 5 horizons (each with 3 class logits)
        logits = self.fc2(x)
        # Reshape to (batch_size, 5, 3)
        logits = logits.view(-1, 5, 3)
        return logits

###############################################
# 3. Dataset Class for FI-2010 (Placeholder)
###############################################
class FIDataset(Dataset):
    def __init__(self, data_path, split='train'):
        """
        Initialize the dataset.

        For the purpose of the template, this dummy implementation creates random data.
        Replace this with the actual dataset loading and preprocessing code.

        data_path: Path to the FI-2010 data directory.
        split: 'train' or 'test'
        """
        np.random.seed(0)
        # For example purposes: generate 100 samples.
        # Assume each sample is a time window of 10 time steps with 40 features
        # (e.g., 10 levels * 4 features per level [P_ask, V_ask, P_bid, V_bid]).
        self.samples = np.random.randn(100, 10, 40).astype(np.float32)

        # Labels: for 5 horizons, random integers in {0, 1, 2}. In practice, compute based on your labeling strategy.
        self.labels = np.random.randint(0, 3, (100, 5)).astype(np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return sample and label as torch tensors
        sample = self.samples[idx]
        label = self.labels[idx]
        return torch.tensor(sample), torch.tensor(label)

###############################################
# 4. Training and Prediction Functions
###############################################
def train_model(model, dataloader, num_epochs=10, lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            # Forward pass: model outputs shape (batch_size, 5, 3)
            outputs = model(inputs)
            loss = 0.0
            # Compute loss for each of the 5 horizons
            for horizon in range(5):
                # labels[:, horizon]: (batch_size,)
                # outputs[:, horizon, :]: (batch_size, 3)
                loss += criterion(outputs[:, horizon, :], labels[:, horizon])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    print("Training complete.")

def predict(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)  # shape: (batch_size, 5, 3)
            # For each horizon, choose the class with maximum logit
            preds = torch.argmax(outputs, dim=2)  # shape: (batch_size, 5)
            predictions.append(preds.cpu().numpy())
    # Concatenate all predictions; final shape should be (N, 5)
    predictions = np.concatenate(predictions, axis=0)
    return predictions

###############################################
# 5. Main Function: Dataset, Model, Training & Testing
###############################################
if __name__ == '__main__':
    # Define paths and hyperparameters
    data_path = 'path/to/FI-2010'  # update this with your actual data folder after downloading
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001

    # Create dataset and dataloaders (for demonstration we use dummy data)
    train_dataset = FIDataset(data_path=data_path, split='train')
    test_dataset = FIDataset(data_path=data_path, split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define input shape based on the sample design; here, (time_window, num_features)
    input_shape = (10, 40)

    # Instantiate the model
    model = QuantumNeuralNetwork(input_shape=input_shape)

    # Train the model
    train_model(model, train_loader, num_epochs=num_epochs, lr=learning_rate)

    # Generate predictions on the test dataset
    predictions = predict(model, test_loader)
    print("Predictions shape:", predictions.shape)
    # Display a toy example of predictions (for 3 samples)
    print("Sample predictions:\n", predictions[:3])

    # Save predictions as a NumPy array file, as required
    np.save("predictions.npy", predictions)
