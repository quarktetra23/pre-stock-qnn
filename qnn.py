import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class QuantumLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(QuantumLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
        self.bias = nn.Parameter(torch.randn(output_dim) * 0.1)

    def forward(self, x):
        x = torch.matmul(x, self.weight) + self.bias
        x = torch.sigmoid(x)
        return x

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, input_dim, q_output_dim=128):

        super(QuantumNeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)

        self.q_layer = QuantumLayer(256, q_output_dim)

        self.fc2 = nn.Linear(q_output_dim, 15)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.q_layer(x)
        logits = self.fc2(x)
        logits = logits.view(-1, 5, 3)
        return logits

class BenchmarkDataset(Dataset):
    def __init__(self, txt_path):

        if not os.path.isfile(txt_path):
            raise FileNotFoundError(f"File not found: {txt_path}")


        raw_data = np.loadtxt(txt_path)
        data = raw_data.T

        self.features = data[:, :144].astype(np.float32)

        labels = data[:, 144:149].astype(np.int64)
        self.labels = labels - 1

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return torch.tensor(feature), torch.tensor(label)

#train model
def train_model(model, dataloader, num_epochs=10, lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = 0.0
            #loss for each horizon
            for horizon in range(5):
                loss += criterion(outputs[:, horizon, :], labels[:, horizon])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    print("Training complete.")

def predict(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)  # (batch_size, 5, 3)
            preds = torch.argmax(outputs, dim=2)  # (batch_size, 5)
            predictions.append(preds.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    return predictions


if __name__ == '__main__':

    dataset_type = "Auction"          # or "NoAuction"
    normalization  = "Auction_Zscore"
    fold_number    = "1"              # Use fold numbers from 1 to 9.


    train_file = "BenchmarkDatasets/Auction/1.Auction_Zscore/Auction_Zscore_Training/Test_Dst_Auction_ZScore_CF_1.txt"
    test_file  = "BenchmarkDatasets/Auction/1.Auction_Zscore/Auction_Zscore_Testing/Train_Dst_Auction_ZScore_CF_1.txt"

    train_dataset = BenchmarkDataset(train_file)
    test_dataset  = BenchmarkDataset(test_file)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = QuantumNeuralNetwork(input_dim=144)

    num_epochs = 10
    learning_rate = 0.001
    train_model(model, train_loader, num_epochs=num_epochs, lr=learning_rate)

    predictions = predict(model, test_loader)
    print("Predictions shape:", predictions.shape)
    print("Sample predictions:\n", predictions[:3])

    np.save("predictions.npy", predictions)
