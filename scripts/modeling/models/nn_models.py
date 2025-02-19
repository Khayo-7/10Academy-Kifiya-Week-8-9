import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron."""
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class CNN(nn.Module):
    """A simple CNN model for fraud detection."""
    def __init__(self, input_channels=1, output_size=1):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(input_channels, 16, kernel_size=3, stride=1)
        self.fc = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.fc(x.mean(dim=2))
        return torch.sigmoid(x)

class LSTMModel(nn.Module):
    """LSTM model for fraud detection."""
    def __init__(self, input_size, hidden_size=64, output_size=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return torch.sigmoid(self.fc(h_n[-1]))

class GRUModel(nn.Module):
    """GRU model for fraud detection."""
    def __init__(self, input_size, hidden_size=64, output_size=1, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, h_n = self.gru(x)
        return torch.sigmoid(self.fc(h_n[-1]))