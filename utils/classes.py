

import torch
torch.cuda.empty_cache()
import torch.nn as nn



class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.min_val_loss = 999
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if validation_loss < self.min_val_loss:
            self.min_val_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.tolerance:  
                self.early_stop = True

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(True), 
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(True) 
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class DeepEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeepEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(True), 
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(True), 
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(True), 
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.ReLU(True) 
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim[0], input_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(True), 
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(True) 
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim[1], hidden_dim[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dim[0], input_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, n_timesteps, n_features):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.dropout1 = nn.Dropout(p=0.55)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        # Calculate the output size after convolutions and pooling
        self.flatten_output_size = self.calculate_flatten_output_size(n_timesteps, n_features)

        # Define the autoencoder
        hidden_dims = [32,16]
        self.autoencoder = Encoder(input_dim=self.flatten_output_size, hidden_dim=hidden_dims)

        #self.fc1 = nn.Linear(self.flatten_output_size, 16)
        self.fc1 = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        self.dropout2 = nn.Dropout(p=0.6) #
        self.fc2 = nn.Linear(hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def calculate_flatten_output_size(self, n_timesteps, n_features):
        # calculation for convolutions followed by pooling
        x = torch.randn(1, n_features, n_timesteps)  # input shape (batch_size, in_channels, sequence_length)
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.dropout1(x)
        x = self.maxpool(x)
        flatten_size = x.view(x.size(0), -1).size(1)
        return flatten_size

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.dropout1(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        # Pass through the autoencoder
        x = self.autoencoder(x)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    