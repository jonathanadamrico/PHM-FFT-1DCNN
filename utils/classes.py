

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

'''class Encoder(nn.Module):
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
        return x'''
    
'''class DeepEncoder(nn.Module):
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
        return x'''

'''class Autoencoder(nn.Module):
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
        return x'''

'''class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(True), 
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(True) 
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dims[0], input_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x'''

'''class ConvNet(nn.Module):
    def __init__(self, n_timesteps, n_features):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=128, kernel_size=3)
        #self.dropout1 = nn.Dropout(p=0.4)
        #self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.dropout2 = nn.Dropout(p=0.55)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        # Calculate the output size after convolutions and pooling
        self.flatten_output_size = self.calculate_flatten_output_size(n_timesteps, n_features)

        # Define the autoencoder
        hidden_dims = [32,16]
        self.autoencoder = Encoder(input_dim=self.flatten_output_size, hidden_dim=hidden_dims)

        #self.fc1 = nn.Linear(self.flatten_output_size, 16)
        self.fc1 = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        self.dropout3 = nn.Dropout(p=0.6) #
        self.fc2 = nn.Linear(hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def calculate_flatten_output_size(self, n_timesteps, n_features):
        # calculation for convolutions followed by pooling
        x = torch.randn(1, n_features, n_timesteps)  # input shape (batch_size, in_channels, sequence_length)
        x = self.conv1(x)
        x = torch.tanh(x)
        #x = self.dropout1(x)
        #x = self.maxpool(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.dropout2(x)
        x = self.maxpool(x)
        flatten_size = x.view(x.size(0), -1).size(1)
        return flatten_size

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        #x = self.dropout1(x)
        #x = self.maxpool(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.dropout2(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        # Pass through the autoencoder
        x = self.autoencoder(x)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x'''
    


'''class CNNFeatureExtractor_outdated(nn.Module):
    def __init__(self, input_channels, input_length):
        super(CNNFeatureExtractor_outdated, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.55)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) 
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.dropout(x)
        x = self.pool1(x)
        
        return x '''



class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels, input_length):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=9, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=2)
        self.bnorm1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=9, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=2)
        self.bnorm2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=9, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.bnorm3 = nn.BatchNorm1d(256)
        self.pool3 = nn.AdaptiveAvgPool1d(128)

        #self.fc1 = nn.Linear(128, 60)
        #self.tanh1 = nn.Tanh()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.bnorm1(x)

        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.bnorm2(x)

        x = self.relu3(self.conv3(x))
        x = self.bnorm3(x)
        x = self.pool3(x)

        #x = self.fc1(x)
        #x = self.tanh1(x)

        return x


'''class AutoencoderFC(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(AutoencoderFC, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)  # First fully connected layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, latent_dim)  # Latent space
        self.fc3 = nn.Linear(latent_dim, 512)  # Decoder starts here
        self.relu2 = nn.ReLU()
        self.fc4 = nn.Linear(512, input_size)  # Output layer to match original input size
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        latent = self.fc2(x)
        
        x = self.relu2(self.fc3(latent))
        x = self.fc4(x)
        
        return x, latent'''

'''class AutoencoderFC_flatten(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoencoderFC_flatten, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)  # Flatten the input to (batch_size, input_dim)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent'''

'''class CNN_Autoencoder(nn.Module):
    def __init__(self, input_channels, input_length, latent_dim):
        super(CNN_Autoencoder, self).__init__()
        
        # CNN feature extractor
        self.cnn = CNNFeatureExtractor(input_channels, input_length)
        
        # Latent dimension size after CNN processing (calculate based on pooling)
        cnn_output_size = 32 * (input_length // 8)  # (32 channels, length reduced by 8 due to 3 pooling layers)
        
        # Fully connected autoencoder
        self.autoencoder = AutoencoderFC(cnn_output_size, latent_dim)
        
    def forward(self, x):
        # Pass through CNN feature extractor
        cnn_features = self.cnn(x)
        
        # Flatten CNN features
        cnn_features = cnn_features.view(cnn_features.size(0), -1)  # Flatten to (batch_size, cnn_output_size)
        
        # Pass through fully connected layers of the autoencoder
        reconstructed = self.autoencoder(cnn_features)
        
        return reconstructed'''


# Classifier Model (Fully Connected for Normal/Anomaly Classification)
class Classifier(nn.Module):
    def __init__(self, latent_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 32),
            #nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            #nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Linear(16, 16),
            #nn.BatchNorm1d(16),
            nn.ReLU(True),
            #nn.Tanh(),
            nn.Dropout(p=0.6),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Sigmoid for binary classification
        )
    
    def forward(self, x):
        return self.fc(x)
    

'''class ConvAutoencoder_outdated(nn.Module):
    def __init__(self, input_channels, input_length, latent_dim):
        super(ConvAutoencoder_outdated, self).__init__()

        # Encoder 
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=3, stride=1, padding=1),  
            #nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, latent_dim * 2, kernel_size=3, stride=1, padding=1),  
            #nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=0.55),
            nn.MaxPool1d(kernel_size=2, stride=2), 
            #nn.Conv1d(64, latent_dim, kernel_size=3, stride=1, padding=1),  
            #nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim * 2, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            #nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, input_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # Pass through the encoder and decoder
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x, latent'''







class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels, input_length, latent_dim):
        super(ConvAutoencoder, self).__init__()

        # Encoder 
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=9, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 128, kernel_size=9, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 256, kernel_size=9, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(128)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=9, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(size=8000-9, mode='linear'),  # Adjusting to the target length
            nn.BatchNorm1d(128),

            nn.ConvTranspose1d(128, 64, kernel_size=9, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear'),  # Inverse of MaxPool
            nn.BatchNorm1d(64),

            nn.ConvTranspose1d(64, input_channels, kernel_size=9, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear'),  # Inverse of MaxPool
            nn.BatchNorm1d(input_channels),
        )        

    def forward(self, x):
        # Pass through the encoder and decoder
        latent = self.encoder(x)
        x = self.decoder(latent)

        return x, latent




'''# Define the Encoder and Decoder for the VAE
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_fc1 = nn.Linear(input_dim, 64)
        self.encoder_fc2 = nn.Linear(64, latent_dim)

        self.encoder_fc3 = nn.Linear(64, latent_dim)  # For the log-variance
        
        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, 64)
        self.decoder_fc2 = nn.Linear(64, input_dim)
        
    def encode(self, x):
        h1 = torch.relu(self.encoder_fc1(x))
        mu = self.encoder_fc2(h1)
        log_var = self.encoder_fc3(h1)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = torch.relu(self.decoder_fc1(z))
        return self.decoder_fc2(h3)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var'''
    


'''# Define the Encoder and Decoder for the VAE
class DeepVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DeepVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_fc1 = nn.Linear(input_dim, 512)
        self.encoder_fc2 = nn.Linear(512, 128)
        self.encoder_fc3 = nn.Linear(128, latent_dim)

        self.encoder_fc4 = nn.Linear(128, latent_dim)  # For the log-variance
        
        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, 128)
        self.decoder_fc2 = nn.Linear(128, 512)
        self.decoder_fc3 = nn.Linear(512, input_dim)
        
    def encode(self, x):
        h1 = torch.relu(self.encoder_fc1(x))
        h2 = torch.relu(self.encoder_fc2(h1))
        mu = self.encoder_fc3(h2)
        log_var = self.encoder_fc4(h2)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = torch.relu(self.decoder_fc1(z))
        h4 = torch.relu(self.decoder_fc2(h3))
        return self.decoder_fc3(h4)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var
'''

