#remove_speed_effects.py

import pandas as pd
import numpy as np
import sys

from utils.functions import get_features, get_new_test_features, scale_per_sensor
from utils.classes import ConvAutoencoder

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

fs = 64000 # sampling frequency
n_features = 21
n_timesteps = fs//2

TRAIN_DIR = "Preliminary stage/Data_Pre Stage/Data_Pre Stage/Data_Pre Stage/Training data"
NEW_TEST_DIR = "Prelim Test reorganized/Test reorganized"

component_channels = {
    "motor":["CH1","CH2","CH3","CH4","CH5","CH6","CH7","CH8","CH9"],
    "gearbox":["CH10","CH11","CH12","CH13","CH14","CH15"],
    "leftaxlebox":["CH16","CH17","CH18"],
    "rightaxlebox":["CH19","CH20","CH21"]
}

components = ("motor","gearbox","leftaxlebox","rightaxlebox")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train 2 autoencoders and save them
# AE_1 maps speed 40Hz to 20Hz
# AE_2 maps speed 60Hz to 20Hz
# This requires a prior knowledge on the working conditions (speed only) of the data 

# Load data
# Get normal from Test_prelim reorganized folder
# Normal data in test_prelim and in train prelim
# Gather all normal data with 20Hz
# Gather all normal data with 40Hz
# Gather all normal data with 60Hz
# Why normal data only? to separate the effects of anomalies
# Problem is, we don't know the effect of lateral load. Probably remove CH21?

# All normal samples from prelim train and test set
samples_w_20 = get_features(components, fault_type="TYPE0", sample_n="Sample1", n_datapoints=fs)
sample_temp = get_new_test_features(components, fault_type="TYPE0", sample_n="Sample1", n_datapoints=fs)
samples_w_20 = np.vstack([samples_w_20, sample_temp])  
sample_temp = get_new_test_features(components, fault_type="TYPE0", sample_n="Sample2", n_datapoints=fs)
samples_w_20 = np.vstack([samples_w_20, sample_temp])  

samples_w_40 = get_features(components, fault_type="TYPE0", sample_n="Sample2", n_datapoints=fs)
sample_temp = get_new_test_features(components, fault_type="TYPE0", sample_n="Sample3", n_datapoints=fs)
samples_w_40 = np.vstack([samples_w_40, sample_temp])  
sample_temp = get_new_test_features(components, fault_type="TYPE0", sample_n="Sample4", n_datapoints=fs)
samples_w_40 = np.vstack([samples_w_40, sample_temp])  

samples_w_60 = get_features(components, fault_type="TYPE0", sample_n="Sample3", n_datapoints=fs) 
sample_temp = get_new_test_features(components, fault_type="TYPE0", sample_n="Sample5", n_datapoints=fs)
samples_w_60 = np.vstack([samples_w_60, sample_temp])  
sample_temp = get_new_test_features(components, fault_type="TYPE0", sample_n="Sample6", n_datapoints=fs)
samples_w_60 = np.vstack([samples_w_60, sample_temp])  

# Normalize the data
samples_w_20 = scale_per_sensor(samples_w_20)
samples_w_40 = scale_per_sensor(samples_w_40)
samples_w_60 = scale_per_sensor(samples_w_60)

print(samples_w_20.shape, samples_w_40.shape, samples_w_60.shape)

# instantiate autoencoders 
autoencoder_40Hz = ConvAutoencoder(input_channels=n_features, input_length=n_timesteps, latent_dim=256).to(device)
autoencoder_60Hz = ConvAutoencoder(input_channels=n_features, input_length=n_timesteps, latent_dim=256).to(device)

criterion = nn.MSELoss()
optimizer_40 = optim.Adam(autoencoder_40Hz.parameters(), lr=1e-3)
optimizer_60 = optim.Adam(autoencoder_60Hz.parameters(), lr=1e-3)
#scheduler_40 = StepLR(optimizer_40, step_size=10, gamma=0.1)
#scheduler_60 = StepLR(optimizer_60, step_size=10, gamma=0.1)
batch_size, epochs = 32, 50

# Build autoencoder_40Hz
for epoch in range(epochs):
    autoencoder_40Hz.train()
    x = torch.tensor(samples_w_40, dtype=torch.float32).to(device)
    optimizer_40.zero_grad()
    output = autoencoder_40Hz(x)
    random_permutation = np.random.permutation(len(samples_w_20))
    desired_output = samples_w_20[random_permutation,:,:] # the target output is a sample from 20Hz
    desired_output = torch.tensor(desired_output, dtype=torch.float32).to(device)
    loss = criterion(output, desired_output)
    loss.backward()
    optimizer_40.step()
    #scheduler_40.step()

    print(f"Epoch {epoch}/{epochs} | loss: {loss.item():.4}")

# save autoencoder
torch.save(autoencoder_40Hz.state_dict(), 'saved_models'+'/'+f'autoencoder_40Hz.pth')

# Build autoencoder_60Hz
for epoch in range(epochs):
    autoencoder_60Hz.train()
    x = torch.tensor(samples_w_60, dtype=torch.float32).to(device)
    optimizer_60.zero_grad()
    output = autoencoder_60Hz(x)
    random_permutation = np.random.permutation(len(samples_w_20))
    desired_output = samples_w_20[random_permutation,:,:] # the target output is a sample from 20Hz
    desired_output = torch.tensor(desired_output, dtype=torch.float32).to(device)
    loss = criterion(output, desired_output)
    loss.backward()
    optimizer_60.step()
    #scheduler_60.step()

    print(f"Epoch {epoch}/{epochs} | loss: {loss.item():.4}")

# save autoencoder
torch.save(autoencoder_60Hz.state_dict(), 'saved_models'+'/'+f'autoencoder_60Hz.pth')


