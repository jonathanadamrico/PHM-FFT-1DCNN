import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import sys
import gc
import pickle

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils.functions import get_features, scale_per_sensor
from utils.classes import AutoencoderFC, CNNFeatureExtractor, Classifier

TRAIN_DIR = "Preliminary stage/Data_Pre Stage/Data_Pre Stage/Data_Pre Stage/Training data"
TEST_DIR = "Preliminary stage/Data_Pre Stage/Data_Pre Stage/Data_Pre Stage/Test data"
FINAL_TRAIN_DIR = "Data_Final_Stage/Data_Final_Stage/Training"
FINAL_TEST_DIR = "Data_Final_Stage/Data_Final_Stage/Test_Final_round8/Test"

AUG_TRAIN_DIR = "Data_Augmentation/Prelim_Train"

EXT_FEAT_DIR = "extracted_features"

load_model = True
quick_load = True
quick_load_holdout = True
quick_load_infer = True
fs = 64000
anomaly_type = sys.argv[1] 
print(anomaly_type)
SEED = 163

component_fault = {
"motor": ["TYPE1","TYPE2","TYPE3","TYPE4"],
"gearbox": ["TYPE5","TYPE6","TYPE7","TYPE8","TYPE9","TYPE10","TYPE11","TYPE12"],
"leftaxlebox" : ["TYPE13","TYPE14","TYPE15","TYPE16"],
"rightaxlebox" : ["TYPE17"]
}

component_channels = {
    "motor":["CH1","CH2","CH3","CH4","CH5","CH6","CH7","CH8","CH9"],
    "gearbox":["CH10","CH11","CH12","CH13","CH14","CH15"],
    "leftaxlebox":["CH16","CH17","CH18"],
    "rightaxlebox":["CH19","CH20","CH21"]
}

type_to_fault = {
    "TYPE0":"all_normal",
    "TYPE1":"M1",
    "TYPE2":"M2",
    "TYPE3":"M3",
    "TYPE4":"M4",
    "TYPE5":"G1",
    "TYPE6":"G2",
    "TYPE7":"G3",
    "TYPE8":"G4",
    "TYPE9":"G5",
    "TYPE10":"G6",
    "TYPE11":"G7",
    "TYPE12":"G8",
    "TYPE13":"LA1",
    "TYPE14":"LA2",
    "TYPE15":"LA3",
    "TYPE16":"LA4",
    "TYPE17":"RA1"
}

full_fault_to_type = {
    "M0_G0_LA0_RA0":"TYPE0",
    "M1_G0_LA0_RA0":"TYPE1",
    "M2_G0_LA0_RA0":"TYPE2",
    "M3_G0_LA0_RA0":"TYPE3",
    "M4_G0_LA0_RA0":"TYPE4",
    "M0_G1_LA0_RA0":"TYPE5",
    "M0_G2_LA0_RA0":"TYPE6",
    "M0_G3_LA0_RA0":"TYPE7",
    "M0_G4_LA0_RA0":"TYPE8",
    "M0_G5_LA0_RA0":"TYPE9",
    "M0_G6_LA0_RA0":"TYPE10",
    "M0_G7_LA0_RA0":"TYPE11",
    "M0_G8_LA0_RA0":"TYPE12",
    "M0_G0_LA1_RA0":"TYPE13",
    "M0_G0_LA2_RA0":"TYPE14",
    "M0_G0_LA3_RA0":"TYPE15",
    "M0_G0_LA4_RA0":"TYPE16",
    "M0_G0_LA0_RA1":"TYPE17"
}

baselines_faults = {
    "M0":["TYPE5","TYPE6","TYPE7","TYPE8","TYPE9","TYPE10","TYPE11","TYPE12","TYPE13","TYPE14","TYPE15","TYPE16"],
    "G0":["TYPE1","TYPE2","TYPE3","TYPE4","TYPE13","TYPE14","TYPE15","TYPE16"],
    "LA0":["TYPE1","TYPE2","TYPE3","TYPE4","TYPE5","TYPE6","TYPE7","TYPE8","TYPE9","TYPE10","TYPE11","TYPE12"],
    "RA0":["all"],
}

components = component_channels.keys()
final_train_labels = [item for item in os.listdir(FINAL_TRAIN_DIR) if os.path.isdir(os.path.join(FINAL_TRAIN_DIR,item))]
test_labels = pd.read_csv("Preliminary stage/Test_Labels_prelim.csv")



#========================#
#
#      LOAD DATA
#
#========================#

if quick_load:
    normal_data = np.load(f"Quick imports/normal_data_ae_{anomaly_type}.npy")
    normal_val_data = np.load(f"Quick imports/normal_val_data_ae_{anomaly_type}.npy")
    val_data = np.load(f"Quick imports/val_data_ae_{anomaly_type}.npy")
    print(normal_data.shape, val_data.shape)

else:

    # For normal types from prelim stage
    normal_arr = None
    for n_fault_type in tqdm(range(0,17)):
        fault_type = f"TYPE{n_fault_type}"
        if fault_type == anomaly_type:
            continue
        
        for n_sample in range(1,4):
            sample_n = f"Sample{n_sample}"
            normal_temp = get_features(components, fault_type=fault_type, sample_n=sample_n, n_datapoints=fs)
            if len(normal_temp) == 0:
                continue
            if type(normal_arr) == type(None):
                normal_arr = normal_temp.copy()
            else:
                normal_arr = np.vstack([normal_arr, normal_temp])  
            

    # For normal types from final stage
    for folder_name in tqdm(final_train_labels):
        if type_to_fault[anomaly_type] not in folder_name:
            total_samples = len(os.listdir(os.path.join(FINAL_TRAIN_DIR,folder_name)))
            for n_sample in range(1,total_samples+1):
                sample_n = f"Sample_{n_sample}"
                normal_temp = get_features(components, fault_type=fault_type, sample_n=sample_n, n_datapoints=fs, folder_name=folder_name)
                if len(normal_temp) == 0:
                        continue
                if type(normal_arr) == type(None):
                    normal_arr = normal_temp.copy()
                else:
                    normal_arr = np.vstack([normal_arr, normal_temp])
            

    # For anomalous types from prelim stage
    anomaly_arr = None
    for n_sample in tqdm(range(1,4)):
        if anomaly_type == "TYPE17":
            break
        sample_n = f"Sample{n_sample}"
        anomaly_temp = get_features(components, fault_type=anomaly_type, sample_n=sample_n, n_datapoints=fs)
        if len(anomaly_temp) == 0:
                continue
        if type(anomaly_arr) == type(None):
            anomaly_arr = anomaly_temp.copy()
        else:
            anomaly_arr = np.vstack([anomaly_arr, anomaly_temp])  

    # For anomalous types from final stage
    for folder_name in tqdm(final_train_labels):
        if type_to_fault[anomaly_type] in folder_name:
            total_samples = len(os.listdir(os.path.join(FINAL_TRAIN_DIR,folder_name)))
            for n_sample in range(1,total_samples+1):
                sample_n = f"Sample_{n_sample}"
                anomaly_temp = get_features(components, fault_type=fault_type, sample_n=sample_n, n_datapoints=fs, folder_name=folder_name)
                if len(anomaly_temp) == 0:
                        continue
                if type(anomaly_arr) == type(None):
                    anomaly_arr = anomaly_temp.copy()
                else:
                    anomaly_arr = np.vstack([anomaly_arr, anomaly_temp]) 
            
    print(normal_arr.shape)
    print(anomaly_arr.shape)

    # Only the first axis is shuffled. The contents remain the same.
    np.random.seed(SEED)
    np.random.shuffle(anomaly_arr)
    np.random.seed(SEED)
    np.random.shuffle(normal_arr)

    normal_train_len = int(0.8 * len(normal_arr))
    normal_data = normal_arr[:normal_train_len]
    normal_val_data = normal_arr[normal_train_len:]
    val_data = np.array(np.vstack([normal_val_data, anomaly_arr]))

    print(normal_data.shape, val_data.shape)

    del normal_arr, anomaly_arr, normal_temp, anomaly_temp
    gc.collect()

    np.save(f"Quick imports/normal_data_ae_{anomaly_type}.npy", normal_data)
    np.save(f"Quick imports/normal_val_data_ae_{anomaly_type}.npy", normal_val_data)
    np.save(f"Quick imports/val_data_ae_{anomaly_type}.npy", val_data)


print(f"Train-Val shape", normal_data.shape, val_data.shape)

# Data scaling for each sensor
# Idea: each tri-axial sensor should have the same scaling regardless of speed and lateral load.
# This way, we don't need to store and retrieve scalers.
# loop through each sample i and each tri-axial sensor j to j+3 [i,j:j+3,:]
normal_data = scale_per_sensor(normal_data)
normal_val_data = scale_per_sensor(normal_val_data)
val_data = scale_per_sensor(val_data)

# Convert to PyTorch tensors (add batch dimension)
normal_data_tensor = torch.tensor(normal_data)
normal_val_data_tensor = torch.tensor(normal_val_data)
val_data_tensor = torch.tensor(val_data)

# Create DataLoader for training
train_dataset = TensorDataset(normal_data_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(val_data_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# fit and evaluate a model using pytorch
def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, torch.cuda.get_device_name(0))
    verbose, epochs = 1, 100

    # Instantiate the model
    input_channels = 21  # Number of channels
    input_length = 32000  # Length of each signal (time steps)
    latent_dim = 128  # Size of the latent space

    # Define model
    cnn_output_size = 32 * (input_length // 8)
    autoencoder = AutoencoderFC(cnn_output_size, latent_dim=latent_dim).to(device)
    cnn = CNNFeatureExtractor(input_channels, input_length).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    if load_model:
        autoencoder.load_state_dict(torch.load('saved_models'+'/'+f'autoencoder_{anomaly_type}.pth', map_location=device))
    else:
        # Training loop
        history = {f"train":[], f"val":[]} 
        best_val_loss = 999
        for epoch in range(epochs):
            autoencoder.train()

            # Iterate over batches
            for batch in train_loader:
                x = batch[0].to(device)

                # Pass through CNN feature extractor
                cnn_features = cnn(x)
                    
                # Flatten CNN features
                inputs = cnn_features.view(cnn_features.size(0), -1)  # Flatten to (batch_size, cnn_output_size)

                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs, _ = autoencoder(inputs)

                # Compute the loss
                loss = criterion(outputs, inputs)  # Reconstruction error
                loss.backward()
                
                # Update the weights
                optimizer.step()

            cnn.eval()
            autoencoder.eval()

            with torch.no_grad():  # No need to track gradients during evaluation
                val_loss = 0
                for val_batch in val_loader:
                    val_inputs = val_batch[0].to(device) 
                    cnn_features = cnn(val_inputs)  # Output shape will be (batch_size, 32, 4000)
                    cnn_features_flat = cnn_features.view(cnn_features.size(0), -1)  # Flatten to (batch_size, 32 * 4000)
                    val_outputs, _ = autoencoder(cnn_features_flat)
                    val_loss += criterion(val_outputs, cnn_features_flat)

            print((f"\n Epoch {epoch}/{epochs} | train loss: {loss.item():.4f} | validation loss: {val_loss.item():.4f}"))
            history[f"train"].append(loss.item())
            history[f"val"].append(val_loss.item())

            # save model
            if val_loss.item() <= best_val_loss:
                torch.save(autoencoder.state_dict(), 'saved_models'+'/'+f'autoencoder_{anomaly_type}.pth')
                best_val_loss = val_loss.item()

        pd.DataFrame(history).to_csv(f"train_history/autoencoder_{anomaly_type}.csv", index=False)


    # Get latent features from the autoencoder for normal and anomalous data
    autoencoder.eval()  # Set to evaluation mode
    with torch.no_grad():
        cnn_features = cnn(val_data_tensor.to(device))
        cnn_features_flat = cnn_features.view(cnn_features.size(0), -1)
        _, X_latent = autoencoder(cnn_features_flat)  # Latent features for validation data

        y_latent = torch.cat((torch.zeros(len(normal_val_data)), torch.ones(len(val_data)-len(normal_val_data))), dim=0)

        X_train_latent, X_val_latent, y_train, y_val = train_test_split(X_latent, y_latent, test_size=0.2, random_state=42)

    # Initialize Classifier model
    classifier = Classifier(latent_dim=latent_dim).to(device)

    # Loss and optimizer for the classifier
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    classifier_criterion = nn.BCELoss()  # Binary Cross-Entropy loss

    # Train the classifier
    for epoch in range(50):
        classifier.train()
        classifier_optimizer.zero_grad()
        
        # Forward pass
        outputs = classifier(X_train_latent)
        
        # Compute loss
        loss = classifier_criterion(outputs.squeeze(), y_train.to(device))
        
        # Backpropagation
        loss.backward()
        classifier_optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{50}, Loss: {loss.item()}')


    # Evaluate the classifier on validation data
    classifier.eval()
    with torch.no_grad():
        y_pred = classifier(X_val_latent).squeeze()
        y_pred = (y_pred > 0.5).float().cpu().numpy()  # Apply threshold at 0.5 for binary classification
        
        # Compute metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")


evaluate_model()

