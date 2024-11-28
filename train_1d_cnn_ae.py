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
from sklearn.preprocessing import MinMaxScaler

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR

from utils.functions import get_features, scale_per_sensor, scale_per_sample
from utils.classes import AutoencoderFC_flatten, CNNFeatureExtractor, Classifier, ConvAutoencoder

TRAIN_DIR = "Preliminary stage/Data_Pre Stage/Data_Pre Stage/Data_Pre Stage/Training data"
TEST_DIR = "Preliminary stage/Data_Pre Stage/Data_Pre Stage/Data_Pre Stage/Test data"
FINAL_TRAIN_DIR = "Data_Final_Stage/Data_Final_Stage/Training"
FINAL_TEST_DIR = "Data_Final_Stage/Data_Final_Stage/Test_Final_round8/Test"

AUG_TRAIN_DIR = "Data_Augmentation/Prelim_Train"

EXT_FEAT_DIR = "extracted_features"

load_model = False
quick_load = True
quick_load_holdout = True
quick_load_infer = True
fs = 64000
anomaly_type = sys.argv[1] 
print(anomaly_type)
SEED = 163
n_features = 21
n_timesteps = 32000

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

components = ("motor","gearbox","leftaxlebox","rightaxlebox")
final_train_labels = [item for item in os.listdir(FINAL_TRAIN_DIR) if os.path.isdir(os.path.join(FINAL_TRAIN_DIR,item))]
test_labels = pd.read_csv("Preliminary stage/Test_Labels_prelim.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#========================#
#
#      LOAD DATA
#
#========================#

if quick_load:

    X_train = np.load(f"Quick imports/X_train_ae_{anomaly_type}.npy")
    y_train = np.load(f"Quick imports/y_train_ae_{anomaly_type}.npy")
    X_val = np.load(f"Quick imports/X_val_ae_{anomaly_type}.npy")
    y_val = np.load(f"Quick imports/y_val_ae_{anomaly_type}.npy")
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
else:

    # For normal types from prelim stage
    normal_arr = None
    for n_fault_type in tqdm(range(0,17)):
        fault_type = f"TYPE{n_fault_type}"
        if fault_type == anomaly_type:
            continue
        
        for n_sample in range(1,4):
            sample_n = f"Sample{n_sample}"
            normal_temp = get_features(components, fault_type=fault_type, sample_n=sample_n, fs=fs)
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
                normal_temp = get_features(components, fault_type=fault_type, sample_n=sample_n, fs=fs, folder_name=folder_name)
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
        anomaly_temp = get_features(components, fault_type=anomaly_type, sample_n=sample_n, fs=fs)
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
                anomaly_temp = get_features(components, fault_type=fault_type, sample_n=sample_n, fs=fs, folder_name=folder_name)
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

    # Build a X_train, y_train, X_val, y_val from normal_arr and anomaly_arr
    # 80% train 20% validation
    train_0_len = int(0.8 * len(normal_arr))
    train_1_len = int(0.8 * len(anomaly_arr))
    X_train = np.vstack([normal_arr[:train_0_len], anomaly_arr[:train_1_len]])
    y_train = np.array(list([0]* train_0_len) +  list([1]* train_1_len))
    X_val = np.vstack([normal_arr[train_0_len:], anomaly_arr[train_1_len:]])
    y_val = np.array(list([0]* (len(normal_arr)-train_0_len)) +  list([1]* (len(anomaly_arr)-train_1_len)))

    y_train = y_train.reshape(-1,1)
    y_val = y_val.reshape(-1,1)

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    normal_len = normal_arr.shape[0]
    anomaly_len = anomaly_arr.shape[0]
    del normal_arr, anomaly_arr, normal_temp, anomaly_temp
    gc.collect()

    np.save(f"Quick imports/X_train_ae_{anomaly_type}.npy", X_train)
    np.save(f"Quick imports/y_train_ae_{anomaly_type}.npy", y_train)
    np.save(f"Quick imports/X_val_ae_{anomaly_type}.npy", X_val)
    np.save(f"Quick imports/y_val_ae_{anomaly_type}.npy", y_val)

    

y_train = y_train.reshape(-1,1)
y_val = y_val.reshape(-1,1)

# Generate a random permutation
permutation_train = np.random.permutation(len(X_train))

# Shuffle both arrays using the same permutation
X_train = X_train[permutation_train]
y_train = y_train[permutation_train]

anomaly_len = sum(y_train)[0]
normal_len = len(y_train) - anomaly_len
print(anomaly_len, normal_len)

# Data scaling
scalers = []
for n_scaler in range(X_train.shape[1]):
    scaler = MinMaxScaler((0,1))
    X_train[:,n_scaler,:] = scaler.fit_transform(X_train[:,n_scaler,:])
    X_val[:,n_scaler,:] = scaler.transform(X_val[:,n_scaler,:])
    scalers.append(scaler)

normal_data = X_train[y_train.flatten() == 0]
normal_val_data = X_val[y_val.flatten() == 0]
print(f"Train-Val shape", normal_data.shape, normal_val_data.shape)


'''# Convert to PyTorch tensors (add batch dimension)
normal_data_tensor = torch.tensor(normal_data)
normal_val_data_tensor = torch.tensor(normal_val_data)

# Create DataLoader for training autoencoder
train_dataset = TensorDataset(normal_data_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(normal_val_data_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Free some memory
del normal_data, normal_val_data
gc.collect()'''

# fit and evaluate a model using pytorch
def train_autoencoder_classifier(X_train, y_train, X_val, y_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, torch.cuda.get_device_name(0))
    verbose, epochs, batch_size = 1, 20, 32

    # Convert data to torch tensor
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Instantiate the model
    input_channels = n_features  # Number of channels
    input_length = n_timesteps  # Length of each signal (time steps)
    latent_dim = 32  # Size of the latent space

    # Define model
    cnn_output_size = 32 * (input_length // 1)
    autoencoder = AutoencoderFC_flatten(cnn_output_size, latent_dim=latent_dim).to(device)
    #autoencoder = ConvAutoencoder(input_channels=n_features, input_length=n_timesteps, latent_dim=latent_dim).to(device)
    cnn = CNNFeatureExtractor(input_channels, input_length).to(device)
    
    # class weights for imbalanced data
    weight_for_0 = torch.tensor(1.0 / normal_len, dtype=torch.float32).to(device)
    weight_for_1 = torch.tensor(1.0 / anomaly_len, dtype=torch.float32).to(device)
    class_weights = torch.tensor([weight_for_0, weight_for_1]).to(device)

    # Initialize Classifier model
    classifier = Classifier(latent_dim=latent_dim).to(device)

    # Loss and optimizer for the classifier
    pos_weight = class_weights[1] / class_weights[0]  # Calculate the positive weight
    classifier_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.MSELoss()
    lr_decay_rate = 0.1
    lr_decay_steps = 10000
    optimizer = optim.SGD(list(autoencoder.parameters())+list(classifier.parameters()), lr=1e-4, momentum=0.9, nesterov=True)
    scheduler = ExponentialLR(optimizer, lr_decay_rate**(1.0 / lr_decay_steps))
    lambda_reconstruction = 0.01

    if load_model:
        autoencoder.load_state_dict(torch.load('saved_models'+'/'+f'autoencoder_{anomaly_type}.pth', weights_only=False, map_location=device))
    else:

        # Training loop
        history = {f"train":[], f"val":[]} 
        best_val_loss = 999
        for epoch in range(epochs):
            cnn.train()
            autoencoder.train()
            classifier.train()
            train_loss = 0

            # Shuffle both arrays using the same permutation
            permutation_train = np.random.permutation(len(X_train))
            X_train = X_train[permutation_train]
            y_train = y_train[permutation_train]
            # Iterate over batches
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                # Pass through CNN feature extractor
                cnn_features = cnn(batch_X)
                    
                # Flatten CNN features
                inputs = cnn_features.view(cnn_features.size(0), -1)  # Flatten to (batch_size, cnn_output_size)

                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs, latent = autoencoder(inputs)
                logits = classifier(latent)

                # Compute the loss = reconstruction loss + classification loss
                loss = criterion(outputs, inputs)  + lambda_reconstruction * classifier_criterion(logits, batch_y)
                loss.backward()
                train_loss += loss
                
                # Update the weights
                optimizer.step()
                scheduler.step()

            cnn.eval()
            autoencoder.eval()

            with torch.no_grad():  # No need to track gradients during evaluation
                val_loss = 0 
                for i in range(0, len(X_val), batch_size):
                    batch_X = X_val[i:i+batch_size]
                    cnn_features = cnn(batch_X)  
                    cnn_features_flat = cnn_features.view(cnn_features.size(0), -1) 
                    val_outputs, _ = autoencoder(cnn_features_flat)
                    val_loss += criterion(val_outputs, cnn_features_flat)

            print((f"\n Epoch {epoch}/{epochs} | train loss: {train_loss.item():.4f} | validation loss: {val_loss.item():.4f}"))
            history[f"train"].append(train_loss.item())
            history[f"val"].append(val_loss.item())

            # save model
            if val_loss.item() <= best_val_loss:
                torch.save(autoencoder.state_dict(), 'saved_models'+'/'+f'autoencoder_{anomaly_type}.pth')
                best_val_loss = val_loss.item()

        pd.DataFrame(history).to_csv(f"train_history/autoencoder_{anomaly_type}.csv", index=False)


    # Get latent features from the autoencoder for normal and anomalous data
    autoencoder = AutoencoderFC_flatten(cnn_output_size, latent_dim=latent_dim).to(device)
    autoencoder.load_state_dict(torch.load('saved_models'+'/'+f'autoencoder_{anomaly_type}.pth', weights_only=False, map_location=device))
    cnn.eval()
    autoencoder.eval()  # Set to evaluation mode

    # class weights for imbalanced data
    weight_for_0 = torch.tensor(1.0 / normal_len, dtype=torch.float32).to(device)
    weight_for_1 = torch.tensor(1.0 / anomaly_len, dtype=torch.float32).to(device)
    class_weights = torch.tensor([weight_for_0, weight_for_1]).to(device)

    # Initialize Classifier model
    classifier = Classifier(latent_dim=latent_dim).to(device)

    # Loss and optimizer for the classifier
    pos_weight = class_weights[1] / class_weights[0]  # Calculate the positive weight
    classifier_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    lr_decay_rate = 0.1
    lr_decay_steps = 10000
    classifier_optimizer = optim.SGD(classifier.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
    scheduler = ExponentialLR(classifier_optimizer, lr_decay_rate**(1.0 / lr_decay_steps))

    batch_size, epochs = 32, 100

    # Train the classifier
    history = {f"train":[], f"val":[]} 
    best_val_loss = 999
    for epoch in range(epochs):
        classifier.train()
        classifier_optimizer.zero_grad()
        train_loss = 0
        
        # Shuffle both arrays using the same permutation
        permutation_train = np.random.permutation(len(X_train))
        X_train = X_train[permutation_train]
        y_train = y_train[permutation_train]
        # Iterate over batches
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
        
            cnn_features = cnn(batch_X)
            inputs = cnn_features.view(cnn_features.size(0), -1) 
            _, latent = autoencoder(inputs)
            
            classifier_optimizer.zero_grad()

            # Forward pass
            outputs = classifier(latent)

            # Compute loss
            loss = classifier_criterion(outputs, batch_y)

            # Backpropagation
            loss.backward()
            classifier_optimizer.step()
            train_loss += loss
            scheduler.step()

        # Validation loss
        classifier.eval()
        with torch.no_grad():
            val_loss = 0
            # Iterate over batches
            for i in range(0, len(X_val), batch_size):
                batch_X = X_val[i:i+batch_size]
                batch_y = y_val[i:i+batch_size]

                cnn_features = cnn(batch_X)
                inputs = cnn_features.view(cnn_features.size(0), -1) 
                _, latent = autoencoder(inputs)
                y_pred = classifier(latent)
                loss = classifier_criterion(y_pred, batch_y)
                val_loss += loss

        if epoch % 10 == 0:
            print(f'epoch {epoch}/{epochs} | train loss: {train_loss.item():.4f} | val loss: {val_loss.item():.4f}')
        history[f"train"].append(train_loss.item())
        history[f"val"].append(val_loss.item())

        # save model
        if val_loss.item() <= best_val_loss:
            torch.save(classifier.state_dict(), 'saved_models'+'/'+f'classifier_{anomaly_type}.pth')
            best_val_loss = val_loss.item()

    pd.DataFrame(history).to_csv(f"train_history/classifier_{anomaly_type}.csv", index=False)

    # Evaluate the classifier on validation data
    classifier = Classifier(latent_dim=latent_dim).to(device)
    classifier.load_state_dict(torch.load('saved_models'+'/'+f'classifier_{anomaly_type}.pth', weights_only=False, map_location=device))
    classifier.eval()
    with torch.no_grad():
        y_preds = []
        # Iterate over batches
        for i in range(0, len(X_val), batch_size):
            batch_X = X_val[i:i+batch_size]
            batch_y = y_val[i:i+batch_size]

            cnn_features = cnn(batch_X)
            inputs = cnn_features.view(cnn_features.size(0), -1) 
            _, latent = autoencoder(inputs)
            y_pred = classifier(latent).squeeze()
            y_pred = (y_pred > 0.5).float().cpu().numpy()  # Apply threshold at 0.5 for binary classification
            if inputs.cpu().numpy().shape[0] < 2:
                break
            y_preds.extend(y_pred)

        y_val = y_val.cpu().numpy()[:len(y_preds)]

        # Compute metrics
        accuracy = accuracy_score(y_val, y_preds)
        precision = precision_score(y_val, y_preds)
        recall = recall_score(y_val, y_preds)
        f1 = f1_score(y_val, y_preds)
        z_metric = 0.4*accuracy + 0.2*precision + 0.2*recall + 0.2*f1
        print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f} | Z: {z_metric:.4f}")


train_autoencoder_classifier(X_train, y_train, X_val, y_val)

