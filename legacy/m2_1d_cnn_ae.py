import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import sys
import gc
import pickle

import tsfel

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

from utils.functions import get_features, scale_per_sensor, scale_per_sensor_2d, scale_per_sample, round_to_nearest, get_fundamental_frequency
from utils.classes import EarlyStopping, ConvNet, ConvAutoencoder

TRAIN_DIR = "Preliminary stage/Data_Pre Stage/Data_Pre Stage/Data_Pre Stage/Training data"
TEST_DIR = "Preliminary stage/Data_Pre Stage/Data_Pre Stage/Data_Pre Stage/Test data"
FINAL_TRAIN_DIR = "Data_Final_Stage/Data_Final_Stage/Training"
FINAL_TEST_DIR = "Data_Final_Stage/Data_Final_Stage/Test_Final_round8/Test"

AUG_TRAIN_DIR = "Data_Augmentation/Prelim_Train"

EXT_FEAT_DIR = "extracted_features"

load_model = False
quick_load = True
fs = 64000
n_features = 21
n_timesteps = fs // 2
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

    # For anomalous types data augmentation from prelim stage    
    '''if anomaly_type in ["TYPE12"]:
        for n_sample in tqdm(range(1,4)):
            sample_n = f"Sample{n_sample}"
            anomaly_temp = get_features(components, fault_type=fault_type, sample_n=sample_n, fs=fs, folder_name='augmentation')
            if len(anomaly_temp) == 0:
                    continue
            else:
                anomaly_arr = np.vstack([anomaly_arr, anomaly_temp])  '''

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
print(f"Train-Val shape", X_train.shape, y_train.shape, X_val.shape, y_val.shape)

# Generate a random permutation
permutation_train = np.random.permutation(len(X_train))

# Shuffle both arrays using the same permutation
X_train = X_train[permutation_train]
y_train = y_train[permutation_train]

# Remove different speed effects using autoencoder
def remove_speedfx(data_):
    data = data_.copy()
    for i in range(len(data)):
        positive_magnitude = data[i,7,:] # CH8
        fundamental_frequency = get_fundamental_frequency(positive_magnitude, fs)
        fundamental_frequency = round_to_nearest(fundamental_frequency, [20,40,60])
        data[i,:,:] = scale_per_sample(data[i,:,:])
        if fundamental_frequency == 20:
            continue
        
        # instantiate autoencoders 
        autoencoder_40Hz = ConvAutoencoder(input_channels=n_features, input_length=n_timesteps, latent_dim=256).to(device)
        autoencoder_60Hz = ConvAutoencoder(input_channels=n_features, input_length=n_timesteps, latent_dim=256).to(device)

        # load pretrained weights of the autoencoder models
        autoencoder_40Hz.load_state_dict(torch.load('saved_models'+'/'+f'autoencoder_40Hz.pth', map_location=device, weights_only=False))
        autoencoder_60Hz.load_state_dict(torch.load('saved_models'+'/'+f'autoencoder_60Hz.pth', map_location=device, weights_only=False))

        if fundamental_frequency == 40:
            data[i,:,:] = autoencoder_40Hz(torch.tensor(data[i,:,:], dtype=torch.float32).to(device)).detach().cpu().numpy()
        if fundamental_frequency == 60:
            data[i,:,:] = autoencoder_60Hz(torch.tensor(data[i,:,:], dtype=torch.float32).to(device)).detach().cpu().numpy()
    return data

plt.plot(X_train[1,0,:])
plt.savefig('X_train.png')
plt.clf()

plt.plot(X_val[1,0,:])
plt.savefig('X_val.png')
plt.clf()

#X_train = remove_speedfx(X_train)
#X_val = remove_speedfx(X_val)

# Data scaling for each sample
#X_train = scale_per_sample(X_train)
#X_val = scale_per_sample(X_val)

# Data scaling
scalers = []
for n_scaler in range(X_train.shape[1]):
    scaler = MinMaxScaler((0,1))
    X_train[:,n_scaler,:] = scaler.fit_transform(X_train[:,n_scaler,:])
    X_val[:,n_scaler,:] = scaler.transform(X_val[:,n_scaler,:])
    scalers.append(scaler)

plt.plot(X_train[1,0,:])
plt.savefig('X_train_scaled.png')
plt.clf()

plt.plot(X_val[1,0,:])
plt.savefig('X_val_scaled.png')
plt.clf()

# fit and evaluate a model using pytorch
def evaluate_model(trainX, trainy, valX, valy):
    
    print(device, torch.cuda.get_device_name(0))
    verbose, epochs, batch_size = 1, 200, 32
    n_timesteps, n_features, n_outputs = trainX.shape[2], trainX.shape[1], trainy.shape[1]
    
    trainX = torch.tensor(trainX, dtype=torch.float32).to(device)
    trainy = torch.tensor(trainy, dtype=torch.float32).to(device)
    valX = torch.tensor(valX, dtype=torch.float32).to(device)
    valy = torch.tensor(valy, dtype=torch.float32).to(device)

    # Calculate class weights
    normal_len = torch.sum(trainy == 0).item()
    anomaly_len = torch.sum(trainy == 1).item()
    print(normal_len, anomaly_len)

    weight_for_0 = torch.tensor(1.0 / normal_len, dtype=torch.float32).to(device)
    weight_for_1 = torch.tensor(1.0 / anomaly_len, dtype=torch.float32).to(device)
    class_weights = torch.tensor([weight_for_0, weight_for_1]).to(device)
    #class_weights /= class_weights.sum()

    # Define model
    model = ConvNet(n_timesteps, n_features).to(device)

    # Define optimizer and loss function
    pos_weight = class_weights[1] / class_weights[0]  # Calculate the positive weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    lr_decay_rate = 0.1
    lr_decay_steps = 10000
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    scheduler = ExponentialLR(optimizer, lr_decay_rate**(1.0 / lr_decay_steps))
    early_stopping = EarlyStopping(tolerance=20, min_delta=10)

    if load_model:
        model.load_state_dict(torch.load('saved_models'+'/'+f'1dcnn_ae_model_{anomaly_type}.pth', map_location=device))
    else:
        # Training loop
        history = {f"train":[], f"val":[]} 
        best_val_loss = 999
        best_test_z = 0
        for epoch in tqdm(range(epochs)):
            model.train()
            optimizer.zero_grad()

            # Iterate over batches
            for i in range(0, len(trainX), batch_size):
                batch_X = trainX[i:i+batch_size]
                batch_y = trainy[i:i+batch_size]

                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # Clear gradients
                optimizer.zero_grad()

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

            with torch.no_grad():
                val_outputs = model(valX[:600])
                val_loss = criterion(val_outputs, valy[:600])
            print((f"\n train loss: {loss.item():.4f} | validation loss: {val_loss.item():.4f}"))
            history[f"train"].append(loss.item())
            history[f"val"].append(val_loss.item())

            # save model
            if val_loss.item() <= best_val_loss:
                torch.save(model.state_dict(), 'saved_models'+'/'+f'1dcnn_ae_model_{anomaly_type}.pth')
                best_val_loss = val_loss.item()

        pd.DataFrame(history).to_csv(f"train_history/1dcnn_ae_{anomaly_type}.csv", index=False)

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(valX)

        threshold = 0.5
        predicted = (outputs > threshold).float()
        correct = (predicted == valy).sum().item()
        
        # Calculate val performance
        accuracy = accuracy_score(valy.cpu().numpy(), predicted.cpu().numpy())
        precision = precision_score(valy.cpu().numpy(), predicted.cpu().numpy())
        recall = recall_score(valy.cpu().numpy(), predicted.cpu().numpy())
        f1 = f1_score(valy.cpu().numpy(), predicted.cpu().numpy())
        z_metric = 0.4*accuracy + 0.2*precision + 0.2*recall + 0.2*f1
        print(f"{anomaly_type} Validation | Accuracy: {accuracy:.4} | Precision: {precision:.4} | Recall: {recall:.4} | F1: {f1:.4} | Z:{z_metric:.4}")

    return model, outputs.cpu().numpy()


model, val_pred = evaluate_model(X_train, y_train, X_val, y_val)

