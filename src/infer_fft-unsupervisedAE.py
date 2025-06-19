
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import sys
import pickle

from ptflops import get_model_complexity_info

import torch
torch.cuda.empty_cache()
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.functions import *
from utils.classes import ConvAutoencoder, Classifier

TRAIN_DIR = "Preliminary stage/Data_Pre Stage/Data_Pre Stage/Data_Pre Stage/Training data"
TEST_DIR = "Preliminary stage/Data_Pre Stage/Data_Pre Stage/Data_Pre Stage/Test data"
FINAL_TRAIN_DIR = "Data_Final_Stage/Data_Final_Stage/Training"
FINAL_TEST_DIR = "Data_Final_Stage/Data_Final_Stage/Test_Final_round8/Test"

anomaly_type = sys.argv[1] 
print(anomaly_type)
fs = 64000
quick_load_holdout = False
quick_load_infer = True
if anomaly_type == "TYPE1":
    quick_load_infer = False
n_features = 21
n_datapoints = fs
batch_size = 32
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

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
min_max_values = np.load(f"saved_scalers/{anomaly_type}_min_max_values.npy")

# Data scaling
'''scalers = []
for n_scaler in range(n_features):
    scaler = pickle.load(open(f'saved_scalers/scaler_ae_{anomaly_type}_{n_scaler}', 'rb'))
    scalers.append(scaler)'''

#X_train = np.load(f"Quick imports/X_train_raw_{anomaly_type}.npy")
#train_speed_wc = get_speed_wc_arr(X_train)
#condition_means, condition_stds = compute_condition_stats(X_train, train_speed_wc)
#scalers = np.load(f'saved_scalers/scaler_ae_{anomaly_type}.npy')

ae_thresholds = {
    "TYPE1":0.7,
    "TYPE2":1.0,
    "TYPE3":0.76,
    "TYPE4":0.65,
    "TYPE5":0.93,
    "TYPE6":0.99,
    "TYPE7":0.99,
    "TYPE8":0.84,
    "TYPE9":0.98,
    "TYPE10":0.37,
    "TYPE11":0.44,
    "TYPE12":0.99,
    "TYPE13":0.8,
    "TYPE14":0.91,
    "TYPE15":0.84,
    "TYPE16":0.78,
    "TYPE17":0.94,
}

#======================#
#
#    TEST EVALUATION
#
#======================#

def remove_channels(X_array):
    # Remove unnecessary channels based on physical interactions
    motor_channels = list(range(0,9))
    gearbox_channels = list(range(9,15))
    leftaxlebox_channels = list(range(15,18))
    rightaxlebox_channels = list(range(18,21))

    if anomaly_type in component_fault['gearbox']:
        return X_array  
    if anomaly_type in component_fault['motor']:
        X_array = X_array[:,motor_channels + gearbox_channels,:]
        return X_array 
    if anomaly_type in component_fault['leftaxlebox']:
        X_array = X_array[:,gearbox_channels + leftaxlebox_channels,:]
        return X_array 
    if anomaly_type in component_fault['rightaxlebox']:
        X_array = X_array[:,gearbox_channels + rightaxlebox_channels,:] 
        return X_array 

if anomaly_type in component_fault['motor']:
    n_features = 15
if anomaly_type in component_fault['leftaxlebox']:
    n_features = 9
if anomaly_type in component_fault['rightaxlebox']:
    n_features = 9


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128
n_timesteps = n_datapoints // 2

autoencoder = ConvAutoencoder(n_features, n_timesteps, latent_dim).to(device)
autoencoder.load_state_dict(torch.load('saved_models'+'/'+f'autoencoder_{anomaly_type}.pth', weights_only=False, map_location=device))
autoencoder_criterion = nn.MSELoss()

predictions_prelim = []
for n_sample in range(1,103):
    sample_n = f"Sample{n_sample}"

    if quick_load_holdout:
        X_test = np.load(f"Quick imports/X_holdout_fft_{sample_n}.npy")
    else:
        df = pd.DataFrame()
        for component in components:
            data_df = pd.read_csv(f"{TEST_DIR}/{sample_n}/data_{component}.csv")
            # Calculate FFT
            data_df, _ = fft_preprocessing(data_df, fs=64000)
            df = pd.concat([df, data_df], axis=1)
        df = reduce_mem_usage(df, verbose=False)
        X_test = df_to_numpy(df, 1, 21, fs//2)
        speed_wc = get_speed_wc_arr(X_test)
        X_test, _ = normalize_data(X_test, speed_wc, min_max_values)
        np.save(f"Quick imports/X_holdout_fft_{sample_n}.npy", X_test)

    #for n_scaler in range(X_test.shape[1]):
    #    min_val, max_val = scalers[n_scaler]
    #    X_test[:,n_scaler,:] = (X_test[:,n_scaler,:] - min_val) / (max_val - min_val)
    
    X_test = remove_channels(X_test)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Evaluation
    autoencoder.eval()
    with torch.no_grad():

        # Forward pass
        decoder_output, latent = autoencoder(X_test)
        reconstruction_errors = autoencoder_criterion(X_test, decoder_output)
        reconstruction_errors = reconstruction_errors.cpu().numpy()
        preds = (reconstruction_errors > ae_thresholds[anomaly_type]).astype(int)

    predictions_prelim.append(preds)

# Calculate test performance
true = (test_labels["FaultCode"] == anomaly_type).astype(int).to_numpy()

threshold = 0.5
pred = (np.array(predictions_prelim) > threshold)

accuracy = accuracy_score(true, pred)
precision = precision_score(true, pred)
recall = recall_score(true, pred)
f1 = f1_score(true, pred)
z_metric = 0.4*accuracy + 0.2*precision + 0.2*recall + 0.2*f1
print(f"{anomaly_type} Test | Accuracy: {accuracy:.4} | Precision: {precision:.4} | Recall: {recall:.4} | F1: {f1:.4} | Z:{z_metric:.4}")

df = pd.DataFrame({"sample":test_labels["Sample Number"],
                   f"{type_to_fault[anomaly_type]}":predictions_prelim, #[:,0]# if with scaling
                   "true label":test_labels["FaultCode"]
                   })

df.to_csv(f"preds_1dcnn/preds_{type_to_fault[anomaly_type]}_1dcnn.csv", index=False)

del X_test

#=====================##
#
#     INFERENCE
#
#======================#

# FINAL STAGE INFERENCE
samples = []
predictions_final = []
for n_sample in tqdm(range(1,len(os.listdir(FINAL_TEST_DIR))+1)):
    sample_n = f"Sample{n_sample}"

    if quick_load_infer:
        X_test1 = np.load(f"Quick imports/X_test1_fft_{sample_n}.npy")
    else:
        df = pd.DataFrame()
        for component in components:
            data_df = pd.read_csv(f"{FINAL_TEST_DIR}/{sample_n}/data_{component}.csv")
            # Calculate FFT
            data_df, _ = fft_preprocessing(data_df, fs=64000)
            df = pd.concat([df, data_df], axis=1)
        n_samples = len(df) // (fs//2)
        df = reduce_mem_usage(df, verbose=False)
        X_test1 = df_to_numpy(df, n_samples, 21, fs//2)
        speed_wc = get_speed_wc_arr(X_test1)
        X_test1, _ = normalize_data(X_test1, speed_wc, min_max_values)
        np.save(f"Quick imports/X_test1_fft_{sample_n}.npy", X_test1)

    #for n_scaler in range(X_test1.shape[1]):
    #    min_val, max_val = scalers[n_scaler]
    #    X_test1[:,n_scaler,:] = (X_test1[:,n_scaler,:] - min_val) / (max_val - min_val)
    
    X_test1 = remove_channels(X_test1)
    X_test1 = torch.tensor(X_test1, dtype=torch.float32).to(device)

    # Evaluation
    with torch.no_grad():
        # Forward pass
        decoder_output, latent = autoencoder(X_test1)
        reconstruction_errors = autoencoder_criterion(X_test1, decoder_output)
        reconstruction_errors = reconstruction_errors.cpu().numpy()
        preds = (reconstruction_errors > ae_thresholds[anomaly_type]).astype(int)
        
    samples.append(sample_n)
    predictions_final.append(preds)

final_df = pd.DataFrame({
                    "sample":samples,
                    f"{type_to_fault[anomaly_type]}":predictions_final,
                   })

final_df.to_csv(f"preds_1dcnn_ae_final/preds_{type_to_fault[anomaly_type]}_1dcnn.csv", index=False)


# Compute time complexity in FLOPS
params1, flops1 = get_model_complexity_info(autoencoder, (n_features, n_timesteps), as_strings=True, print_per_layer_stat=False)
print(f"Params: {params1}, FLOPs: {flops1}")
