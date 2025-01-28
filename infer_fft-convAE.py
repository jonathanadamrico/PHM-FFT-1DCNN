
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import sys
import pickle

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
n_features = 21
n_datapoints = fs
batch_size = 32

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

# Data scaling
'''scalers = []
for n_scaler in range(n_features):
    scaler = pickle.load(open(f'saved_scalers/scaler_ae_{anomaly_type}_{n_scaler}', 'rb'))
    scalers.append(scaler)'''

X_train = np.load(f"Quick imports/X_train_raw_{anomaly_type}.npy")
train_speed_wc = get_speed_wc_arr(X_train)
condition_means, condition_stds = compute_condition_stats(X_train, train_speed_wc)
scalers = np.load(f'saved_scalers/scaler_ae_{anomaly_type}.npy')

#======================#
#
#    TEST EVALUATION
#
#======================#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 32
n_timesteps = n_datapoints // 2
flatten_size = latent_dim * n_timesteps

autoencoder = ConvAutoencoder(n_features, n_timesteps, latent_dim).to(device)
autoencoder.load_state_dict(torch.load('saved_models'+'/'+f'autoencoder_{anomaly_type}.pth', weights_only=False, map_location=device))

classifier = Classifier(latent_dim=flatten_size).to(device)
classifier.load_state_dict(torch.load('saved_models'+'/'+f'classifier_{anomaly_type}.pth', map_location=device, weights_only=False))

predictions_prelim = []
for n_sample in range(1,103):
    sample_n = f"Sample{n_sample}"

    if quick_load_holdout:
        X_test = np.load(f"Quick imports/X_holdout_fft_{sample_n}.npy")
    else:
        X_test = combine_test_raw_data(components, anomaly_type, sample_n, fs, final_stage=False)
        speed_wc = get_speed_wc_arr(X_test)
        X_test = normalize_data(X_test, speed_wc, condition_means, condition_stds)
        X_test = calc_fft(X_test)
        np.save(f"Quick imports/X_holdout_fft_{sample_n}.npy", X_test)

    for n_scaler in range(X_test.shape[1]):
        min_val, max_val = scalers[n_scaler]
        X_test[:,n_scaler,:] = (X_test[:,n_scaler,:] - min_val) / (max_val - min_val)
    
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Evaluation
    autoencoder.eval()
    classifier.eval()
    with torch.no_grad():

        # Forward pass
        decoder_output, latent = autoencoder(X_test)

        # flatten the latent then input to the classifier
        flattened_latent = latent.view(latent.size(0), -1) 
        preds = classifier(flattened_latent).cpu().numpy()

    predictions_prelim.append(preds[0][0])

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
        X_test1 = combine_test_raw_data(components, anomaly_type, sample_n, fs, final_stage=True)

        speed_wc = get_speed_wc_arr(X_test1)
        X_test1 = normalize_data(X_test1, speed_wc, condition_means, condition_stds)
        X_test1 = calc_fft(X_test1)
        np.save(f"Quick imports/X_test1_fft_{sample_n}.npy", X_test1)

    for n_scaler in range(X_test1.shape[1]):
        min_val, max_val = scalers[n_scaler]
        X_test1[:,n_scaler,:] = (X_test1[:,n_scaler,:] - min_val) / (max_val - min_val)
    
    X_test1 = torch.tensor(X_test1, dtype=torch.float32).to(device)

    # Evaluation
    with torch.no_grad():
        # Forward pass
        decoder_output, latent = autoencoder(X_test)

        # flatten the latent then input to the classifier
        flattened_latent = latent.view(latent.size(0), -1) 
        preds = classifier(flattened_latent).cpu().numpy()
        
    samples.append(sample_n)
    predictions_final.append(preds)

final_df = pd.DataFrame({
                    "sample":samples,
                    f"{type_to_fault[anomaly_type]}":predictions_final,
                   })

final_df.to_csv(f"preds_1dcnn_ae_final/preds_{type_to_fault[anomaly_type]}_1dcnn.csv", index=False)
