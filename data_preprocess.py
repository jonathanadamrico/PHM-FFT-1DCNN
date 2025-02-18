#data_preprocess.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import seaborn as sns
import os
import sys
import gc
import pickle

from utils.functions import *

TRAIN_DIR = "Preliminary stage/Data_Pre Stage/Data_Pre Stage/Data_Pre Stage/Training data"
FINAL_TRAIN_DIR = "Data_Final_Stage/Data_Final_Stage/Training"

fs = 64000

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
cv_folds = pd.read_csv("Folder_5fold_CV/CV_5fold.csv")

# Create new directory for preprocessing
os.makedirs("Preliminary state/Data_Pre Stage/Data_Pre Stage/FFT", exist_ok=True)
os.makedirs("Preliminary state/Data_Pre Stage/Data_Pre Stage/Training data", exist_ok=True)
os.makedirs("Data_Final_Stage/FFT", exist_ok=True)
os.makedirs("Data_Final_Stage/FFT/Training", exist_ok=True)

FFT_TRAIN_DIR = "Preliminary state/Data_Pre Stage/Data_Pre Stage/Training data"
FFT_TRAIN_FINAL_DIR = "Data_Final_Stage/FFT/Training"

# Preprocess the prelim train save to FFT folder
# The preprocessed data is in numpy array (samples, features, datapoints)
for n_fault_type in tqdm(range(0,17)):
    fault_type = f"TYPE{n_fault_type}"
    os.makedirs(f"{FFT_TRAIN_DIR}/{fault_type}", exist_ok=True)
    for n_sample in range(1,4):
        sample_n = f"Sample{n_sample}"
        df = pd.DataFrame()
        for component in components:
            data_df = pd.read_csv(f"{TRAIN_DIR}/{fault_type}/{sample_n}/data_{component}.csv")
            # Calculate FFT
            data_df, _ = fft_preprocessing(data_df, fs=64000)
            df = pd.concat([df, data_df], axis=1)

        
        n_samples = len(df) // (fs // 2)
        n_features = len(df.columns)
        df = reduce_mem_usage(df, verbose=False)
        data_arr = df_to_numpy(df, n_samples, n_features, fs//2)
        np.save(f"{FFT_TRAIN_DIR}/{fault_type}/{sample_n}.npy", data_arr)


# Preprocess the final train save to FFT folder
# The preprocessed data is in numpy array (samples, features, datapoints)
for folder_name in tqdm(final_train_labels):
    total_samples = len(os.listdir(os.path.join(FINAL_TRAIN_DIR,folder_name)))
    os.makedirs(f"{FFT_TRAIN_FINAL_DIR}/{folder_name}", exist_ok=True)
    for n_sample in range(1,total_samples+1):
        sample_n = f"Sample_{n_sample}"    
    
        df = pd.DataFrame()
        for component in components:
            data_df = pd.read_csv(f"{FINAL_TRAIN_DIR}/{folder_name}/{sample_n}/data_{component}.csv")
            # Calculate FFT
            data_df, _ = fft_preprocessing(data_df, fs=64000)
            df = pd.concat([df, data_df], axis=1)

        df = reduce_mem_usage(df, verbose=False)
        n_samples = len(df) // (fs // 2)
        n_features = len(df.columns)
        df = reduce_mem_usage(df, verbose=False)
        data_arr = df_to_numpy(df, n_samples, n_features, fs//2)
        np.save(f"{FFT_TRAIN_FINAL_DIR}/{folder_name}/{sample_n}.npy", data_arr)

