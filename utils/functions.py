
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import sys
import math
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.preprocessing import LabelEncoder

from scipy.fft import fft, fftfreq

from utils.metrics import convert_type_to_fault
from utils.classes import ConvAutoencoder

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "Preliminary stage/Data_Pre Stage/Data_Pre Stage/Data_Pre Stage/Training data"
TEST_DIR = "Preliminary stage/Data_Pre Stage/Data_Pre Stage/Data_Pre Stage/Test data"
NEW_TEST_DIR = "Prelim Test reorganized/Test reorganized"
FINAL_TRAIN_DIR = "Data_Final_Stage/Data_Final_Stage/Training"
FINAL_TEST_DIR = "Data_Final_Stage/Data_Final_Stage/Test_Final_round8/Test"
AUG_TRAIN_DIR = "Data_Augmentation/Prelim_Train"
EXT_FEAT_DIR = "extracted_features"

#======================#
#
#       FUNCTIONS
#
#======================#

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# Preprocess each channel in a certain way based on EDA
def fft_preprocessing(df_, fs, test=False):
    df = df_.copy()
    out = pd.DataFrame()
    for col in df.columns:
        fft_amplitude = []
        fft_phase = []
        for i in range(int(len(df)/fs)):  
            signal = df.loc[i*fs:(i+1)*fs-1, col]
            fft_result = np.fft.rfft(signal)
            fft_abs = abs(fft_result[:len(fft_result)-1])
            fft_amplitude.extend(fft_abs)
            if test:
                break
        out[col] = fft_amplitude
    # Normalize current channels
    #vibration_cols = [c for c in df.columns if c not in ('CH7','CH8','CH9')]
    #out['CH7'] = out['CH7'] / out['CH7'].max() * out[vibration_cols].max().max()
    #out['CH8'] = out['CH8'] / out['CH8'].max() * out[vibration_cols].max().max()
    #out['CH9'] = out['CH9'] / out['CH9'].max() * out[vibration_cols].max().max()
    return out, len(fft_abs)

# Get extracted features
def get_tabular_features(fault_type, sample_n, folder_name='', test=False, final_stage=False, window=1):

    if test:
        if final_stage == False: # prelim test
            df = pd.read_csv(f"{EXT_FEAT_DIR}/extracted_features_spectral_test_normalized.csv")

            # Get all rows in the df
            df = df[df["sample"]==sample_n]
            
        else: # final test
            df = pd.read_csv(f"{EXT_FEAT_DIR}/extracted_features_spectral_test_final_normalized.csv")

            # Get all rows in the df
            df = df[df["sample"]==sample_n]
            df = df[df["window"]==window-1]

    else:
        if folder_name == '': # prelim train
            df = pd.read_csv(f"{EXT_FEAT_DIR}/extracted_features_spectral_train_normalized.csv")
            s = f'/kaggle/input/ieee-phm-data-challenge-beijing-2024/Data_Pre Stage/Data_Pre Stage/Training data/Training data/{fault_type}/{fault_type}/{sample_n}'
        else: # final train
            df = pd.read_csv(f"{EXT_FEAT_DIR}/extracted_features_spectral_train_final_normalized.csv")
            s = f'/kaggle/input/ieee-phm-data-challenge-beijing-2024/Data_Final_Stage/Data_Final_Stage/Training/{folder_name}/{sample_n}'

        # Get all rows in the df with string s in kaggle_directory column
        df = df[df["kaggle_directory"]==s]

    print(folder_name, sample_n)

    # Remove unnecessary columns such as labels, samples, kaggle_directory
    cols = [c for c in df.columns if c not in ("label","sample","kaggle_directory","window","label.1","sample.1","kaggle_directory.1","window.1")]
    df = df[cols]

    # Pad zeros to complete 16000 datapoints
    cols_to_add = 16000 - len(cols)
    for i in range(cols_to_add):
        df[f'zero_{i}'] = 0

    # Duplicate to make 32000 datapoints
    df = pd.concat([df,df], axis=1)

    # should be 320000 items
    feat_list = df.to_numpy().flatten(order='C').tolist()
    
    return feat_list

def df_to_numpy(df, num_samples, num_channels, sample_size):
    '''
    Manually split and reshape the data without mixing the columns
    '''
    data = df.to_numpy()
    reshaped_data = []
    for i in range(num_samples):
        sample = []
        for j in range(num_channels):
            # For each channel, slice the appropriate rows (sample_size datapoints)
            sample.append(data[i * sample_size: (i + 1) * sample_size, j])
        reshaped_data.append(sample)
    
    return np.array(reshaped_data)

# Get data from all features
def get_features(components, fault_type, sample_n, fs, folder_name=''):
    '''
    returns an array of shape (n_samples, n_features, n_datapoints)
    '''
    df = pd.DataFrame()
    for component in components:
        if folder_name == 'augmentation':
            data_df = pd.read_csv(f"{AUG_TRAIN_DIR}/{fault_type}/{sample_n}/data_{component}.csv")
            #data_df = data_df * -1
            data_df = pd.DataFrame(np.flip(data_df.to_numpy(), axis=0), columns=data_df.columns)
        elif folder_name != '':
            data_df = pd.read_csv(f"{FINAL_TRAIN_DIR}/{folder_name}/{sample_n}/data_{component}.csv")
        else:
            data_df = pd.read_csv(f"{TRAIN_DIR}/{fault_type}/{sample_n}/data_{component}.csv")
        df = pd.concat([df, data_df], axis=1)
    
    #if fault_type == "TYPE1":
    #    df = df[["CH7","CH8","CH9"]]
    
    df, n_timesteps  = fft_preprocessing(df, fs=fs)
    
    n_samples = int(len(df) / n_timesteps)
    n_features = len(df.columns)
    df = reduce_mem_usage(df, verbose=False)
    data_arr = df.to_numpy()
    data_arr = data_arr.reshape(n_samples, n_features, n_timesteps)
    #data_arr = df_to_numpy(df, n_samples, n_features, n_timesteps)
    return data_arr


# Get data from all test features
def get_test_features(components, sample_n, fs, anomaly_type, final_stage=False, window=1):
    '''
    returns an array of shape (n_samples, n_features, n_datapoints)
    '''
    df = pd.DataFrame()
    for component in components:
        if final_stage:
            data_df = pd.read_csv(f"{FINAL_TEST_DIR}/{sample_n}/data_{component}.csv")
        else:
            data_df = pd.read_csv(f"{TEST_DIR}/{sample_n}/data_{component}.csv")
  
        df = pd.concat([df, data_df], axis=1)
    
    df = df.loc[(window-1)*fs:(window)*fs].reset_index(drop=True)

    #if fault_type == "TYPE1":
    #    df = df[["CH7","CH8","CH9"]]

    df, n_timesteps = fft_preprocessing(df, fs=fs, test=True)

    n_samples = int(len(df) / n_timesteps)
    n_features = len(df.columns)
    df = reduce_mem_usage(df, verbose=False)
    data_arr = df.to_numpy()
    data_arr = data_arr.reshape(n_samples, n_features, n_timesteps)
    #data_arr = df_to_numpy(df, n_samples, n_features, n_timesteps)
    return data_arr

# Get data from all test features
def get_new_test_features(components, sample_n, fs, fault_type, final_stage=False, window=1):
    '''
    returns an array of shape (n_samples, n_features, n_datapoints)
    '''
    df = pd.DataFrame()
    for component in components:
        if final_stage:
            data_df = pd.read_csv(f"{FINAL_TEST_DIR}/{sample_n}/data_{component}.csv")
        else:
            data_df = pd.read_csv(f"{NEW_TEST_DIR}/{fault_type}/{sample_n}/data_{component}.csv")
  
        df = pd.concat([df, data_df], axis=1)
    
    df = df.loc[(window-1)*fs:(window)*fs].reset_index(drop=True)

    #if fault_type == "TYPE1":
    #    df = df[["CH7","CH8","CH9"]]

    df, n_timesteps = fft_preprocessing(df, fs=fs, test=True)

    n_samples = int(len(df) / n_timesteps)
    n_features = len(df.columns)
    df = reduce_mem_usage(df, verbose=False)
    data_arr = df.to_numpy()
    data_arr = data_arr.reshape(n_samples, n_features, n_timesteps)
    #data_arr = df_to_numpy(df, n_samples, n_features, n_timesteps)
    return data_arr

# Data scaling for each sensor
# Idea: each tri-axial sensor should have the same scaling regardless of speed and lateral load.
# This way, we don't need to store and retrieve scalers.
# loop through each sample i and each tri-axial sensor j to j+3 [i,j:j+3,:]
def scale_per_sensor(data_):
    data = data_.copy()
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]-2,3):
            scaler = MinMaxScaler((0,1))
            data[i,j:j+3,:] = scaler.fit_transform(data[i,j:j+3,:].T).T
    return data

def scale_per_sensor_2d(data_):
    data = data_.copy()
    for j in range(0,data.shape[0]-2,3):
        scaler = MinMaxScaler((0,1))
        data[j:j+3,:] = scaler.fit_transform(data[j:j+3,:].T).T
    return data

def scale_per_sample(data_):
    data = data_.copy()
    assert len(data.shape) in (2,3)
    if len(data.shape) == 3:
        for i in range(0,data.shape[0]):
            scaler = MinMaxScaler((0,1))
            data[i,:,:] = scaler.fit_transform(data[i,:,:].T).T
        return data
    scaler = MinMaxScaler((0,1))
    data = scaler.fit_transform(data.T).T
    return data

def round_to_nearest(num, pool):
    # Find the value in the pool that has the smallest absolute difference to 'num'
    #print("round to nearest:",round(num,4), min(pool, key=lambda x: abs(x - num)))
    return min(pool, key=lambda x: abs(x - num))

def get_fundamental_frequency(fft_signal_, sampling_rate):
    """
    Calculate the fundamental frequency of a signal using FFT.
    
    :param signal: Input fft of signal (pandas series).
    :param sampling_rate: The sampling rate of the signal (Hz).
    :return: The fundamental frequency (Hz).
    """
    fft_signal = np.array(fft_signal_)
    # Perform FFT to get frequency components
    fft_signal = np.pad(fft_signal, (0, 1), mode='constant', constant_values=0)
    signal = np.fft.irfft(fft_signal)
    N = len(signal)
    freqs = fftfreq(N, d=1/sampling_rate)[:N // 2]  # Frequency range (only positive frequencies)
    #fft_values = np.abs(fft(signal))[:N // 2]  # FFT magnitude (only positive frequencies)
    # Find the peak frequency (fundamental frequency)
    #fundamental_freq_idx = np.argmax(fft_values)  # Index of the max peak
    fundamental_freq_idx = np.argmax(fft_signal) 
    fundamental_frequency = freqs[fundamental_freq_idx]

    return fundamental_frequency

def plot_train_val_loss(dir:str, filenames: list=None, n_rows:int=3):
    '''
    Plot the training and validation loss history for each binary classifier. 

    :filenames: list of filenames of csv of training history
    :n_rows: number of rows for which to plot the graphs
    '''
    if filenames == None:
        filenames = [f"1dcnn_ae_TYPE{n}.csv" for n in range(1,18)]

    n_columns = math.ceil(len(filenames) / n_rows)

    fig, axs = plt.subplots(n_rows, n_columns, figsize=(n_columns*5, n_rows*5))

    file_n = 0
    for i in range(n_rows):
        for j in range(n_columns):
            if file_n >= len(filenames):
                plt.delaxes(axs[i,j])
                continue
            df = pd.read_csv(f'{dir}/{filenames[file_n]}')
            axs[i,j].plot(df['train'], label='train')
            axs[i,j].plot(df['val'], label='val')
            axs[i,j].set_title(f'TYPE{file_n + 1}')
            axs[i,j].set_xlabel('Epochs')
            axs[i,j].set_ylabel('Loss')

            # Add legend
            axs[i, j].legend()

            file_n += 1

    fig.tight_layout(pad=3.0)
    plt.savefig('train_val_loss.pdf', dpi=300)


def plot_confusion_matrix(y_true, y_pred, title):
    # Define fault types (17 classes)
    faults = {'M0':0,'M1':0,'M2':0,'M3':0,'M4':0,
              'G0':0,'G1':0,'G2':0,'G3':0,'G4':0,'G5':0,'G6':0,'G7':0,'G8':0,
              'LA0':0,'LA1':0,'LA2':0,'LA3':0,'LA4':0,
              'RA0':0,'RA1':0
              }
    labels = list(faults.keys())
    n_rows = 3
    n_columns = 7

    # Compute confusion matrix
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    # Plotting the confusion matrix
    #fig, axes = plt.subplots(1, len(labels), figsize=(20, 5))
    fig, axs = plt.subplots(n_rows, n_columns, figsize=(n_columns*5, n_rows*5))

    n = 0
    for i in range(n_rows):
        for j in range(n_columns):
            if n >= len(labels):
                plt.delaxes(axs[i,j])
                continue
            sns.heatmap(mcm[n], annot=True, fmt='d', cmap="Blues", ax=axs[i,j], cbar=False,
                        xticklabels=["Pred False", "Pred True"], yticklabels=["True False", "True True"])
            axs[i,j].set_title(f'Confusion Matrix for {labels[n]}')
            axs[i,j].set_xlabel('Predicted')
            axs[i,j].set_ylabel('True')
            n += 1

    plt.tight_layout(pad=3.0)
    plt.savefig('confusion_matrix.pdf', dpi=300)

def convert_label_to_preds(label):
    faults = {'M0':0,'M1':0,'M2':0,'M3':0,'M4':0,
              'G0':0,'G1':0,'G2':0,'G3':0,'G4':0,'G5':0,'G6':0,'G7':0,'G8':0,
              'LA0':0,'LA1':0,'LA2':0,'LA3':0,'LA4':0,
              'RA0':0,'RA1':0
              }
    label = label.replace('+','_')
    codes = label.split('_')
    for code in codes:
        faults[code] = 1
    preds = list(faults.values())
    return preds


def plot_confusion_matrix_from_csv_file(filename, true_col_name='true_labels', pred_col_name='label', title="Confusion Matrix", prelim=False):
    df = pd.read_csv(filename)
    labels = df[true_col_name].tolist()
    y_true = []
    for label in labels:
        if prelim:
            label = convert_type_to_fault(label)
        y_true.append(convert_label_to_preds(label))
    
    labels = df[pred_col_name].tolist()
    y_pred = []
    for label in labels:
        y_pred.append(convert_label_to_preds(label))

    plot_confusion_matrix(y_true, y_pred, title=title)



# Remove different speed effects using autoencoder
def remove_speedfx(data_, n_features, n_timesteps, fs):
    data = data_.copy()
    for i in range(len(data)):
        positive_magnitude = data[i,7,:]
        freqs = np.fft.fftfreq(fs//2, d=1/fs)
        positive_freqs = freqs[:fs // 4]
        # Find the index of the peak frequency
        peak_index = np.argmax(positive_magnitude)
        fundamental_frequency = positive_freqs[peak_index]
        if fundamental_frequency == 20:
            return data
        
        # instantiate autoencoders 
        autoencoder_40Hz = ConvAutoencoder(input_channels=n_features, input_length=n_timesteps, latent_dim=256).to(device)
        autoencoder_60Hz = ConvAutoencoder(input_channels=n_features, input_length=n_timesteps, latent_dim=256).to(device)

        # load pretrained weights of the autoencoder models
        autoencoder_40Hz.load_state_dict(torch.load('saved_models'+'/'+f'autoencoder_40Hz.pth', map_location=device))
        autoencoder_60Hz.load_state_dict(torch.load('saved_models'+'/'+f'autoencoder_60Hz.pth', map_location=device))

        if fundamental_frequency == 40:
            data[i,:,:] = autoencoder_40Hz(data[i,:,:])
        if fundamental_frequency == 60:
            data[i,:,:] = autoencoder_60Hz(data[i,:,:])
        return data