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

#from keras import Input
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Flatten
#from keras.layers import Dropout
#from keras.layers import Conv1D
#from keras.layers import MaxPooling1D
#from keras.utils import to_categorical

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

TRAIN_DIR = "Preliminary stage/Data_Pre Stage/Data_Pre Stage/Data_Pre Stage/Training data"
TEST_DIR = "Preliminary stage/Data_Pre Stage/Data_Pre Stage/Data_Pre Stage/Test data"
FINAL_TRAIN_DIR = "Data_Final_Stage/Data_Final_Stage/Training"
FINAL_TEST_DIR = "Data_Final_Stage/Data_Final_Stage/Test_Final_round5/Test"

load_model = True
fs = 64000
anomaly_type = sys.argv[1] #"TYPE5"
print(anomaly_type)
SEED = 123

component_fault = {
"motor": ["TYPE1","TYPE2","TYPE3","TYPE4"],
"gearbox": ["TYPE5","TYPE6","TYPE7","TYPE8","TYPE9","TYPE10","TYPE11","TYPE12"],
"leftaxlebox" : ["TYPE13","TYPE14","TYPE15","TYPE16"]
}

component_channels = {
    "motor":["CH1","CH2","CH3","CH4","CH5","CH6","CH7","CH8","CH9"],
    "gearbox":["CH10","CH11","CH12","CH13","CH14","CH15"],
    "leftaxlebox":["CH16","CH17","CH18"],
    "rightaxlebox":["CH19","CH20","CH21"]
}

type_to_fault = {
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
    "TYPE16":"LA4"
}

baselines_faults = {
    "M0":["TYPE5","TYPE6","TYPE7","TYPE8","TYPE9","TYPE10","TYPE11","TYPE12","TYPE13","TYPE14","TYPE15","TYPE16"],
    "G0":["TYPE1","TYPE2","TYPE3","TYPE4","TYPE13","TYPE14","TYPE15","TYPE16"],
    "LA0":["TYPE1","TYPE2","TYPE3","TYPE4","TYPE5","TYPE6","TYPE7","TYPE8","TYPE9","TYPE10","TYPE11","TYPE12"],
    "RA0":["all"],
}

components = component_channels.keys()
final_train_labels = [item for item in os.listdir(FINAL_TRAIN_DIR) if os.path.isdir(os.path.join(FINAL_TRAIN_DIR,item))]

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

# Filter high frequency signals 
def filter_high_freqs(x, fs, cutoff_freq=60):
    # Perform FFT
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1/fs)

    # Remove high-frequency components
    X_filtered = X.copy()
    X_filtered[np.abs(freqs) > cutoff_freq] = 0

    # Inverse FFT
    filtered_signal = np.fft.irfft(X_filtered)
    return filtered_signal

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
            #fft_phase.extend(np.unwrap(np.angle(fft_result[:-1])))
            #fft_phase.extend(np.angle(fft_result[:-1]))
            if test:
                break
        out[col] = fft_amplitude
        #out[col+'_phase'] = fft_phase
    return out, len(fft_abs)


# Get data from all features
def get_features(components, fault_type, sample_n, n_datapoints, folder_name=''):
    '''
    returns an array of shape (n_samples, n_features, n_datapoints)
    '''
    df = pd.DataFrame()
    for component in components:
        if folder_name != '':
            data_df = pd.read_csv(f"{FINAL_TRAIN_DIR}/{folder_name}/{sample_n}/data_{component}.csv")
        else:
            data_df = pd.read_csv(f"{TRAIN_DIR}/{fault_type}/{sample_n}/data_{component}.csv")
        df = pd.concat([df, data_df], axis=1)
    
    if anomaly_type == "TYPE1":
        df = df[["CH7","CH8","CH9"]]
    
    df, n_datapoints  = fft_preprocessing(df, fs=fs)
    
    n_samples = int(len(df) / n_datapoints)
    n_features = len(df.columns)
    df = reduce_mem_usage(df, verbose=False)
    data_arr = df.to_numpy()
    data_arr = data_arr.reshape(n_samples, n_features, n_datapoints)
    return data_arr


# Get data from all test features
def get_test_features(components, sample_n, n_datapoints, final_stage=False):
    '''
    returns an array of shape (n_samples, n_features, n_datapoints)
    '''
    df = pd.DataFrame()
    for component in components:
        if final_stage:
            data_df = pd.read_csv(f"{FINAL_TEST_DIR}/{sample_n}/data_{component}.csv")
        else:
            data_df = pd.read_csv(f"{TEST_DIR}/{sample_n}/data_{component}.csv")
        # Check fundamental frequency
        #if component == "motor":
            #X = data_df.loc[:, "CH9"]
            #fund_freq = tsfel.feature_extraction.features.fundamental_frequency(X, n_datapoints)
            #print(fund_freq)
            #if fund_freq < 31:
            #    return np.array([])
            #if fund_freq > 50:
            #    return np.array([])
            #data_df = data_df[["CH1","CH2","CH3","CH4","CH5","CH6"]]
        df = pd.concat([df, data_df], axis=1)
    
    if anomaly_type == "TYPE1":
        df = df[["CH7","CH8","CH9"]]

    df, n_datapoints = fft_preprocessing(df, fs=fs, test=True)

    n_samples = int(len(df) / n_datapoints)
    n_features = len(df.columns)
    df = reduce_mem_usage(df, verbose=False)
    data_arr = df.to_numpy()
    data_arr = data_arr.reshape(n_samples, n_features, n_datapoints)
    return data_arr

#========================#
#
#      LOAD DATA
#
#========================#

quick_load = True

if quick_load:
    X_train = np.load(f"Quick imports/X_train_{anomaly_type}.npy")
    y_train = np.load(f"Quick imports/y_train_{anomaly_type}.npy")
    X_val = np.load(f"Quick imports/X_val_{anomaly_type}.npy")
    y_val = np.load(f"Quick imports/y_val_{anomaly_type}.npy")
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
else:
    # For normal types
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
            #if anomaly_type=="TYPE8" and fault_type not in ("TYPE6","TYPE12"): # too many false positives of type6 and type12 for type8
            #        break 
            

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
                #if anomaly_type=="TYPE8" and fault_type not in ("TYPE6","TYPE12"): # too many false positives of type6 and type12 for type8
                #    break 
            

    # For anomalous types
    fault_type = anomaly_type
    anomaly_arr = None
    for n_sample in tqdm(range(1,4)):
        sample_n = f"Sample{n_sample}"
        anomaly_temp = get_features(components, fault_type=fault_type, sample_n=sample_n, n_datapoints=fs)
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

    # downsample normal_class
    if anomaly_arr.shape[0] < 31:
        normal_arr = normal_arr[:anomaly_arr.shape[0]*6,:,:]
        print(normal_arr.shape)
        print(anomaly_arr.shape)

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

    np.save(f"Quick imports/X_train_{anomaly_type}.npy", X_train)
    np.save(f"Quick imports/y_train_{anomaly_type}.npy", y_train)
    np.save(f"Quick imports/X_val_{anomaly_type}.npy", X_val)
    np.save(f"Quick imports/y_val_{anomaly_type}.npy", y_val)


# Generate a random permutation
permutation_train = np.random.permutation(len(X_train))
permutation_val = np.random.permutation(len(X_val))

# Shuffle both arrays using the same permutation
X_train = X_train[permutation_train]
y_train = y_train[permutation_train]
X_val = X_val[permutation_val]
y_val = y_val[permutation_val]

# Data scaling
scalers = []
for n_scaler in range(X_train.shape[1]):
    scaler = MinMaxScaler((0,1))
    X_train[:,n_scaler,:] = scaler.fit_transform(X_train[:,n_scaler,:])
    X_val[:,n_scaler,:] = scaler.transform(X_val[:,n_scaler,:])
    scalers.append(scaler)


'''# fit and evaluate a model with keras
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 1, 4, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    weight_for_0 = 1.0 / normal_len
    weight_for_1 = 1.0 / anomaly_len
    model = Sequential()
    model.add(Input(shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.7))
    #model.add(Dense(64, activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    #model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.8))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['auc'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, class_weight={0: weight_for_0, 1: weight_for_1})
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
        
    return model, accuracy'''

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.min_val_loss = 999
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        #if (validation_loss - train_loss) > self.min_delta:
        #    self.counter +=1
        if validation_loss < self.min_val_loss:
            self.min_val_loss = validation_loss
        else:
            self.counter += 1
            if self.counter >= self.tolerance:  
                self.early_stop = True

class ConvNet(nn.Module):
    def __init__(self, n_timesteps, n_features):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.dropout = nn.Dropout(p=0.55)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        # Calculate the output size after convolutions and pooling
        self.flatten_output_size = self.calculate_flatten_output_size(n_timesteps, n_features)
        self.fc1 = nn.Linear(self.flatten_output_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def calculate_flatten_output_size(self, n_timesteps, n_features):
        # Example calculation for convolutions followed by pooling
        x = torch.randn(1, n_features, n_timesteps)  # Example input shape (batch_size, in_channels, sequence_length)
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        flatten_size = x.view(x.size(0), -1).size(1)
        return flatten_size

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# fit and evaluate a model using pytorch
def evaluate_model(trainX, trainy, testX, testy):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, torch.cuda.get_device_name(0))
    verbose, epochs, batch_size = 1, 100, 32
    n_timesteps, n_features, n_outputs = trainX.shape[2], trainX.shape[1], trainy.shape[1]
    
    trainX = torch.tensor(trainX, dtype=torch.float32).to(device)
    trainy = torch.tensor(trainy, dtype=torch.float32).to(device)
    testX = torch.tensor(testX, dtype=torch.float32).to(device)
    testy = torch.tensor(testy, dtype=torch.float32).to(device)

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
    #optimizer = optim.Adam(model.parameters())
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=2)
    lr_decay_rate = 0.1
    lr_decay_steps = 10000
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
    scheduler = ExponentialLR(optimizer, lr_decay_rate**(1.0 / lr_decay_steps))
    early_stopping = EarlyStopping(tolerance=5, min_delta=10)

    if load_model:
        model.load_state_dict(torch.load('saved_models'+'/'+f'1dcnn_model_{anomaly_type}.pth'))
    else:
        # Training loop
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
                val_outputs = model(testX[:400])
                val_loss = criterion(val_outputs, testy[:400])
            print((f"\n train loss: {loss.item():.4f} | validation loss: {val_loss.item():.4f}"))

            # early stopping
            #early_stopping(loss, val_loss)
            #if early_stopping.early_stop:
            #    print("Early stopping at epoch:", epoch)
            #    break
        # save model
        torch.save(model.state_dict(), 'saved_models'+'/'+f'1dcnn_model_{anomaly_type}.pth')

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(testX)
        predicted = (outputs > 0.5).float()
        correct = (predicted == testy).sum().item()
        
        # Calculate test performance
        accuracy = accuracy_score(testy.cpu().numpy(), predicted.cpu().numpy())
        precision = precision_score(testy.cpu().numpy(), predicted.cpu().numpy())
        recall = recall_score(testy.cpu().numpy(), predicted.cpu().numpy())
        f1 = f1_score(testy.cpu().numpy(), predicted.cpu().numpy())
        z_metric = 0.4*accuracy + 0.2*precision + 0.2*recall + 0.2*f1
        print(f"{anomaly_type} | Accuracy: {accuracy:.4} | Precision: {precision:.4} | Recall: {recall:.4} | F1: {f1:.4} | Z:{z_metric:.4}")

        accuracy = correct / testy.shape[0]
    return model, accuracy


accuracy = 0
while accuracy < 0.70:
    model, accuracy = evaluate_model(X_train, y_train, X_val, y_val)
    print(accuracy)


#=====================##
#
#     TEST EVALUATION
#
#======================#

#device = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_labels = pd.read_csv("Test_Labels_prelim.csv")

predictions_prelim = []
for n_sample in range(1,103):
    sample_n = f"Sample{n_sample}"

    X_test = get_test_features(components, sample_n=sample_n, n_datapoints=fs)
    # Skip if not the same fundamental frequency
    if len(X_test) == 0:
        continue
    for n_scaler in range(X_test.shape[1]):
        X_test[:,n_scaler,:] = scalers[n_scaler].transform(X_test[:,n_scaler,:])
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        preds = preds.cpu().numpy()
    predictions_prelim.append(preds[0][0])
    print(sample_n, preds[0][0], test_labels.loc[n_sample-1, "FaultCode"] )

# Calculate test performance
pred = np.round(predictions_prelim)
true = (test_labels["FaultCode"] == anomaly_type).astype(int).to_numpy()
accuracy = accuracy_score(true, pred)
precision = precision_score(true, pred)
recall = recall_score(true, pred)
f1 = f1_score(true, pred)
z_metric = 0.4*accuracy + 0.2*precision + 0.2*recall + 0.2*f1
print(f"{anomaly_type} | Accuracy: {accuracy:.4} | Precision: {precision:.4} | Recall: {recall:.4} | F1: {f1:.4} | Z:{z_metric:.4}")

#scaler = MinMaxScaler()
#predictions = scaler.fit_transform(np.array(predictions).reshape(-1,1))

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
for n_sample in range(1,258+1):
    sample_n = f"Sample{n_sample}"

    X_test = get_test_features(components, sample_n=sample_n, n_datapoints=fs, final_stage=True)
    # Skip if not the same fundamental frequency
    if len(X_test) == 0:
        continue
    for n_scaler in range(X_test.shape[1]):
        X_test[:,n_scaler,:] = scalers[n_scaler].transform(X_test[:,n_scaler,:])
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        preds = preds.cpu().numpy()
    samples.append(sample_n)
    predictions_final.append(preds[0][0])

final_df = pd.DataFrame({"sample":samples,
                   f"{type_to_fault[anomaly_type]}":predictions_final,
                   })

final_df.to_csv(f"preds_1dcnn_final/preds_{type_to_fault[anomaly_type]}_1dcnn.csv", index=False)

# Idea: create separate models for each working condition (worse performance), better to use all working conditions
# Idea: scale each feature from -1 to 1, works great! Random splitting plays a role whether the model will learn or not
# Idea: Use final stage data for training (done)
# Idea: Only use the first 1-second on the training data (worse, the model did not learn)
# Idea: Create models for M0, G0, LA0, RA0
# Idea: Use Fourier amplitude spectrum as the input instead of the raw data (worked well!)
# Idea: Use Fourier (unwrapped) phase spectrum as additional input (slight worse)
# Observation: I noticed that models with high class imbalance perform bad
# Idea: Downsample the normal class if there is high class imbalance
# Idea: Add Gaussian noise (not yet implemented)
# Idea: leftaxlebox may be symmetrical with rightaxlbox. (Needs exploration)
# Idea: Do not apply abs() on the fft amplitude, then normalize to (-1,1) (error with imaginary values)
# Idea: 100 epochs with early stopping worked well! as well as several revisions in learning
# learning updates: use GPU, MinMaxScaler, tanh, dropout, small initial_lr, lr_scheduler, early_stopping
