import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import sys
import gc
import pickle
import umap

from ptflops import get_model_complexity_info

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.functions import *
from utils.classes import CNNFeatureExtractor, Classifier

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

TRAIN_DIR = "Preliminary stage/Data_Pre Stage/Data_Pre Stage/Data_Pre Stage/Training data"
TEST_DIR = "Preliminary stage/Data_Pre Stage/Data_Pre Stage/Data_Pre Stage/Test data"
FINAL_TRAIN_DIR = "Data_Final_Stage/Data_Final_Stage/Training"
FINAL_TEST_DIR = "Data_Final_Stage/Data_Final_Stage/Test_Final_round8/Test"

load_model = False
fs = 64000
anomaly_type = sys.argv[1] 
print(anomaly_type)
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
cv_folds = pd.read_csv("Folder_5fold_CV/CV_5fold.csv")
cv_folds = cv_folds[cv_folds['train_test']=='train'].reset_index(drop=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#========================#
#
#      LOAD DATA
#
#========================#

def load_data_with_cv():

    # Use CV folds
    fold = 1
    cv_Xtrain = cv_folds[cv_folds[f'{type_to_fault[anomaly_type]}_fold']!=fold][['prelim_final','train_test','label','sample']].reset_index(drop=True)
    cv_ytrain = cv_folds[cv_folds[f'{type_to_fault[anomaly_type]}_fold']!=fold][type_to_fault[anomaly_type]].reset_index(drop=True)
    cv_Xval = cv_folds[cv_folds[f'{type_to_fault[anomaly_type]}_fold']==fold][['prelim_final','train_test','label','sample']].reset_index(drop=True)
    cv_yval = cv_folds[cv_folds[f'{type_to_fault[anomaly_type]}_fold']==fold][type_to_fault[anomaly_type]].reset_index(drop=True)
    
    # Retrieve the data for train set according to folds
    data_ytrain = []
    for i in tqdm(range(len(cv_Xtrain))):
        prelim_final = cv_Xtrain.loc[i, 'prelim_final']
        label = cv_Xtrain.loc[i, 'label']
        sample_n = cv_Xtrain.loc[i, 'sample']
        binary_label = cv_ytrain.iloc[i]
    
        if prelim_final == 'prelim':
            fault_type = full_fault_to_type[label]
            if i == 0:

                df = pd.DataFrame()
                for component in components:
                    data_df = pd.read_csv(f"{TRAIN_DIR}/{fault_type}/{sample_n}/data_{component}.csv")
                    df = pd.concat([df, data_df], axis=1)
                
                n_samples = len(df) // (fs)
                n_features = len(df.columns)
                df = reduce_mem_usage(df, verbose=False)
                data_arr_Xtrain = df_to_numpy(df, n_samples, n_features, fs)
                data_ytrain.extend([binary_label]*len(data_arr_Xtrain))
            else:

                df = pd.DataFrame()
                for component in components:
                    data_df = pd.read_csv(f"{TRAIN_DIR}/{fault_type}/{sample_n}/data_{component}.csv")
                    df = pd.concat([df, data_df], axis=1)
                
                n_samples = len(df) // (fs)
                n_features = len(df.columns)
                df = reduce_mem_usage(df, verbose=False)
                temp_arr_Xtrain = df_to_numpy(df, n_samples, n_features, fs)
                data_ytrain.extend([binary_label]*len(temp_arr_Xtrain))
                data_arr_Xtrain = np.vstack((data_arr_Xtrain, temp_arr_Xtrain))

        elif prelim_final == 'final':
            if i == 0:
                df = pd.DataFrame()
                for component in components:
                    data_df = pd.read_csv(f"{FINAL_TRAIN_DIR}/{label}/{sample_n}/data_{component}.csv")
                    df = pd.concat([df, data_df], axis=1)

                df = reduce_mem_usage(df, verbose=False)
                n_samples = len(df) // (fs)
                n_features = len(df.columns)
                
                data_arr_Xtrain = df_to_numpy(df, n_samples, n_features, fs)
                data_ytrain.extend([binary_label]*len(data_arr_Xtrain))
            else:
                df = pd.DataFrame()
                for component in components:
                    data_df = pd.read_csv(f"{FINAL_TRAIN_DIR}/{label}/{sample_n}/data_{component}.csv")
                    df = pd.concat([df, data_df], axis=1)

                df = reduce_mem_usage(df, verbose=False)
                n_samples = len(df) // (fs)
                n_features = len(df.columns)
                
                temp_arr_Xtrain = df_to_numpy(df, n_samples, n_features, fs)
                data_ytrain.extend([binary_label]*len(temp_arr_Xtrain))
                data_arr_Xtrain = np.vstack((data_arr_Xtrain, temp_arr_Xtrain))

    # Retrieve the data for validation set according to folds
    data_yval = []
    for i in tqdm(range(len(cv_Xval))):
        prelim_final = cv_Xval.loc[i, 'prelim_final']
        label = cv_Xval.loc[i, 'label']
        sample_n = cv_Xval.loc[i, 'sample']
        binary_label = cv_yval.iloc[i]
        
        if prelim_final == 'prelim':
            fault_type = full_fault_to_type[label]
            if i == 0:

                df = pd.DataFrame()
                for component in components:
                    data_df = pd.read_csv(f"{TRAIN_DIR}/{fault_type}/{sample_n}/data_{component}.csv")
                    df = pd.concat([df, data_df], axis=1)
                
                n_samples = len(df) // (fs)
                n_features = len(df.columns)
                df = reduce_mem_usage(df, verbose=False)
                data_arr_Xval = df_to_numpy(df, n_samples, n_features, fs)
                data_yval.extend([binary_label]*len(data_arr_Xval))

            else:

                df = pd.DataFrame()
                for component in components:
                    data_df = pd.read_csv(f"{TRAIN_DIR}/{fault_type}/{sample_n}/data_{component}.csv")
                    df = pd.concat([df, data_df], axis=1)
                
                n_samples = len(df) // (fs)
                n_features = len(df.columns)
                df = reduce_mem_usage(df, verbose=False)
                temp_arr_Xval = df_to_numpy(df, n_samples, n_features, fs)
                data_yval.extend([binary_label]*len(temp_arr_Xval))
                data_arr_Xval = np.vstack((data_arr_Xval, temp_arr_Xval))

        elif prelim_final == 'final':
            if i == 0:

                df = pd.DataFrame()
                for component in components:
                    data_df = pd.read_csv(f"{FINAL_TRAIN_DIR}/{label}/{sample_n}/data_{component}.csv")
                    df = pd.concat([df, data_df], axis=1)

                df = reduce_mem_usage(df, verbose=False)
                n_samples = len(df) // (fs)
                n_features = len(df.columns)
                
                data_arr_Xval = df_to_numpy(df, n_samples, n_features, fs)
                data_yval.extend([binary_label]*len(data_arr_Xval))

            else:

                df = pd.DataFrame()
                for component in components:
                    data_df = pd.read_csv(f"{FINAL_TRAIN_DIR}/{label}/{sample_n}/data_{component}.csv")
                    df = pd.concat([df, data_df], axis=1)

                df = reduce_mem_usage(df, verbose=False)
                n_samples = len(df) // (fs)
                n_features = len(df.columns)
                
                temp_arr_Xval = df_to_numpy(df, n_samples, n_features, fs)
                data_yval.extend([binary_label]*len(temp_arr_Xval))
                data_arr_Xval = np.vstack((data_arr_Xval, temp_arr_Xval))

    y_train = np.array(data_ytrain).reshape(-1,1)
    y_val = np.array(data_yval).reshape(-1,1)

    # Obtain the global min and max per speed working condition per feature
    speed_wc_train = get_speed_wc_arr(data_arr_Xtrain)
    X_train, min_max_values = normalize_data(data_arr_Xtrain, speed_wc_train)
    speed_wc_val = get_speed_wc_arr(data_arr_Xval)
    X_val, _ = normalize_data(data_arr_Xval, speed_wc_val, min_max_values)

    # save min_max_values for the specific fault type
    np.save(f"saved_scalers/{anomaly_type}_min_max_values.npy", min_max_values)

    # Generate a random permutation
    permutation_train = np.random.permutation(len(X_train))

    # Shuffle both arrays using the same permutation
    X_train = X_train[permutation_train]
    y_train = y_train[permutation_train]

    return X_train, y_train, X_val, y_val


X_train, y_train, X_val, y_val = load_data_with_cv()
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

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

X_train = remove_channels(X_train)
X_val = remove_channels(X_val)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

#plot_3d(X_val[2,:,:], filename='X_val_fft_scaled3d.png', title='X_val_fft_scaled')

# fit and evaluate a model using pytorch
def train_cnn_classifier(X_train, y_train, X_val, y_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, torch.cuda.get_device_name(0))
    verbose, epochs, batch_size = 1, 100, 32
    input_channels, input_length = X_train.shape[1], X_train.shape[2]

    anomaly_len = sum(y_train)[0]
    normal_len = len(y_train) - anomaly_len
    print(anomaly_len, normal_len)

    # Convert data to torch tensor
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Data loaders
    train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # class weights for imbalanced data
    weight_for_0 = torch.tensor(anomaly_len / (normal_len + anomaly_len), dtype=torch.float32).to(device)
    weight_for_1 = torch.tensor(normal_len / (normal_len + anomaly_len), dtype=torch.float32).to(device)
    class_weights = torch.tensor([weight_for_0, weight_for_1]).to(device)

    # Initialize Classifier model
    cnn_output_size = 256 * 128 #batch_size * input_length 
    cnn = CNNFeatureExtractor(input_channels, input_length).to(device)
    classifier = Classifier(latent_dim=cnn_output_size).to(device)

    cnn.apply(init_linear_weights)
    cnn.apply(init_conv_weights)
    classifier.apply(init_linear_weights)

    # Loss and optimizer for the classifier
    pos_weight = class_weights[1] / class_weights[0]  # Calculate the positive weight
    classifier_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    cnn_optimizer = optim.Adam(list(cnn.parameters()), lr=1e-3)
    classifier_optimizer = optim.Adam(list(classifier.parameters()), lr=1e-3)
    scheduler = ReduceLROnPlateau(classifier_optimizer, 'min', patience=5, factor=0.5)

    if load_model:
        cnn.load_state_dict(torch.load('saved_models'+'/'+f'cnn_{anomaly_type}.pth', weights_only=False, map_location=device))
        classifier.load_state_dict(torch.load('saved_models'+'/'+f'classifier_{anomaly_type}.pth', weights_only=False, map_location=device))
    else:
        # Train the classifier
        history = {f"train":[], f"val":[]} 
        best_val_loss = 999
        for epoch in range(epochs):
            cnn.train()
            classifier.train()
            cnn_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            train_loss = 0
            
            # Iterate over batches
            for batch_X, batch_y in train_dataloader:
            
                if batch_X.cpu().numpy().shape[0] < 2:
                    break

                cnn_optimizer.zero_grad()
                classifier_optimizer.zero_grad()

                cnn_features = cnn(batch_X)
                inputs = cnn_features.view(cnn_features.size(0), -1) 

                # Forward pass
                outputs = classifier(inputs)

                # Compute loss
                loss = classifier_criterion(outputs, batch_y)

                # Backpropagation
                loss.backward()

                # Check for exploding/vanishing gradients
                '''total_norm = 0
                for param in classifier.parameters():
                    if param.grad is not None:
                        total_norm += param.grad.data.norm(2).item()**2
                total_norm = total_norm**(1./2)
                
                # check if it exceeds a threshold
                if total_norm > 1000:  # arbitrary threshold for exploding gradients
                    print("Exploding gradients detected!")
                    print(f'Gradient norm: {total_norm}')
                    continue
                elif total_norm < 1e-5:  # Threshold for vanishing gradients
                    print("Vanishing gradients detected!")
                    print(f'Gradient norm: {total_norm}')
                    continue'''

                cnn_optimizer.step()
                classifier_optimizer.step()
                train_loss += loss / len(batch_X)
            scheduler.step(train_loss)

            # Validation loss
            cnn.eval()
            classifier.eval()
            with torch.no_grad():
                val_loss = 0
                # Iterate over batches
                for batch_X, batch_y in val_dataloader:

                    cnn_features = cnn(batch_X)
                    inputs = cnn_features.view(cnn_features.size(0), -1) 
                    y_pred = classifier(inputs)
                    loss = classifier_criterion(y_pred, batch_y)
                    val_loss += loss / len(batch_X)

            if epoch % 10 == 0:
                print(f'epoch {epoch}/{epochs} | train loss: {train_loss.item():.4f} | val loss: {val_loss.item():.4f}')
            history[f"train"].append(train_loss.item())
            history[f"val"].append(val_loss.item())

            # save model
            if val_loss.item() <= best_val_loss:
                torch.save(cnn.state_dict(), 'saved_models'+'/'+f'cnn_{anomaly_type}.pth')
                torch.save(classifier.state_dict(), 'saved_models'+'/'+f'classifier_{anomaly_type}.pth')
                best_val_loss = val_loss.item()

        pd.DataFrame(history).to_csv(f"train_history/classifier_{anomaly_type}.csv", index=False)


    ####################
    #           
    #   FEATURE SPACE    
    #           
    ####################

    '''del classifier, history
    gc.collect()

    # Visualize latent space via tSNE
    latent_2d_tsne_all = []
    latent_2d_umap_all = []

    X_combined = X_train #torch.cat((X_val, X_train), dim=0)
    y_combined = y_train #torch.cat((y_val, y_train), dim=0)

    # Get the sorting indices based on y_combined
    sorted_indices = torch.argsort(y_combined, dim=0)

    # Rearrange the tensors using the sorted indices
    X_combined = X_combined[sorted_indices].squeeze(dim=1)
    y_combined = y_combined[sorted_indices].squeeze(dim=2)

    n_batches = 32  # Define the number of batches

    pca = PCA(n_components=batch_size)  # Reduce to smaller dimensions
    reducer = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.5, metric='cosine')
    for i in range(0, len(X_combined), batch_size):
        batch_X = X_combined[i:i+batch_size]
        batch_y = y_combined[i:i+batch_size]
        #if len(batch_X) < batch_size:
        #    break
        # Forward pass

        latent = cnn(batch_X)

        # flatten the latent then input to the classifier
        flattened_latent = latent.view(latent.size(0), -1) 

        if i == 0:
            features_reduced = pca.fit_transform(flattened_latent.detach().cpu().numpy())  
        else:
            features_reduced = pca.transform(flattened_latent.detach().cpu().numpy()) 
            
        # TSNE    
        #tsne = TSNE(n_components=2, perplexity=15, max_iter=250)
        #latent_2d_tsne = tsne.fit_transform(features_reduced)
        #latent_2d_tsne_all.append(latent_2d_tsne)

        # UMAP
        if i == 0:
            embedding = reducer.fit_transform(flattened_latent.detach().cpu().numpy())
        else:
            embedding = reducer.transform(flattened_latent.detach().cpu().numpy())
        latent_2d_umap_all.append(embedding)

        # Parallel Coordinates
        if i == 0:
            df = pd.DataFrame(features_reduced)
            df['label'] = batch_y[:len(df)].squeeze().cpu().numpy()
        df_temp = pd.DataFrame(features_reduced)
        df_temp['label'] = batch_y[:len(df)].squeeze().cpu().numpy()
        df = pd.concat([df, df_temp], axis=0)
        
    # Replace numerical class labels with textual labels
    df['label'] = df['label'].map({0.0: 'Normal', 1.0: 'Anomaly'})
    #df.to_csv('parallel_coordinates.csv', index=False)
    pd.plotting.parallel_coordinates(df, 'label', color=('blue','red'), alpha=0.1)
    plt.title(f'Parallel Coordinates - {anomaly_type}')
    plt.xlabel("Dimensions")
    plt.xticks(rotation=45)
    # Creating custom legend handles without transparency
    legend_handle1 = plt.Line2D([0], [0], color='blue', label='Normal')
    legend_handle2 = plt.Line2D([0], [0], color='red', label='Anomaly')
    plt.legend(handles=[legend_handle1, legend_handle2])
    plt.savefig(f'latent_space_figs/parallel_coord_{anomaly_type}_raw.png', dpi=300)
    plt.clf()

    #latent_2d_tsne_all = np.concatenate(latent_2d_tsne_all, axis=0)
    latent_2d_umap_all = np.concatenate(latent_2d_umap_all, axis=0)
    
    #plot_latent_space(latent_2d_tsne_all, y_combined[:len(latent_2d_tsne_all)].squeeze().cpu().numpy(), f'latent_space_figs/tsne_{anomaly_type}_raw.png', f'Feature Space - {anomaly_type}')
    plot_latent_space(latent_2d_umap_all, y_combined[:len(latent_2d_umap_all)].squeeze().cpu().numpy(), f'latent_space_figs/umap_{anomaly_type}_raw.png', f'Feature Space - {anomaly_type}')

    #del latent_2d_tsne_all, latent_2d_tsne, 
    del flattened_latent, latent_2d_umap_all, df
    gc.collect()'''

 

    #############
    #           #
    #   EVAL    #
    #           #
    #############

    # Evaluate the model on validation data
    cnn = CNNFeatureExtractor(input_channels, input_length).to(device)
    cnn.load_state_dict(torch.load('saved_models'+'/'+f'cnn_{anomaly_type}.pth', weights_only=False, map_location=device))
    cnn.eval()

    classifier = Classifier(latent_dim=cnn_output_size).to(device)
    classifier.load_state_dict(torch.load('saved_models'+'/'+f'classifier_{anomaly_type}.pth', weights_only=False, map_location=device))
    classifier.eval()

    with torch.no_grad():

        y_preds = []
        # Iterate over batches
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            cnn_features = cnn(batch_X)
            inputs = cnn_features.view(cnn_features.size(0), -1) 
            y_pred = classifier(inputs).squeeze()
            y_pred = (y_pred > 0.5).float().cpu().numpy()  # Apply threshold at 0.5 for binary classification
            if inputs.cpu().numpy().shape[0] < 2:
                break
            y_preds.extend(y_pred)

        y_train = y_train.squeeze().cpu().numpy()[:len(y_preds)]

        # Compute metrics
        accuracy = accuracy_score(y_train, y_preds)
        precision = precision_score(y_train, y_preds)
        recall = recall_score(y_train, y_preds)
        f1 = f1_score(y_train, y_preds)
        z_metric = 0.4*accuracy + 0.2*precision + 0.2*recall + 0.2*f1
        print(f"Train | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f} | Z: {z_metric:.4f}")

        y_preds = []
        # Iterate over batches
        for i in range(0, len(X_val), batch_size):
            batch_X = X_val[i:i+batch_size]
            batch_y = y_val[i:i+batch_size]

            cnn_features = cnn(batch_X)
            inputs = cnn_features.view(cnn_features.size(0), -1) 
            y_pred = classifier(inputs).squeeze()
            print(y_pred)
            y_pred = (y_pred > 0.5).float().cpu().numpy()  # Apply threshold at 0.5 for binary classification
            if inputs.cpu().numpy().shape[0] < 2:
                break
            y_preds.extend(y_pred)

        y_val = y_val.squeeze().cpu().numpy()[:len(y_preds)]

        print(y_val)

        # Compute metrics
        accuracy = accuracy_score(y_val, y_preds)
        precision = precision_score(y_val, y_preds)
        recall = recall_score(y_val, y_preds)
        f1 = f1_score(y_val, y_preds)
        z_metric = 0.4*accuracy + 0.2*precision + 0.2*recall + 0.2*f1
        print(f"Val | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f} | Z: {z_metric:.4f}")

train_cnn_classifier(X_train, y_train, X_val, y_val)

