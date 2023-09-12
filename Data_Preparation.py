import os
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # if gpu is available use gpu

def data_transfer(mat_file):
    del_list = []
    for key in mat_file.keys():
        if key[0] != '_':
            mat_file[key] = mat_file[key].squeeze()

        else:
            del_list.append(key)
    for key_to_remove in del_list:
        mat_file.pop(key_to_remove)

    return mat_file


def event_finder(dataframe_events, dataframe_time):
    dataframe_event = deepcopy(dataframe_events)
    dataframe_event["Event"] = 0
    for index, row in dataframe_time.iterrows():
        dataframe_event["Event"][row["first"]:row["last"]] = 1
    return dataframe_event


def create_dataset(sequences):
    dataset = [torch.tensor(s.values) for s in
               sequences]  # converting each sequence to a tensor & adding a dimension

    n_seq, seq_len, n_features = torch.stack(dataset).shape

    return dataset, seq_len, n_features


# Load the data
current_working_directory = os.getcwd()
data_path = os.path.join(current_working_directory, 'raw_data')

# rs_nc3_mat = sio.loadmat(f'{data_path}/RS_NC3.mat')
# s_nc3_mat = sio.loadmat(f'{data_path}/S_NC3.mat')
s_nc2_mat = sio.loadmat(f'{data_path}/S_NC2.mat')
s_nc2_labels = sio.loadmat(f'{data_path}/FirstLast_NC2.mat')
# Transfer the data
# rs_nc3_dataframe = pd.DataFrame(data_transfer(rs_nc3_mat))
# s_nc3_dataframe = pd.DataFrame(data_transfer(s_nc3_mat))
s_nc2_dataframe = pd.DataFrame(data_transfer(s_nc2_mat))
s_nc2_labels_dataframe = pd.DataFrame(data_transfer(s_nc2_labels))
s_nc2_dataframe = event_finder(s_nc2_dataframe, s_nc2_labels_dataframe)
time_event = s_nc2_dataframe[["Event"]]
data_transformer = MinMaxScaler()
s_nc2_dataframe = pd.DataFrame(data_transformer.fit_transform(s_nc2_dataframe),
                               columns=data_transformer.get_feature_names_out().tolist())

s_nc2_dataframe["amplitudeuWS_fourier"] = np.fft.fft(s_nc2_dataframe["amplitudeuWS"])
s_nc2_dataframe["phaseuWS_fourier"] = np.fft.fft(s_nc2_dataframe["phaseuWS"])
s_nc2_dataframe["ampCoulter_N_fourier"] = np.fft.fft(s_nc2_dataframe["ampCoulter_N"])
s_nc2_dataframe.drop(columns=["timeS", "Event"], inplace=True)
s_nc2_dataframe[["Event"]] = time_event

window_size = 2000
step_size = 500
# Create smaller Series using rolling windows
smaller_series_list = []
for start in range(0, len(s_nc2_dataframe) - window_size + 1, step_size):
    end = start + window_size
    window_series = s_nc2_dataframe.iloc[start:end].copy()
    smaller_series_list.append(window_series)

RANDOM_SEED = 42
train_df, val_df = train_test_split(smaller_series_list, test_size=0.15, random_state=RANDOM_SEED, shuffle=True)

# splitting val dataframe to val & test
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=RANDOM_SEED, shuffle=True)

train_dataset, seq_len, n_features = create_dataset(train_df)
val_dataset, seq_len, n_features = create_dataset(val_df)
test_dataset, seq_len, n_features = create_dataset(test_df)

print()
