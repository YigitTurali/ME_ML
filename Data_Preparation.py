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


def fft_denoiser(x, n_components, to_real=True):

    n = len(x)

    # compute the fft
    fft = np.fft.fft(x, n)

    # compute power spectrum density
    # squared magnitud of each fft coefficient
    PSD = fft * np.conj(fft) / n

    # keep high frequencies
    _mask = PSD > n_components
    fft = _mask * fft

    # inverse fourier transform
    clean_data = np.fft.ifft(fft)

    if to_real:
        clean_data = clean_data.real

    return clean_data

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

def create_sequences(X,y,seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        x = X[i:(i+seq_length)]
        y = y[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Load the data
current_working_directory = os.getcwd()
data_path = os.path.join(current_working_directory, 'raw_data')
i = 0
for file in os.listdir(data_path):
    i += 1
    if file.endswith(".mat") and not file.endswith(".csv"):
        data_dict = {}
        data_raw = data_transfer(sio.loadmat(f'{data_path}/{file}'))
        [data_raw.pop(key, None) for key in ["first","last"]]
        data_dict[file] = pd.DataFrame(data_raw)

        data_raw = data_transfer(sio.loadmat(f'{data_path}/{file}'))
        data_dict[f"{file}_labels"] = pd.DataFrame(np.array([data_raw["first"],data_raw["first"]]).reshape(-1,2),columns=["first","last"])
        data_dict[file] = event_finder(data_dict[file], data_dict[f"{file}_labels"])
        time_event = data_dict[file][["Event"]]
        data_dict[file].drop(columns=["timeS", "Event"], inplace=True)
        data_dict[file][["Event"]] = time_event

        num_rows_per_file = 30000
        num_files = len(data_dict[file]) // num_rows_per_file + 1
        for j in range(num_files):
            start_index = i * num_rows_per_file
            end_index = (i + 1) * num_rows_per_file

            # Slice the data for the current file
            subset_data = data_dict[file].iloc[start_index:end_index]
            subset_labels = data_dict[f"{file}_labels"].iloc[start_index:end_index]

            # Save the subset to a new CSV file
            subset_data.to_csv(f'{data_path}/{i}_part_{j}.csv', index=False)
            print(f"Process {i}:{j} is done")

all_files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.csv')]
list_of_dfs = [pd.read_csv(file) for file in all_files]
data_train = pd.concat(list_of_dfs, axis=0, ignore_index=True)
y = data_train["Event"]
X = data_train.drop(columns=["Event"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler(feature_range=(0, 1))

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

seq_length = 4000  # Choose your desired sequence length
X_train, y_train = create_sequences(X_train, y_train, seq_length)
X_test, y_test = create_sequences(X_test, y_test, seq_length)

