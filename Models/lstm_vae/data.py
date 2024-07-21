import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def is_collision(point, collisions_interval):
    for i in range(collisions_interval.shape[1]):
        x = collisions_interval[:, i]
        if point >= x[0] and point <= x[1]:
            return 1
    return 0

def point_adjust(window_label, k=0):
    if window_label[-1] == 1 or window_label.mean() > k:
        return 1
    else:
        return 0

def get_df(filepaths_csv, delimiter=";"):
    dfs = [pd.read_csv(filepath_csv, sep=delimiter) for filepath_csv in filepaths_csv]
    df = pd.concat(dfs)

    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    # Remove timezone information if it exists
    if pd.api.types.is_datetime64tz_dtype(df['time']):
        df['time'] = df['time'].dt.tz_localize(None)
    
    # Set timestamp as index
    df.index = df['time']

    # Drop useless columns
    columns_to_drop = [column for column in df.columns if "Abb" in column or "Temperature" in column]
    df.drop(["machine_nameKuka Robot_export_active_energy",
             "machine_nameKuka Robot_import_reactive_energy"] + columns_to_drop, axis=1, inplace=True)

    return df

def get_records(filepath, sampling, file_numbers):
    filepath_csv = [os.path.join(filepath, f"rec{r}_20220811_rbtc_{sampling}s.csv") for r in file_numbers]
    df = get_df(filepath_csv)
    df = df.drop(columns=['time'])

    return df

def get_collisions(filepath):
    collisions = pd.read_excel(os.path.join(filepath, "20220811_collisions_timestamp.xlsx"))
    collisions_start = collisions[collisions['Inizio/fine'] == "i"].Timestamp - pd.to_timedelta(2, 'h')
    collisions_end = collisions[collisions['Inizio/fine'] == "f"].Timestamp - pd.to_timedelta(2, 'h')
    collisions_interval = np.vstack([collisions_start, collisions_end])

    return collisions_interval

def normalize(records, scaler):
    values = scaler.transform(records.values)
    records = pd.DataFrame(values, columns=records.columns, index=records.index)

    return records

def not_work_normalize(train_records, cal_records, test_records):
    min_max_scaler = MinMaxScaler()

    print(np.max(train_records.values), np.min(train_records.values))
    print(np.max(cal_records.values), np.min(cal_records.values))
    print(np.max(test_records.values), np.min(test_records.values))

    train_values = min_max_scaler.fit_transform(train_records.values)
    cal_values = min_max_scaler.transform(cal_records.values)
    test_values = min_max_scaler.transform(test_records.values)

    print(np.max(train_values), np.min(train_values))
    print(np.max(cal_values), np.min(cal_values))
    print(np.max(test_values), np.min(test_values))

    train = pd.DataFrame(train_values, columns=train_records.columns, index=train_records.index)
    cal = pd.DataFrame(cal_values, columns=cal_records.columns, index=cal_records.index)
    test = pd.DataFrame(test_values, columns=test_records.columns, index=test_records.index)

    return train, cal, test

def get_windows(records, window_size):
    indices = np.arange(window_size)[None, :] + np.arange(records.shape[0] - window_size)[:, None]

    return np.array([records.values[idx] for idx in indices])

def get_windows_labels_pa(records, window_size, collisions_interval, k_pa):
    labels_for_point = np.array([is_collision(t, collisions_interval) for t in records.index])
    indices = np.arange(window_size)[None, :] + np.arange(records.shape[0] - window_size)[:, None]
    windows = get_windows(records, window_size)
    labels = np.array([point_adjust(labels_for_point[idx], k_pa) for idx in indices])

    return windows, labels

def get_train_windows(window_size, filepath='./kuka_dataset/normal', sampling=0.1, file_numbers=[0, 2, 3, 4]):
    print("Reading training data...")
    train_records = get_records(filepath, sampling, file_numbers)
    print("Getting windows for training data...")
    train_windows = get_windows(train_records, window_size)

    return train_windows

def get_cal_windows(window_size, k_pa, filepath='./kuka_dataset/collisions', sampling=0.1, file_numbers=[6]):
    print("Reading calibration data...")
    cal_records = get_records(filepath, sampling, file_numbers)
    print("Reading collisions data...")
    collisions_interval = get_collisions(filepath)
    print("Getting windows and labels for calibration data...")
    cal_windows, cal_labels = get_windows_labels_pa(cal_records, window_size, collisions_interval, k_pa)

    return cal_windows, cal_labels

def get_test_windows(window_size, k_pa, filepath='./kuka_dataset/collisions', sampling=0.1, file_numbers=[1, 5]):
    print("Reading test data...")
    test_records = get_records(filepath, sampling, file_numbers)
    print("Reading collisions data...")
    collisions_interval = get_collisions(filepath)
    print("Getting windows and labels for test data...")
    test_windows, test_labels = get_windows_labels_pa(test_records, window_size, collisions_interval, k_pa)

    return test_windows, test_labels

def get_data_windows(window_size, k_pa, sampling=0.1, file_numbers_train=[0, 2, 3, 4], file_numbers_cal=[6], file_numbers_test=[1, 5], filepath_train='./kuka_dataset/normal', filepath_cal='./kuka_dataset/collisions', filepath_test='./kuka_dataset/collisions'):
    print("Reading training data...")
    train_records = get_records(filepath_train, sampling, file_numbers_train)
    print("Reading calibration data...")
    cal_records = get_records(filepath_cal, sampling, file_numbers_cal)
    print("Reading test data...")
    test_records = get_records(filepath_test, sampling, file_numbers_test)
    print("Reading collisions data...")
    collisions_interval = get_collisions(filepath_cal)

    print("Normalizing data...")
    min_max_scaler = MinMaxScaler()
    train_values = min_max_scaler.fit_transform(train_records.values)
    cal_values = min_max_scaler.transform(cal_records.values)
    test_values = min_max_scaler.transform(test_records.values)

    print(type(train_values), type(cal_values), type(test_values))

    train_records = pd.DataFrame(train_values, columns=train_records.columns, index=train_records.index)
    cal_records = pd.DataFrame(cal_values, columns=cal_records.columns, index=cal_records.index)
    test_records = pd.DataFrame(test_values, columns=test_records.columns, index=test_records.index)

    print(np.max(train_records.values), np.min(train_records.values))
    print(np.max(cal_records.values), np.min(cal_records.values))
    print(np.max(test_records.values), np.min(test_records.values))

    print(type(train_values), type(cal_values), type(test_values))

    print("Getting windows for training data...")
    train_windows = get_windows(train_records, window_size)
    print("Getting windows and labels for calibration data...")
    cal_windows, cal_labels = get_windows_labels_pa(cal_records, window_size, collisions_interval, k_pa)
    print("Getting windows and labels for test data...")
    test_windows, test_labels = get_windows_labels_pa(test_records, window_size, collisions_interval, k_pa)

    return train_windows, cal_windows, cal_labels, test_windows, test_labels
    