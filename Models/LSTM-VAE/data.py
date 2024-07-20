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

def get_df_action(filepaths_csv, filepaths_meta, action2int=None, delimiter=";"):
    # Load dataframes
    print("Loading data.")
    # Make dataframes
    # Some classes show the output boolean parameter as True rather than true. Fix here
    dfs_meta = list()
    for filepath in filepaths_meta:
        df_m = pd.read_csv(filepath, sep=delimiter)
        df_m.str_repr = df_m.str_repr.str.replace('True', 'true')
        df_m['filepath'] = filepath
        dfs_meta.append(df_m)


    df_meta = pd.concat(dfs_meta)
    df_meta.index = pd.to_datetime(df_meta.init_timestamp.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    df_meta['completed_timestamp'] = pd.to_datetime(df_meta.completed_timestamp.astype('datetime64[ms]'),
                                                    format="%Y-%m-%dT%H:%M:%S.%f")
    df_meta['init_timestamp'] = pd.to_datetime(df_meta.init_timestamp.astype('datetime64[ms]'),
                                               format="%Y-%m-%dT%H:%M:%S.%f")

    # Eventually reduce number of classes
    # df_meta['str_repr'] = df_meta.str_repr.str.split('=', expand = True,n=1)[0]
    # df_meta['str_repr'] = df_meta.str_repr.str.split('(', expand=True, n=1)[0]

    actions = df_meta.str_repr.unique()
    dfs = [pd.read_csv(filepath_csv, sep=";") for filepath_csv in filepaths_csv]
    df = pd.concat(dfs)

    # Sort columns by name !!!
    df = df.sort_index(axis=1)

    # Set timestamp as index
    df.index = pd.to_datetime(df.time.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    # Drop useless columns
    columns_to_drop = [column for column in df.columns if "Abb" in column or "Temperature" in column]
    df.drop(["machine_nameKuka Robot_export_active_energy",
             "machine_nameKuka Robot_import_reactive_energy"] + columns_to_drop, axis=1, inplace=True)

    signals = df.columns

    df_action = list()
    for action in actions:
        for index, row in df_meta[df_meta.str_repr == action].iterrows():
            start = row['init_timestamp']
            end = row['completed_timestamp']
            df_tmp = df.loc[start: end].copy()
            df_tmp['action'] = action
            # Duration as string (so is not considered a feature)
            df_tmp['duration'] = str((row['completed_timestamp'] - row['init_timestamp']).total_seconds())

            df_action.append(df_tmp)
    df_action = pd.concat(df_action, ignore_index=True)
    df_action.index = pd.to_datetime(df_action.time.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    df_action = df_action[~df_action.index.duplicated(keep='first')]

    # Drop NaN
    df = df.dropna(axis=0)
    df_action = df_action.dropna(axis=0)

    if action2int is None:
        action2int = dict()
        j = 1
        for label in df_action.action.unique():
            action2int[label] = j
            j += 1

    df_merged = df.merge(df_action[['action']], left_index=True, right_index=True, how="left")
    # print(f"df_merged len: {len(df_merged)}")
    # Where df_merged in NaN Kuka is in idle state
    df_idle = df_merged[df_merged['action'].isna()].copy()
    df_idle['action'] = 'idle'
    df_idle['duration'] = df_action.duration.values.astype(float).mean().astype(str)
    df_action = pd.concat([df_action, df_idle])

    # ile label must be 0 for debug mode
    action2int['idle'] = 0
    print(f"Found {len(set(df_action['action']))} different actions.")
    print("Loading data done.\n")

    return df_action, df, df_meta, action2int

def get_records(filepath, sampling, file_numbers):
    filepath_csv = [os.path.join(filepath, f"rec{r}_20220811_rbtc_{sampling}s.csv") for r in file_numbers]
    filepath_meta = [os.path.join(filepath, f"rec{r}_20220811_rbtc_{sampling}s.metadata") for r in file_numbers]
    df_action, _, _, _ = get_df_action(filepath_csv, filepath_meta)
    df_action = df_action.drop(columns=['time', 'action', 'duration'])

    return df_action

def get_collisions(filepath='./kuka_dataset/collision/'):
    collisions = pd.read_excel(os.path.join(filepath, "20220811_collisions_timestamp.xlsx"))
    collisions_start = collisions[collisions['Inizio/fine'] == "i"].Timestamp - pd.to_timedelta(2, 'h')
    collisions_end = collisions[collisions['Inizio/fine'] == "f"].Timestamp - pd.to_timedelta(2, 'h')
    collisions_interval = np.vstack([collisions_start, collisions_end])

    return collisions_interval

def normalize(train_records, cal_records, test_records):
    min_max_scaler = MinMaxScaler()

    train = pd.DataFrame(min_max_scaler.fit_transform(train_records.values), columns=train_records.columns, index=train_records.index)
    cal = pd.DataFrame(min_max_scaler.transform(cal_records.values), columns=cal_records.columns, index=cal_records.index)
    test = pd.DataFrame(min_max_scaler.transform(test_records.values), columns=test_records.columns, index=test_records.index)

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

def get_train_windows(filepath='./kuka_dataset/normal/', sampling=0.1, file_numbers=[0, 2, 3, 4]):
    train_records = get_records(filepath, sampling, file_numbers)
    train_windows = get_windows(train_records, window_size)

    return train_windows

def get_cal_windows(window_size, k_pa, filepath='./kuka_dataset/collision/', sampling=0.1, file_numbers=[6]):
    cal_records = get_records(filepath, sampling, file_numbers)
    cal_windows = get_windows(cal_records, window_size)
    cal_windows, cal_labels = get_windows_labels_pa(cal_records, window_size, collisions_interval, k_pa)

    return cal_windows, cal_labels

def get_test_windows(window_size, k_pa, filepath='./kuka_dataset/collision/', sampling=0.1, file_numbers=[1, 5]):
    test_records = get_records(filepath, sampling, file_numbers)
    test_windows = get_windows(test_records, window_size)
    test_windows, test_labels = get_windows_labels_pa(test_records, window_size, collisions_interval, k_pa)

    return test_windows, test_labels