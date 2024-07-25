from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os
import tqdm
import random 
from datetime import datetime

from .dataset_utils import wavelet_spectrogram

class KukaDataset(Dataset):
    ds_config = None
    
    def __init__(self, data_path = "", verbose=True, test=False, columns_to_keep=None, keep_faulty=True, risk_encoder=None,
                 wlist=None, time_first=False, drop_default=False, config: dict = None):
        self.time_first=time_first
        self.test = test
        self.keep_faulty = keep_faulty
        self.risk_encoder = risk_encoder
        KukaDataset.ds_config = config
        #load df in memory
        if wlist is not None: 
            self.kuka_df = wlist
        else: 
            anomalies = []
            if self.keep_faulty:
                if verbose: print('reading anomalies...')
                try:
                    print('start adding high risk')
                    df = pd.read_excel(os.path.join(data_path,'20220811_collisions_timestamp.xlsx'))
                    print('found the excel and loaded it...')
                    
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')
                    timezone_offset = -2
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True) + pd.Timedelta(hours=timezone_offset)

                    # Separare le righe di inizio e fine
                    starts = df[df['Inizio/fine'] == 'i'].reset_index(drop=True)
                    ends = df[df['Inizio/fine'] == 'f'].reset_index(drop=True)

                    # Assumiamo che ogni 'i' ha un corrispondente 'f' nella stessa sequenza
                    for i, (start, end) in enumerate(zip(starts['Timestamp'], ends['Timestamp'])):
                        duration = (end - start).total_seconds() * 1000  # Durata in millisecondi
                        anomalies.append({
                            'ID': i + 1,
                            'Timestamp-Start': start,#datetime.strptime(start, '%Y-%m-%d %H:%M:%S'),
                            'Timestamp-End': end,#datetime.strptime(end, '%Y-%m-%d %H:%M:%S'),
                            'Duration (ms)': duration})
                except Exception as e: 
                    raise

            if verbose: print('reading data...')
            #read the whole list of ts
            kuka_ts = [pd.read_csv(os.path.join(data_path, fpath), sep=";") for fpath in os.listdir(data_path)
                       if fpath.endswith('_0.1s.csv')]
            
            # Sort columns by name !!!
            kuka_ts = [df.sort_index(axis=1) for df in kuka_ts]
            
            
            # Drop useless columns
            if drop_default:
                columns_to_drop = [column for column in kuka_ts[0].columns 
                                   if "Abb" in column or "Temperature" in column]
                kuka_ts = [df.drop(["machine_nameKuka Robot_export_active_energy",
                            "machine_nameKuka Robot_import_reactive_energy"] + columns_to_drop, axis=1) 
                            for df in kuka_ts]
            
            columns_to_drop = []
            if not self.test: 
                print("preprocessing ... ")
                self.header_columns = []
                
                concatenated_df = pd.concat(kuka_ts)
                columns_with_nan = concatenated_df.columns[concatenated_df.isna().any()].tolist()
                columns_to_drop += columns_with_nan

                columns_to_drop_ = concatenated_df.loc[:, concatenated_df.apply(pd.Series.nunique) == 1].columns
                columns_to_drop += [x for x in columns_to_drop_ if x not in self.header_columns]

                self.kept_columns = [ x for x in kuka_ts[0].columns if x not in columns_to_drop]

            elif self.test:
                if columns_to_keep is not None:
                    columns_to_drop = [ x for x in kuka_ts[0].columns if x not in columns_to_keep]
                self.header_columns = []
            #drop columns
            [el.drop(columns_to_drop, axis=1, inplace=True) for el in kuka_ts]

            self.kuka_df = [] #for each ts
            self.targets = []
            for idx, el in enumerate(kuka_ts):
                print('windowing ts {}/{}'.format(idx + 1, len(kuka_ts)))
                el['time'] = pd.to_datetime(el['time'])
                el.sort_values(by=['time'], inplace=True) # sort by time             
                for start in range(len(el) - config['trainer_params']['input_length'] + 1):
                    window_df = el.iloc[start:start + config['trainer_params']['input_length']]
                    
                    #if self.keep_faulty and np.logical_or.reduce([
                    #        (window_df['time'] >= anomaly['Timestamp-Start']) \
                    #            & (window_df['time'] <= anomaly['Timestamp-End'])
                    #        for anomaly in anomalies]).any() :
                    last_timestamp = window_df.iloc[-1]['time'].to_pydatetime()
                    if self.keep_faulty and any(
                            (last_timestamp >= anomaly['Timestamp-Start']) &
                            (last_timestamp <= anomaly['Timestamp-End'])
                            for anomaly in anomalies):
                        self.targets.append('High')
                    else: self.targets.append('Low')
                    window_df.drop(columns=['time'], inplace=True)
                    self.kuka_df.append(window_df)
            if 'time' in self.kept_columns: self.kept_columns.remove('time')

        if verbose: print('files were read...')
        print(len(self.kuka_df), self.kuka_df[0].shape)

        #fit one hot encoder on labels
        if not self.test:
            print("--- Train Dataset ---")
            if self.risk_encoder is None:
                self.risk_encoder = OneHotEncoder()
                self.risk_encoder.fit(self.targets.reshape(-1, 1))
            #preprocess df
            #print("preprocessing ... ")
            #self.header_columns = []
            #self.kuka_df = self.__preprocess__(verbose)
            ##save dataframe structure to apply on unseen data         
            #self.kept_columns = self.kuka_df[0].columns
        elif self.test:
            print("--- Test Dataset ---")
            #if columns_to_keep is not None:
            #    column_to_drop = [ x for x in self.kuka_df[0].columns if x not in columns_to_keep]
            #    [window.drop(column_to_drop, axis=1, inplace=True) for window in self.kuka_df] 
            #self.header_columns = []
        if verbose: print('df len is:', len(self.kuka_df), 'window shape is:', self.kuka_df[0].shape)

    @property
    def X(self):
        return self.kuka_df.loc[:,~self.kuka_df.columns.isin(self.header_columns)]
    
    @property
    def y(self):
        self.kuka_df.loc[:, "risk_level"]
    
    #def __preprocess__(self, df, verbose=False):
    #    """
    #    Preprocess the kuka df by removing NaN columns, static columns, and correlated features
    #    """
    #    assert df is not None
#
    #    if verbose: print("Dropping all NaN columns across all windows")
    #    concatenated_df = pd.concat(df)
    #    columns_with_nan = concatenated_df.columns[concatenated_df.isna().any()].tolist()
    #    df.drop(columns_with_nan, axis=1) 
#
    #    if verbose:print("Dropping all static columns across all windows")
    #    columns_to_drop = concatenated_df.loc[:, concatenated_df.apply(pd.Series.nunique) == 1].columns
    #    columns_to_drop = [x for x in columns_to_drop if x not in self.header_columns]
    #    df.drop(columns_to_drop, axis=1)
#
    #    return df

    
    def get_schema(self):
        return self.kept_columns
    
    def get_n_features(self):
        assert self.kuka_df is not None
        features, labels = self[0]
        return features.shape[-1]
        
    def __len__(self):
        return len(self.kuka_df)

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (int): idx over df_lists of chassis df

        Returns:
            tuple: time_series, one_hot labels for each point in time series
        """
        assert idx < len(self), f"Got {idx=} when {len(self)=}"
        time_series = self.kuka_df[idx].values.astype(np.float32)
        # point_wise labels
        if not self.test:
            #train data with labels 
            timestep_labels = self.targets[idx]
            labels = self.risk_encoder.transform(np.array(timestep_labels).reshape(-1, 1)).todense()
        elif self.test:
            #test data without risk_level as key in dataframe
            labels = np.empty((len(time_series),3))
            labels.fill(np.nan)

        if self.time_first: return torch.Tensor(time_series) , torch.Tensor(labels)
        return torch.transpose(torch.Tensor(time_series), 1, 0) , torch.transpose(torch.Tensor(labels), 1, 0)
    
    @staticmethod
    def padding_collate_fn(batch):
        data, labels = zip(*batch)
        
        # get shapes
        n_features = data[0].shape[0]
        n_labels = labels[0].shape[0]
        ## compute max len
        ##max_len = max([d.shape[1] for d in data])
        #max_len = KukaDataset.ds_config['trainer_params']['input_length']
#
        ## allign data with respect to max sequence len
        #data_alligned = torch.zeros((len(batch), n_features, max_len))
        #labels_alligned = torch.zeros((len(batch), n_labels, max_len))
        ## 0 where we are happier this way
        #mask = torch.zeros((len(batch), max_len))
 #
        #for i, d in enumerate(data):
        #    window_offset = random.randrange(d.shape[1])
        #    
        #    #right aligning shorter ts
        #    data_alligned[i, :, max_len - (window_offset - max(0,window_offset - max_len)):] = d[:,max(0,window_offset - max_len):window_offset]
        #    labels_alligned[i, :, max_len - (window_offset - max(0,window_offset - max_len)):] = labels[i][:,max(0,window_offset - max_len):window_offset]
        #    # set 1 where meaningful values
        #    mask[i,:window_offset] = 1
        return data, labels # data_alligned, labels_alligned