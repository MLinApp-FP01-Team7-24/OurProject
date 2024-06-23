from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os
import tqdm
import random 

from .dataset_utils import wavelet_spectrogram

class KukaDataset(Dataset):
    ds_config = None
    
    def __init__(self, data_path = "", verbose=True, test=False, columns_to_keep=None, 
                 wlist=None, config: dict = None):
        self.test = test
        KukaDataset.ds_config = config
        #load df in memory
        if wlist is not None: 
            self.kuka_df = wlist
        else: 
            #read the whole list of ts
            kuka_ts = [pd.read_csv(filepath_csv, sep=";") for filepath_csv in data_path]
            self.kuka_df = [] #for each ts
            for el in kuka_ts:
                el['time'] = pd.to_datetime(el['time'])
                el.sort_values(by=['time']) # sort by time             
                [self.kuka_df.append(el.iloc[start:start + config['trainer_params']['input_length']])
                                     for start in range(len(el) - config['trainer_params']['input_length'] + 1)]
        # add column for risk
        self.kuka_df['risk_level'] = 'Low'
        # add risky intervals
        
        try:
            df = pd.read_csv('20220811_collisions_timestamp.xlsx')
            anomalies = []
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
                    'Timestamp-Start': start.strftime('%Y-%m-%d %H:%M:%S'),
                    'Timestamp-End': end.strftime('%Y-%m-%d %H:%M:%S'),
                    'Duration (ms)': duration})
             
            mask = np.logical_or.reduce([
                    (self.kuka_df['time'] >= anomaly['Timestamp-Start']) \
                        & (self.kuka_df['time'] <= anomaly['Timestamp-End'])
                    for anomaly in anomalies])   
            self.kuka_df.loc[mask, 'risk_level'] = 'High'
        except: pass
        # drop time column
        self.kuka_df = self.kuka_df.drop(columns=['time'])

        #fit one hot encoder on labels
        if not self.test:
            print("--- Train Dataset ---")
            self.risk_encoder = OneHotEncoder()
            self.risk_encoder.fit(self.kuka_df["risk_level"].values.reshape(-1, 1))
            #preprocess df
            print("preprocessing ... ")
            self.kuka_df = self.__preprocess__(verbose)
            #save dataframe structure to apply on unseen data
            self.header_columns = []
            self.kept_columns = self.kuka_df.columns
        elif self.test:
            print("--- Test Dataset ---")
            if columns_to_keep is not None:
                column_to_drop = [ x for x in self.kuka_df.columns if x not in columns_to_keep]
                self.kuka_df.drop(column_to_drop, axis=1, inplace=True)
            self.header_columns = []

    @property
    def X(self):
        return self.kuka_df.loc[:,~self.kuka_df.columns.isin(self.header_columns)]
    
    @property
    def y(self):
        self.kuka_df.loc[:, "risk_level"]

    def __preprocess__(self, verbose = False):
        """
        Preprocess the kuka df by removing NaN columns, static columns and correlated features
        """
        assert self.kuka_df is not None
        
        if verbose:
            print("Dropping all NaN column")
        self.kuka_df.dropna(axis = 1, inplace=True)
        if verbose:
            print("Dropping all static columns")
        columns_to_drop = self.kuka_df.loc[:, self.kuka_df.apply(pd.Series.nunique) == 1].columns
        columns_to_drop = [ x for x in columns_to_drop if x not in self.header_columns]
        self.kuka_df.drop(columns_to_drop, axis=1, inplace=True)
        return self.kuka_df
    
    def get_schema(self):
        return self.kept_columns
        
    def __group_by_chassis__(self, verbose = True):
        assert self.kuka_df is not None
        
        #each chassis has now a df with its multivariate time series
        self.df_list = []
        groups = self.kuka_df.groupby("ChassisId_encoded")
        for name, group_df in tqdm.tqdm(groups, desc="Group and feature extraction"):
            if (not self.test and (len(group_df) < 5) or 
                                  ((not self.keepfaulty) and (np.sum(group_df['risk_level'].values != 'Low') > 0))):
                continue
            
            group_headings = group_df[self.header_columns]
            group_features = group_df.drop(self.header_columns, axis = 1)

            # diffs = group_features.diff(axis=1).fillna(0)
            # diffs.columns = [x + "_diff" for x in group_features.columns]

            #wavelet_df = wavelet_spectrogram(group_features, 5)

            group_df = pd.concat([group_headings.reset_index(drop=True), 
                                  group_features.reset_index(drop=True), 
                                #   diffs.reset_index(drop=True),
                                  #wavelet_df.reset_index(drop=True)
                                  ], axis=1)

            self.df_list.append(group_df)

        if not self.test:
            self.df_list = [x for x in self.df_list if len(x) > 10]
        print(f"{len(self.df_list)}")
        return self.df_list
    
    def get_n_features(self):
        assert self.kuka_df is not None
        features, labels = self[0]
        return features.shape[-1]
        
    def __len__(self):
        return len(self.df_list)

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (int): idx over df_lists of chassis df

        Returns:
            tuple: time_series, one_hot labels for each point in time series
        """
        assert idx < len(self), f"Got {idx=} when {len(self)=}"
        # retrieve the idx-th group
        ts = self.df_list[idx].sort_values(by=["Timesteps"], ascending=True)
        # retrieve all usefull infromation from that df
        chassis = ts["ChassisId_encoded"].iloc[0]
        # generate multivariate timesereies (n_timesteps, 289) 289 atm with simple preprocess

        time_series = ts.drop(self.header_columns, axis = 1).values

        # point_wise labels
        if not self.test:
            #train data with labels 
            timestep_labels = ts["risk_level"]
            labels = self.risk_encoder.transform(timestep_labels.values.reshape(-1, 1)).todense()
        elif self.test:
            #test data without risk_level as key in dataframe
            labels = np.empty((len(time_series),3))
            labels.fill(np.nan)

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
        return data, labels # data_alligned, labels_alligned, mask