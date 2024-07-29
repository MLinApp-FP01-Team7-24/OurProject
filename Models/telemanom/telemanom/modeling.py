
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import logging
import telemanom.helpers as helpers

# Sopprime gli avvisi di ottimizzazione della CPU di TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = helpers.getLogger()


class LSTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, config):
        super(LSTModel, self).__init__()
        self.list_lstm = []
        self.list_dropout = []
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.list_lstm = nn.ModuleList()
        self.list_dropout = nn.ModuleList()

        self.list_lstm.append(nn.LSTM(input_dim, hidden_dim1, batch_first=True))
        for i in range(0, self.config.layer_LSTM - 2):
            self.list_lstm.append(nn.LSTM(hidden_dim1, hidden_dim1, batch_first=True))
        self.list_lstm.append(nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True))

        for i in range(self.config.layer_LSTM):
            self.list_dropout.append(nn.Dropout(self.config.dropout))

        self.fc = nn.Linear(hidden_dim2, self.config.n_predictions)
        if self.config.activation == "relu":
            self.activation = nn.ReLU()  # Aggiunta dell'attivazione ReLU
        if self.config.activation == "sigmoid":
            self.activation = nn.Sigmoid()  # Aggiunta dell'attivazione Sigmoid
        if self.config.activation == "leakyReLu":
            self.activation = nn.LeakyReLU()  # Aggiunta dell'attivazione Sigmoid

    def forward(self, x):
        for lstm, dropout in zip(self.list_lstm, self.list_dropout):
            x, _ = lstm(x)
            x = dropout(x)

        x = self.fc(x)
        return self.activation(x)


class Model:
    def __init__(self, config, run_id, channel):
        """
        Carica/allenare una RNN e prevede i valori di telemetria futuri per un canale.

        Args:
            config (obj): oggetto Config contenente i parametri per l'elaborazione e l'allenamento del modello
            run_id (str): ID di riferimento per il set di previsioni in uso
            channel (obj): oggetto Channel contenente i dati di train/test per X, y per un singolo canale

        Attributes:
            config (obj): vedi Args
            chan_id (str): ID del canale
            run_id (str): vedi Args
            y_hat (arr): valori del canale previsti
            model (obj): modello RNN allenato per prevedere i valori del canale
        """
        self.config = config
        self.chan_id = channel.id
        self.run_id = run_id
        self.y_hat = np.array([])
        self.model = None

        path_model_run_id = helpers.get_correct_path(os.path.join('trained_models/telemanom', self.run_id, 'models', self.chan_id + '.pt'))
        if not self.config.train or os.path.isfile(path_model_run_id):
            logger.info(f"Loading the model for the channel {self.chan_id} in the path : {path_model_run_id}")
            try:
                self.load(path_model_run_id)
                logger.info(f"Correctly loaded the model for the channel {self.chan_id} in the path : {path_model_run_id}")
            except FileNotFoundError:
                logger.info(f"Failed loaded the model for the channel {self.chan_id} in the path : {path_model_run_id}. Starting training new model")
                self.train_new(channel)
                self.save()
        else:
            self.train_new(channel)
            self.save()

    def load(self,path_model):
        """
        Carica il modello per il canale.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.load(path_model,map_location=device)

    def train_new(self, channel):
        """
        Allena un modello LSTM secondo le specifiche in config.yaml.

        Args:
            channel (obj): oggetto Channel contenente i dati di train/test per X, y per un singolo canale
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = LSTModel(channel.X_train.shape[2], self.config.layers[0], self.config.layers[1],self.config).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate,
                                     weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

        if channel.dataset_channel is not None and len(channel.dataset_channel)>0:
            self.calibrate(channel)

        train_data=TensorDataset(torch.tensor(channel.X_train).float(),torch.tensor(channel.y_train).float())
        train_loader = DataLoader(train_data, batch_size=self.config.lstm_batch_size, shuffle=False)
        the_last_loss = float('inf')
        patience_counter = 0

        self.model.train()
        early_stopped = False
        for epoch in range(self.config.epochs+1):
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                outputs = outputs[:,-1, :]
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            scheduler.step(loss)  # Update the learning rate based on the loss

            if loss.item() < the_last_loss - self.config.min_delta:
                the_last_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping triggered at epoch {}".format(epoch))
                    early_stopped = True
                    break

            if epoch % 2 == 0:
                logger.info(f'Epoch {epoch}, Loss: {loss.item()}')

    def save(self):
        """
        Salva il modello allenato.
        """
        torch.save(self.model, helpers.get_correct_path(os.path.join('trained_models/telemanom', self.run_id, 'models', '{}.pt'.format(self.chan_id))))

    def predict(self, x):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.eval()  # Imposta il modello in modalità valutazione (disabilita Dropout ecc.)
        with torch.no_grad():  # Disabilita il calcolo dei gradienti per l'inferenza
            #x = torch.tensor(x).float().unsqueeze(0)  # Converte x in tensore, aggiunge una dimensione batch
            x_tensor = torch.tensor(x, dtype=torch.float).to(device)
            prediction = self.model(x_tensor)  # Usa il metodo forward definito
        return prediction.cpu().numpy()  # Converti il tensore di output in un array numpy

    def aggregate_predictions(self, y_hat_batch, method='first'):
        """
        Aggrega le previsioni per ciascun timestep. Quando si prevede n passi avanti dove n > 1,
        si avranno più previsioni per un timestep.

        Args:
            y_hat_batch (arr): previsioni con forma (<lunghezza batch>, <n_preds>)
            method (string): indica come aggregare per un timestep - "first" o "mean"
        """
        agg_y_hat_batch = np.array([])

        for t in range(len(y_hat_batch)):
            start_idx = t - self.config.n_predictions
            start_idx = start_idx if start_idx >= 0 else 0

            # Le previsioni relative a un timestep specifico si trovano lungo la diagonale
            y_hat_t = np.flipud(y_hat_batch[start_idx:t+1]).diagonal()

            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)

    def batch_predict(self, channel):
        """
        Utilizza il modello LSTM allenato per prevedere i dati di test che arrivano in batch.

        Args:
            channel (obj): oggetto Channel contenente i dati di train/test per X, y per un singolo canale

        Returns:
            channel (obj): oggetto Channel con i valori y_hat come attributo
        """
        num_batches = int((channel.y_test.shape[0] - self.config.l_s) / self.config.batch_size)
        if num_batches < 0:
            raise ValueError('l_s ({}) troppo grande per la lunghezza del flusso {}.'.format(self.config.l_s, channel.y_test.shape[0]))

        for i in range(0, num_batches + 1):
            prior_idx = i * self.config.batch_size
            idx = (i + 1) * self.config.batch_size

            if i + 1 == num_batches + 1:
                idx = channel.y_test.shape[0]

            X_test_batch = channel.X_test[prior_idx:idx]
            y_hat_batch = self.predict(X_test_batch)
            y_hat_batch = y_hat_batch[:,-1,:]
            self.aggregate_predictions(y_hat_batch)

        self.y_hat = np.reshape(self.y_hat, (self.y_hat.size,))
        channel.y_hat = self.y_hat

        np.save(helpers.get_correct_path(os.path.join('trained_models/telemanom', self.run_id, 'y_hat', '{}.npy'.format(self.chan_id))), self.y_hat)

        return channel

    def calibrate(self, channel):
        logger.info("Starting calibration for the model ...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate,weight_decay=self.config.weight_decay)

        calibration_data = TensorDataset(torch.tensor(channel.X_calibration).float(),torch.tensor(channel.y_calibration).float())
        calibration_loader = DataLoader(calibration_data, batch_size=self.config.lstm_batch_size, shuffle=False)

        calibration_epochs = 15
        for epoch in range(calibration_epochs+1):
            for x_calib, y_calib in calibration_loader:
                x_calib, y_calib = x_calib.to(device), y_calib.to(device)
                optimizer.zero_grad()
                outputs = self.model(x_calib)
                outputs = outputs[:, -1, :]
                loss = criterion(outputs, y_calib)
                loss.backward()
                optimizer.step()

        logger.info("Ended calibration for the model ...")