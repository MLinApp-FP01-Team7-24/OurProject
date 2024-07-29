import numpy as np
import pandas as pd
import os
import logging
import telemanom.helpers as helpers

logger = helpers.getLogger()

class Channel:
    def __init__(self, config, chan_id, dataset_sample_rate, dataset_labels, sample_rate, date_prediction):
        """
        Carica e ridimensiona i valori del canale (previsti e reali).

        Args:
            config (obj): oggetto Config contenente i parametri per l'elaborazione
            chan_id (str): id del canale
            normal_path (str): percorso della directory dei dati normali
            collision_path (str): percorso della directory dei dati di collisione

        Attributes:
            id (str): id del canale
            config (obj): vedi Args
            normal_path (str): vedi Args
            collision_path (str): vedi Args
            X_train (arr): input di addestramento con dimensioni [timesteps, l_s, dimensioni di input]
            X_test (arr): input di test con dimensioni [timesteps, l_s, dimensioni di input]
            y_train (arr): valori reali di addestramento del canale con dimensioni [timesteps, n_predictions, 1]
            y_test (arr): valori reali di test del canale con dimensioni [timesteps, n_predictions, 1]
            train (arr): dati di addestramento caricati dal file .npy
            test (arr): dati di test caricati dal file .npy
        """

        assert dataset_sample_rate.get('train') is not None and dataset_sample_rate.get('test') is not None

        self.id = chan_id
        self.config = config
        self.sample_rate = sample_rate
        self.dataset_labels = dataset_labels
        self.train = dataset_sample_rate.get('train')
        self.test = dataset_sample_rate.get('test')
        self.dataset_calibration = None
        if dataset_sample_rate.get("calibration") is not None and len(dataset_sample_rate.get("calibration"))>0:
            self.dataset_calibration = dataset_sample_rate.get("calibration")
        self.date_prediction = date_prediction
        self.dataset_channel = dataset_sample_rate
        self.X_train = None
        self.y_train = None
        self.y_train_timestamp = None
        self.X_test = None
        self.y_test = None
        self.X_calibration = None
        self.y_calibration = None
        self.y_test_timestamp = None
        self.y_hat = None

        self.prepare_data()

    def shape_data(self, arr, type=""):
        """
        Prepara le sequenze di dati per l'addestramento o il testing del modello LSTM.
        Questa funzione trasforma un array di serie temporali in un formato adatto per il modello LSTM,
        creando sequenze di una lunghezza specificata ('l_s' + 'n_predictions') e separando
        queste sequenze in input (X) e output target (y).

        Args:
            arr (np.array): Array di input con dimensioni [timesteps, 2] dove la prima colonna è il timestamp
                            e la seconda colonna è il valore di telemetria.
            train (bool): Indica se i dati sono destinati all'addestramento (True) o al testing (False).

        Dettagli del Processo:
            - Ciascuna sequenza estratta contiene dati sufficienti per l'input del modello e per la previsione del target.
            - Le sequenze vengono estratte in modo tale che ciascuna sequenza inizi da un punto dati successivo al precedente,
              permettendo al modello di imparare da diverse fasi della serie temporale.
            - Dopo l'estrazione, le sequenze vengono trasformate in un array numpy per l'elaborazione.
            - Le sequenze vengono poi divise in componenti di input e target.
        """

        data = []
        # Per ogni indice i in arr, la funzione crea una sequenza di lunghezza l_s + n_predictions
        # (dove l_s è la lunghezza della sequenza di input e n_predictions è il numero di timestep futuri che il modello deve prevedere).
        # Questo viene fatto tramite lo slicing dell'array arr
        # Calcola il numero di sequenze possibili da estrarre basandosi sulla lunghezza specificata 'l_s' + 'n_predictions'
        for i in range(len(arr) - self.config.l_s - self.config.n_predictions):
            # Estrae una singola sequenza che include sia i dati per l'input che i dati per il target
            data.append(arr[i:i + self.config.l_s + self.config.n_predictions])
        data = np.array(data)

        # Assicurarsi che la forma dell'array risultante sia tridimensionale
        assert len(
            data.shape) == 3, "L'array di dati trasformato deve essere tridimensionale per soddisfare i requisiti del modello LSTM"

        # Esempio di Valori in self.X_train e self.y_train
        # Dato che l_s = 250 e n_predictions = 10, ogni sequenza sarà lunga 260. Di questi:
        #
        # I primi 250 valori di telemetria di ogni sequenza andranno in self.X_train.
        # Gli ultimi 10 valori di telemetria di ogni sequenza andranno in self.y_train.
        if type == "train":
            # self.X_train (o self.X_test se train=False) sarà composto da tutti i punti nella sequenza eccetto gli ultimi n_predictions.
            #  Dal esempio, per ogni sequenza in data, la porzione data[:, :-self.config.n_predictions, 1]
            #  seleziona tutti i valori di telemetria dall'inizio della sequenza fino al punto prima dell'inizio
            #  delle n_predictions. Questo esclude il timestamp e seleziona solo i valori di telemetria.
            self.X_train = data[:, :-self.config.n_predictions,1]  # Il secondo indice '1' assicura che venga presa solo la telemetria
            self.X_train = np.expand_dims(self.X_train,axis=2).astype(np.float32)
            # self.y_train (o self.y_test se train=False) sarà composto dagli ultimi n_predictions valori di ogni sequenza
            # (solo telemetria), come specificato da data[:, -self.config.n_predictions:, 1].
            self.y_train = data[:, -self.config.n_predictions:, 1]
            self.y_train = self.y_train.astype(np.float32)
            self.y_train_timestamp = data[:,-self.config.n_predictions, :]
        elif type == "test":
            self.X_test = data[:, :-self.config.n_predictions, 1]
            self.X_test =  np.expand_dims(self.X_test,axis=2).astype(np.float32)
            self.y_test = data[:, -self.config.n_predictions:, 1]
            self.y_test = self.y_test.astype(np.float32)
            self.y_test_timestamp = data[:, -self.config.n_predictions, :]
        elif type == "calibration":
            self.X_calibration = data[:, :-self.config.n_predictions, 1]
            self.X_calibration = np.expand_dims(self.X_calibration,axis=2).astype(np.float32)
            self.y_calibration = data[:, -self.config.n_predictions:, 1]
            self.y_calibration = self.y_calibration.astype(np.float32)

    def prepare_data(self):
        """
        Carica i dati di addestramento e test da locale.
        """
        try:
            self.dataset_channel = self.extract_specific_channel_data()

            self.train = self.dataset_channel['train'].values
            self.test = self.dataset_channel['test'].values
            if self.dataset_calibration is not None and len(self.dataset_calibration)>0:
                self.calibration = self.dataset_calibration[['time', self.id]].values
                self.shape_data(self.calibration, "calibration")

            self.shape_data(self.train,"train")
            self.shape_data(self.test, "test")

        except FileNotFoundError as e:
            logger.critical(e)
            logger.critical("Dati di origine non trovati, potrebbe essere necessario aggiungere dati al repository.")

    def extract_specific_channel_data(self, ):
            """
            Estrae le colonne 'time' e 'channel_name' per ogni sample rate sotto una data specifica.

            Args:
                dataset (dict): Il dataset annidato.
                folder_date_name (str): La chiave per la data specifica nel dataset.
                channel_name (str): Il nome della colonna (canale) da estrarre insieme alla colonna 'time'.

            Returns:
                dict: Un dizionario con chiave il sample_rate e valore un DataFrame con le colonne 'time' e 'channel_name'.
            """

            assert 'time' in self.train.columns and self.id in self.train.columns and 'time' in self.test.columns and self.id in self.test.columns

            return {'train': self.train[['time', self.id]], 'test': self.test[['time', self.id]]}


