import os
import numpy as np
from datetime import datetime as dt
import logging
from sklearn.decomposition import PCA
from telemanom.helpers import Config
from telemanom.errors import Errors
import telemanom.helpers as helpers
from telemanom.channel import Channel
from telemanom.modeling import Model
import pandas as pd
from types import SimpleNamespace
import pytz

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Configurazione del logger
logger = helpers.getLogger()


class Anomaly_Detector:
    def __init__(self, dataset_labels=None, result_path='results/', config_path='config.yaml', dataset=None,
                 date_prediction="20220811", args=None,dataset_calibration = None):
        """
        Classe principale per eseguire il rilevamento delle anomalie su un gruppo di canali
        con valori memorizzati in file .npy. Valuta anche le prestazioni rispetto a un
        set di etichette se fornito.

        Args:
            labels_path (str): percorso del file .csv contenente intervalli di anomalie etichettate
            result_path (str): directory dove salvare i risultati in formato .csv
            config_path (str): percorso del file config.yaml
            normal_path (str): percorso della directory dei dati normali
            collision_path (str): percorso della directory dei dati di collisione

        Attributes:
            labels_path (str): vedi Args
            results (list of dicts): contiene i risultati per ogni canale
            result_df (dataframe): risultati convertiti in dataframe pandas
            chan_df (dataframe): contiene tutte le informazioni sui canali dal file labels .csv
            result_tracker (dict): se sono fornite etichette, tiene traccia dei risultati durante l'elaborazione
            config (obj): oggetto Config che contiene i dati di train/test per un singolo canale
            y_hat (arr): valori del canale previsti
            id (str): identificatore datetime per tracciare esecuzioni diverse
            result_path (str): vedi Args
            normal_path (str): vedi Args
            collision_path (str): vedi Args
        """

        self.dataset_labels = dataset_labels
        self.dataset = dataset
        self.date_prediction = date_prediction
        self.dataset_calibration = dataset_calibration
        self.results = []
        self.result_df = None
        self.chan_df = self.verify_dataset_columns()
        self.args = args

        self.result_tracker = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }

        self.config = Config(helpers.get_correct_path(config_path),args)

        self.y_hat = None

        if self.config.run_id and self.config.run_id != "None":
            self.id = self.config.run_id
        else:
            self.id = dt.now(pytz.timezone('Europe/Rome')).strftime('%Y-%m-%d_%H.%M.%S')

        helpers.make_dirs(self.id,self.config)

        logger.info("Ended Setup Anomaly Detector...")

        # Aggiunta di un FileHandler al logger basato sull'ID
        hdlr = logging.FileHandler('trained_models/telemanom/logs/%s.log' % self.id)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

    def verify_dataset_columns(self):
        """
        Verifica che tutti i DataFrame all'interno di un dato livello di data abbiano lo stesso numero di colonne.
        Args:
            dataset (dict): Il dataset che contiene le date e i sample rates con DataFrame.
            date_key (str): La chiave della data specifica da verificare.

        Returns:
            list: Lista dei nomi delle colonne se tutte corrispondono.

        Raises:
            ValueError: Se i DataFrame non hanno lo stesso numero di colonne.
        """
        if self.date_prediction not in self.dataset:
            raise ValueError("The specified date is not present in the dataset.")

        data = self.dataset[self.date_prediction]
        headers = {}
        removed_columns = set()

        for dataset_type in data.keys():
            for sample_rate, df in data.get(dataset_type).items():
                headers[sample_rate] = None
                zero_value_columns = df.columns[(df == 0).all()]
                removed_columns.update(zero_value_columns)
                df.drop(zero_value_columns, axis=1, inplace=True)
                logger.info(
                    f"For {dataset_type} dataset with sample : {sample_rate} Pruned channels {list(removed_columns)} because they contain only zero values.")
                if hasattr(self, 'config') and getattr(self.config, 'pca', False):
                    logger.info("Applying PCA to retain 99% of variance.")
                    # Verifica che ci siano abbastanza colonne
                    if df.shape[1] <= 1:
                        logger.info("Not enough columns to apply PCA.")
                        raise ValueError("Not enough columns to apply PCA.")

                        # Preparazione dei dati per PCA
                    pca = PCA(n_components=0.99)  # Conserva il 99% della varianza
                    pca.fit(df)
                    transformed_data = pca.transform(df)

                headers[sample_rate] = df.columns.tolist()

        logger.info(f"Analyzing {len(df.columns) - 1} columns after pruning ... ")
        return headers

    def evaluate_sequences(self, errors, label_row):
        """
        Confronta le sequenze anomale identificate con le sequenze anomale etichettate.

        Args:
            errors (obj): oggetto Errors contenente le sequenze di anomalie rilevate per un canale
            label_row (pandas Series): contiene etichette e dettagli delle anomalie reali per un canale

        Returns:
            result_row (dict): precisione e risultati del rilevamento delle anomalie
        """

        result_row = {
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0,
            'fp_sequences': [],
            'tp_sequences': [],
            'num_true_anoms': 0
        }

        matched_true_seqs = []

        label_row['anomaly_sequences'] = eval(label_row['anomaly_sequences'])
        result_row['num_true_anoms'] += len(label_row['anomaly_sequences'])
        result_row['scores'] = errors.anom_scores

        if len(errors.E_seq) == 0:
            result_row['false_negatives'] = result_row['num_true_anoms']
        else:
            true_indices_grouped = [list(range(e[0], e[1] + 1)) for e in label_row['anomaly_sequences']]
            true_indices_flat = set([i for group in true_indices_grouped for i in group])

            for e_seq in errors.E_seq:
                i_anom_predicted = set(range(e_seq[0], e_seq[1] + 1))

                matched_indices = list(i_anom_predicted & true_indices_flat)
                valid = True if len(matched_indices) > 0 else False

                if valid:
                    result_row['tp_sequences'].append(e_seq)
                    true_seq_index = [i for i in range(len(true_indices_grouped)) if
                                      len(np.intersect1d(list(i_anom_predicted), true_indices_grouped[i])) > 0]

                    if not true_seq_index[0] in matched_true_seqs:
                        matched_true_seqs.append(true_seq_index[0])
                        result_row['true_positives'] += 1
                else:
                    result_row['fp_sequences'].append([e_seq[0], e_seq[1]])
                    result_row['false_positives'] += 1

            result_row["false_negatives"] = len(np.delete(label_row['anomaly_sequences'], matched_true_seqs, axis=0))

        logger.info('Channel Stats: TP: {}  FP: {}  FN: {}'.format(result_row['true_positives'],
                                                                   result_row['false_positives'],
                                                                   result_row['false_negatives']))

        for key, value in result_row.items():
            if key in self.result_tracker:
                self.result_tracker[key] += result_row[key]

        return result_row

    def merge_anomalies(self,anomalies_lists):
        merged_anomalies = {}

        # Itera su ciascuna lista di anomalie
        for anomaly_list in anomalies_lists:
            for anomaly in anomaly_list:
                # Crea una chiave univoca per l'anomalia basata su start_idx e end_idx
                key = (anomaly['start_idx'], anomaly['end_idx'])

                # Se la chiave esiste già, confronta i punteggi e conserva il maggiore
                if key in merged_anomalies:
                    existing_score = merged_anomalies[key]['score']
                    if anomaly['score'] > existing_score:
                        merged_anomalies[key]['score'] = anomaly['score']
                else:
                    # Se la chiave non esiste, aggiungi l'anomalia all'elenco
                    merged_anomalies[key] = anomaly

        # Converti il dizionario in una lista
        final_anomalies = sorted(merged_anomalies.values(), key=lambda x: x['start_idx'])
        return final_anomalies


    def compute_threshold_and_prun(self,anomalies_lists,data_to_show):
        precision, recall, f1_score, auroc, auprc, optimal_threshold = helpers.calculate_metrics(
            real_anomalies=pd.DataFrame(self.dataset_labels,
                                            columns=['ID',
                                                 'Timestamp-Start',
                                                 'Timestamp-End',
                                                 'Duration (ms)'])
            , predicted_anomalies=anomalies_lists
            , df_timestamp_data_channel_test=pd.DataFrame(
                [item[0] for item in data_to_show], columns=['timestamp']))

        logger.info(f"Applying pruning with dynamic threshold computed : {optimal_threshold}")
        return [anomaly for anomaly in anomalies_lists if anomaly['score'] > optimal_threshold]



    def log_final_stats(self, sample_rate=None, error_total_score=None, data_to_show=None, num_channels=0,
                        list_sample_final=[], dataset_labels=None):
        """
        Registra le statistiche finali alla fine dell'esperimento.
        """

        precision, recall, f1_score, auroc, auprc,optimal_threshold = helpers.calculate_metrics(real_anomalies=pd.DataFrame(self.dataset_labels,
                                                                                            columns=['ID',
                                                                                                     'Timestamp-Start',
                                                                                                     'Timestamp-End',
                                                                                                     'Duration (ms)'])
                                                                , predicted_anomalies=error_total_score
                                                                , df_timestamp_data_channel_test=pd.DataFrame(
                [item[0] for item in data_to_show], columns=['timestamp']))
        precision = precision * 100
        recall = recall * 100
        f1_score = f1_score * 100
        auroc = auroc * 100
        auprc = auprc * 100

        text = None
        if sample_rate:
            text = f"Final Metrics for sample rate {sample_rate} with {num_channels} different channels\nPrecision: {precision:.2f}%\nRecall: {recall:.2f}%\nF1-score: {f1_score:.2f}%\nAUROC: {auroc:.2f}%\nAUPRC: {auprc:.2f}%\n"
        elif list_sample_final and len(list_sample_final) > 0:
            text = f"Final Metrics for list sample rate {list_sample_final}\nPrecision: {precision:.2f}%\nRecall: {recall:.2f}%\nF1-score: {f1_score:.2f}%\nAUROC: {auroc:.2f}%\nAUPRC: {auprc:.2f}%\n"
        else:
            raise ValueError

        logger.info(text)

        if self.args.save_on_drive:
            helpers.saveResultsMetrics(run_id=self.id, sample_rate=sample_rate, channel_name="",
                                       precision=precision, recall=recall, auprc = auprc , auroc = auroc,
                                       f1_score=f1_score, average=True, num_channels=num_channels,
                                       num_anomalies_predict=0)

        if not self.args.skip_graphics:
            name_file_graphs_sample_rate = f"final_anomalies_prediction_sample_rate_{sample_rate}"
            path_show_graphics = os.path.join('data', self.id, 'plot_predictions',name_file_graphs_sample_rate + ".pdf")
            error_details = {
                "normalized": None,
                "anom_scores": error_total_score,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "auroc": auroc,
                "auprc":auprc
            }

            errors = SimpleNamespace(**error_details)
            helpers.show_and_save_graphic_results(path_show_graphics,None, errors, data_to_show, dataset_labels)

    def extract_dataset_from_date_and_sample_rate(self, sample_rate):

        dataset_type = ['train', 'test']

        return self.dataset.get(self.date_prediction).get(dataset_type[0]).get(sample_rate), self.dataset.get(
            self.date_prediction).get(dataset_type[1]).get(sample_rate)

    def run(self):
        """
        Avvia l'elaborazione per tutti i canali.
        """
        channel = None
        error_total = []
        error_total_score = []
        error_total_score = []
        y_test_timestamp = None
        num_channels = 0
        helpers.log_config_details(self.id,self.config)
        for sample_rate, data_sample_rate in self.dataset.get(self.date_prediction).get('test').items():
            if len(self.args.sample_rate) == 0 or (len(self.args.sample_rate) > 0 and sample_rate in self.args.sample_rate):
                logger.info(f"Processing date: {self.date_prediction}, rate: {sample_rate}")
                dataset_sample_rate_train, dataset_sample_rate_test = self.extract_dataset_from_date_and_sample_rate(
                    sample_rate=sample_rate)
                num_channels = 0
                error_total_sample_rate = []
                error_total_score_sample_rate = []
                for i, channel_name in enumerate(self.chan_df[sample_rate]):
                    if not "time" in channel_name and (not hasattr(self.config,"num_channels") or i<=self.config.num_channels):
                        logger.info(f"Processing Stream # {i}: {channel_name} , sample-rate : {sample_rate}, date : {self.date_prediction} ")
                        channel = Channel(config=self.config, chan_id=channel_name,
                                          dataset_sample_rate={'train': dataset_sample_rate_train,
                                                               'test': dataset_sample_rate_test,
                                                               "calibration" :self.dataset_calibration },
                                          dataset_labels=self.dataset_labels, sample_rate=sample_rate,
                                          date_prediction=self.date_prediction)

                        if self.config.predict:
                            model = Model(self.config, self.id, channel)
                            channel = model.batch_predict(channel)
                            helpers.saveInfoLogger(model, self.config, self.id)
                        else:
                            path_load_prediction_model = helpers.get_correct_path(os.path.join("trained_models/telemanom",self.id,'y_hat', '{}.npy'.format(channel.id)))
                            logger.info(f"Starting retrieving model prediction for the run_id : {self.id} for the channel : {channel.id} in the path : {path_load_prediction_model}")
                            channel.y_hat = np.load(path_load_prediction_model)
                            logger.info(f"Correctly ended retrieving model prediction for the run_id : {self.id} for the channel : {channel.id}")

                        errors = Errors(channel, self.config, self.id)
                        errors.process_batches(channel)
                        if y_test_timestamp is None:
                            y_test_timestamp = channel.y_test_timestamp

                        precision, recall, f1_score,auroc, auprc,optimal_threshold = helpers.calculate_metrics(
                            real_anomalies=pd.DataFrame(self.dataset_labels,columns=['ID', 'Timestamp-Start', 'Timestamp-End','Duration (ms)'])
                            , predicted_anomalies=errors.anom_scores
                            , df_timestamp_data_channel_test=pd.DataFrame(y_test_timestamp,columns=['timestamp', channel.id]))

                        num_channels += 1  # Incrementa il contatore dei canali validi
                        precision_percent = precision * 100
                        recall_percent = recall * 100
                        f1_score_percent = f1_score * 100
                        auroc_percent = auroc * 100
                        auprc_percent = auprc * 100

                        # Logga i risultati formattati
                        logger.info(
                            f"Metrics for the channel {channel_name} found n {len(errors.E_seq)} anomalies predicted\n"
                            f"Precision: {precision_percent:.2f}%\n"
                            f"Recall: {recall_percent:.2f}%\n"
                            f"F1-score: {f1_score_percent:.2f}%\n"
                            f"AUROC: {auroc_percent:.2f}%\n"
                            f"AUPRC: {auprc_percent:.2f}%")

                        helpers.saveResultsMetrics(run_id=self.id, sample_rate=sample_rate,
                                                   channel_name=channel_name, precision=precision, recall=recall,
                                                   f1_score=f1_score, auroc=auroc,auprc=auprc,average=False, num_channels=0,
                                                   num_anomalies_predict=len(errors.E_seq))

                        if not self.config.skip_graphics:
                            path_show_graphics = os.path.join('trained_models/telemanom', self.id, 'plot_predictions', sample_rate,
                                                          channel_name + ".pdf")

                            if not os.path.exists(path_show_graphics):
                                helpers.show_and_save_graphic_results(path_show_graphics, channel, errors, y_test_timestamp,
                                                              self.dataset_labels)

                        if self.config.save_on_drive:
                            helpers.saveOnDrive(self.id)

                        if len(errors.E_seq) > 0:
                            #error_total_sample_rate.append(errors.E_seq)
                            error_total_score_sample_rate.append(errors.anom_scores)
                            #logger.info(f"Step {i} error_total : {error_total_sample_rate}")
                            #merged_error_total_sample = self.merge_anomalies_seq_diff_channels(error_total_sample_rate)
                            #logger.info(f"Step {i} merged_error_total : {merged_error_total_sample}")

                if num_channels > 0 and len(error_total_score_sample_rate) > 0:
                    if not y_test_timestamp.any() and channel is not None and channel.y_test_timestamp.any():
                        y_test_timestamp = channel.y_test_timestamp

                    # logger.info(f"Final Step for sample_rate {sample_rate} error_total: {error_total_sample_rate}")
                    merged_error_total_sample_score = self.merge_anomalies(error_total_score_sample_rate)
                    if self.config.threshold:
                        merged_error_total_sample_score = self.compute_threshold_and_prun(merged_error_total_sample_score,y_test_timestamp)

                    logger.info(f"Final Prediction for sample_rate {sample_rate} final list anomalies found : {merged_error_total_sample_score}")
                    error_total_score.append(merged_error_total_sample_score)
                    self.log_final_stats(sample_rate=sample_rate, error_total_score=merged_error_total_sample_score,
                                         data_to_show=y_test_timestamp, num_channels=num_channels,
                                         dataset_labels=self.dataset_labels)

                    if self.args.save_on_drive:
                        helpers.saveOnDrive(self.id)

                logger.info(f"Ended Processing date: {self.date_prediction}, rate: {sample_rate}")

        # if len(error_total_score)>0:
        #     merged_error_total = self.merge_anomalies(error_total_score)
        #     if self.config.threshold:
        #         merged_error_total = self.compute_threshold_and_prun(merged_error_total,y_test_timestamp)
        #
        #     logger.info(f"Final Predictions with sample_rate list : {self.dataset.get(self.date_prediction).get('test').keys()} final list anomalies found : {merged_error_total}")
        #     self.log_final_stats(sample_rate=None, error_total_score=merged_error_total,
        #                          data_to_show=y_test_timestamp, num_channels=0,
        #                          list_sample_final=list(self.dataset.get(self.date_prediction).get('test').keys()),
        #                          dataset_labels=self.dataset_labels)

