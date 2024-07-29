import logging
import math
import yaml
import json
import sys
import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import shutil
import gdown
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import confusion_matrix

sys.path.append('../telemanom')
def get_correct_path(path):
    if '/content' in os.getcwd():
        return "/content/OurProject/" + path

    return path

class Config:
    """Loads parameters from config.yaml into global object

    """

    def __init__(self, path_to_config,args=None):

        if '/content' in os.getcwd():
            path_to_config = "/content/OurProject/Models/telemanom/config.yaml"

        if os.path.isfile(path_to_config):
            pass
        else:
            self.path_to_config = '../{}'.format(path_to_config)

        with open(path_to_config, "r") as f:
            dictionary = yaml.load(f.read(), Loader=yaml.FullLoader)

        if dictionary['run_id'] != "None" or args.run_id is not None:
            run_id = dictionary['run_id'] if dictionary['run_id'] != "None" else args.run_id
            path_config_load = get_correct_path(f"trained_models/telemanom/{run_id}/info_model/info_model.txt")

            getLogger().info(f"Starting retrieving configuration for run_id : {run_id} from the path : {path_config_load}")
            self.load_config(path_config_load)
            self.run_id = run_id
            getLogger().info(f"Ended retrieving configuration for run_id : {run_id} from the path : {path_config_load}")

        for k, v in dictionary.items():
            if getattr(self,k,None) is None:
                if getattr(args, k, None) is not None:
                    setattr(self, k, getattr(args, k))
                else:
                    setattr(self, k, v)
        # Now, check for any extra keys in args that are not in the dictionary
        # and set them as attributes if they are not None
        args_dict = vars(args)  # Convert Namespace to dictionary
        for k, v in args_dict.items():
            if v is not None and not hasattr(self, k):
                setattr(self, k, v)

    def build_group_lookup(self, path_to_groupings):

        channel_group_lookup = {}

        with open(path_to_groupings, "r") as f:
            groupings = json.loads(f.read())

            for subsystem in groupings.keys():
                for subgroup in groupings[subsystem].keys():
                    for chan in groupings[subsystem][subgroup]:
                        channel_group_lookup[chan["key"]] = {}
                        channel_group_lookup[chan["key"]]["subsystem"] = subsystem
                        channel_group_lookup[chan["key"]]["subgroup"] = subgroup

        return channel_group_lookup

    def load_config(self,path_config):

        if not os.path.exists(path_config) or os.path.getsize(path_config) == 0:
            getLogger().info(f"Failed loading configuration in the path : {path_config}")
            return

        with open(path_config, "r") as file:
            content = file.read()

        start_marker = "###### CONFIGURATION ######"
        end_marker = "###### MODEL INFO ######"
        config_section = content.split(start_marker)[1].split(end_marker)[0].strip()

        # Parse the configuration section
        try:
            for line in config_section.split('\n'):
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    setattr(self, key.strip(), yaml.safe_load(value))
        except Exception as error:
            getLogger().info(f"Failed loading configuration for the file : {path_config}")


def make_dirs(_id,config):
    '''Create directories for storing data in repo (using datetime ID) if they don't already exist'''
    starting_path = get_correct_path("")
    if not config.train or not config.predict:
        if not os.path.isdir(starting_path+'trained_models/telemanom/%s' % _id):
            if not os.path.isdir(starting_path+'trained_models/telemanom/'):
                os.mkdir(starting_path+"trained_models/telemanom/")

            os.mkdir(starting_path+'trained_models/telemanom/%s' % _id)

    paths = [starting_path+'trained_models/telemanom/', starting_path+'trained_models/telemanom/%s' % _id, starting_path+'trained_models/telemanom/logs',
             starting_path+'trained_models/telemanom/%s/models' % _id, starting_path+'trained_models/telemanom/%s/smoothed_errors' % _id,
             starting_path+'trained_models/telemanom/%s/y_hat' % _id, starting_path+'trained_models/telemanom/%s/info_model' % _id,  starting_path+'trained_models/telemanom/%s/results' % _id]

    for p in paths:
        if not os.path.isdir(p):
            os.mkdir(p)


def getLogger():
    '''Configure logging object to track parameter settings, training, and evaluation.

    Args:
        config(obj): Global object specifying system runtime params.

    Returns:
        logger (obj): Logging object
        _id (str): Unique identifier generated from datetime for storing data/models/results
    '''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logger = logging.getLogger('telemanom')

    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        stdout = logging.StreamHandler(sys.stdout)
        stdout.setLevel(logging.INFO)
        logger.addHandler(stdout)

    return logger


def split_data_by_time(data):
    """
    Divides the data into sections where each section ends when the time gap
    to the next data point exceeds 2 minutes.

    :param data: DataFrame with columns ['timestamp', 'value']
    :return: A list of DataFrames, each containing continuous time sections
    """
    sections = []
    current_section = []

    # Converte la colonna 'timestamp' in datetime se non lo è già
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    for i in range(len(data) - 1):
        current_section.append(data.iloc[i])

        # Calcola la differenza di tempo in minuti
        time_diff = (data.iloc[i + 1]['timestamp'] - data.iloc[i]['timestamp']).total_seconds() / 60

        # Se la differenza supera 2 minuti, termina la sezione corrente
        if time_diff > 2:
            sections.append(pd.DataFrame(current_section))
            current_section = []

    # Aggiungi l'ultima sezione se non vuota
    if current_section:
        sections.append(pd.DataFrame(current_section))

    return sections

def log_config_details(run_id=None,config=None,header="",body="",footer=""):

    logger = getLogger()
    header = "#### Configuration Details ####"
    message_parts = [header]

    for attr_name, attr_value in vars(config).items():
        message_parts.append(f"{attr_name} = {attr_value}")

    footer = f"Please for more info about the model and the configuration see : data/{run_id}/info_model/info_model.txt\n####End of Configuration Details ####\n"
    message_parts.append(footer)

    formatted_message = "\n".join(message_parts)
    logger.info(formatted_message)

def highlight_anomalies(ax, anomalies):
    # Controlla se l'etichetta "Anomalia" è già nella legenda
    existing_labels = [text.get_text() for text in ax.get_legend().get_texts()] if ax.get_legend() else []
    for _, row in anomalies.iterrows():
        start = pd.to_datetime(row['Timestamp-Start'])
        end = pd.to_datetime(row['Timestamp-End'])
        label = 'Anomalia' if 'Anomalia' not in existing_labels else None
        ax.axvspan(start, end, color='purple', alpha=0.3, label=label)
        if label:  # Aggiunge l'etichetta alla lista per evitare duplicazioni future
            existing_labels.append(label)

def highlight_predicted_anomalies(ax, data_to_show, anomalies_scores, segment):
    # Assicurati di ottenere tutte le etichette attualmente presenti nella legenda
    existing_labels = [text.get_text() for text in ax.get_legend().get_texts()] if ax.get_legend() else []

    for anomaly in anomalies_scores:
        start_idx = anomaly['start_idx'] - 1
        end_idx = anomaly['end_idx']
        score = anomaly['score']
        # Verifica che l'intervallo dell'anomalia intersechi il segmento corrente
        if start_idx <= segment.index[-1] and end_idx >= segment.index[0]:
            # Adegua gli indici se l'anomalia inizia prima o finisce dopo il segmento corrente
            adjusted_start_idx = max(start_idx, segment.index[0])
            adjusted_end_idx = min(end_idx, segment.index[-1])

            # Ottieni i timestamp corrispondenti agli indici adeguati
            start_timestamp = data_to_show['timestamp'].iloc[data_to_show.index.get_loc(adjusted_start_idx)]
            end_timestamp = data_to_show['timestamp'].iloc[data_to_show.index.get_loc(adjusted_end_idx)]

            labelAnomalyPrediction = 'Anomalia Predetta' if 'Anomalia Predetta' not in existing_labels else None
            # Evidenzia l'anomalia nel grafico con la gestione appropriata dell'etichetta
            ax.axvspan(start_timestamp, end_timestamp, color='gold', alpha=0.5, label=labelAnomalyPrediction)

            # Aggiungi il testo con lo score sopra l'intervallo evidenziato
            mid_point = start_timestamp + (end_timestamp - start_timestamp) / 2
            vertical_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2  # Calcola il centro verticale del grafico
            ax.text(mid_point, vertical_center, f'{score:.2f}', color='red', fontsize=25, ha='center', va='center')

            # Dopo il primo uso, assicurati di non aggiungere l'etichetta di nuovo
            if labelAnomalyPrediction:
                existing_labels.append(labelAnomalyPrediction)


def custom_x_asses(x, pos):
    return pd.to_datetime(mdates.num2date(x)).strftime('%H:%M:%S.%f')[:-3]


def segment_data(data, channel_id = None,min_gap_minutes=120):
    """Segmenta i dati basandosi su una soglia di gap temporale."""
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=['timestamp', channel_id])
        convert_back_to_numpy = True
    else:
        convert_back_to_numpy = False

    segments = []
    current_segment = [data.iloc[0]]

    for i in range(1, len(data)):
        current_time = pd.to_datetime(data.iloc[i]['timestamp'])
        last_time = pd.to_datetime(data.iloc[i - 1]['timestamp'])
        if (current_time - last_time).total_seconds() / 60 > min_gap_minutes:
            segments.append(pd.DataFrame(current_segment))
            current_segment = []
        current_segment.append(data.iloc[i])

    if current_segment:
        segments.append(pd.DataFrame(current_segment))

    if convert_back_to_numpy:
        numpy_segments = [seg.to_numpy() for seg in segments]
        return numpy_segments

    return segments

def calculate_metrics(real_anomalies, predicted_anomalies,df_timestamp_data_channel_test=None):

    real_intervals = [(pd.to_datetime(row['Timestamp-Start']).tz_localize(None), pd.to_datetime(row['Timestamp-End']).tz_localize(None)) for idx, row in real_anomalies.iterrows()]
    predicted_intervals = []

    #if last:
        #print(f"Last predicted_intervals : {predicted_intervals}")
    for anomaly in predicted_anomalies:
        start_idx = anomaly['start_idx']
        end_idx = anomaly['end_idx']
        score = anomaly['score']
        if start_idx < len(df_timestamp_data_channel_test) and end_idx < len(df_timestamp_data_channel_test):
            #if last:
                #print(f"Last start_idx : {start_idx} -- end_idx : {end_idx} -- len(df_timestamp_data_channel_test) : {len(df_timestamp_data_channel_test)}")
                #print(f"Last df_timestamp_data_channel_test['timestamp']: {df_timestamp_data_channel_test['timestamp']}")
                #print(f"Last df_timestamp_data_channel_test['timestamp'].iloc[start_idx]): {df_timestamp_data_channel_test['timestamp'].iloc[start_idx]}")
            start_timestamp =pd.to_datetime(df_timestamp_data_channel_test['timestamp'].iloc[start_idx]).tz_localize(None)
            end_timestamp = pd.to_datetime(df_timestamp_data_channel_test['timestamp'].iloc[end_idx]).tz_localize(None)
            predicted_intervals.append({"start" : start_timestamp, "end" : end_timestamp , "score" : score})
        else:
            print(f"Out of bounds: start_idx={start_idx}, end_idx={end_idx}, df_length={len(df_timestamp_data_channel_test)}")

    true_labels = []
    pred_scores = []
    optimal_threshold = None

    # Contare i veri positivi e i falsi negativi
    for real_start, real_end in real_intervals:
        found = False
        for anomaly in predicted_intervals:
            pred_start = anomaly['start']
            pred_end = anomaly['end']
            score = anomaly['score']

            if pred_start <= real_end and pred_end >= real_start:
                true_labels.append(1)  # True Positive
                pred_scores.append(score)
                found = True
                print(f"True Positive: Predicted anomaly from {pred_start} to {pred_end} overlaps with real anomaly from {real_start} to {real_end}")
                break

        if not found:
            true_labels.append(0)  # False Negative
            pred_scores.append(0)  # No score since it's a miss
            print(f"False Negative: No predicted anomaly overlaps with real anomaly from {real_start} to {real_end}")

    for anomaly in predicted_intervals:
        pred_start = anomaly['start']
        pred_end = anomaly['end']
        score = anomaly['score']
        if not any(real_start <= pred_end and real_end >= pred_start for real_start, real_end in real_intervals):
            true_labels.append(0)
            pred_scores.append(score)
            print(f"False Positive: Predicted anomaly from {pred_start} to {pred_end} does not overlap with any real anomaly")

    if any(pred_scores) and len(predicted_intervals)>0: # Check if there are any non-zero scores
        precision, recall, thresholds = precision_recall_curve(true_labels, pred_scores)
        fpr, tpr, roc_thresholds = roc_curve(true_labels, pred_scores)
        auroc = auc(fpr, tpr)
        if math.isnan(auroc):
            auroc = 0.0
        auprc = auc(recall, precision)
        if math.isnan(auroc):
            auprc = 0.0
        optimal_idx = np.argmax(2 * precision * recall / (precision + recall + 1e-8))
        precision = precision[optimal_idx]
        recall = recall[optimal_idx]
        optimal_threshold = thresholds[optimal_idx]
        f1_score = 2 * precision* recall / (precision + recall + 1e-8)
    else:
        # Default metrics when no predictions are available
        precision = recall = f1_score = auroc = auprc = 0.0

    return precision, recall, f1_score, auroc, auprc,optimal_threshold

def show_and_save_graphic_results(path_file="", channel=None, errors=None,y_test_timestamp=None,dataset_labels=None):
    data_to_show = pd.DataFrame(y_test_timestamp, columns=['timestamp', 'channel_id'])
    data_to_show['timestamp'] = pd.to_datetime(data_to_show['timestamp'])
    if channel:
        data_to_show = pd.DataFrame(y_test_timestamp, columns=['timestamp', channel.id])
        y_hat = channel.y_hat[:, 0] if len(channel.y_hat.shape) == 2 else channel.y_hat
        data_to_show['Predizione'] = y_hat

    segments = segment_data(data_to_show)
    num_segments = len(segments)

    # Creazione della directory se non esiste
    directory = os.path.dirname(path_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.ioff()  # Disabilita la visualizzazione interattiva delle figure
    fig, axes = plt.subplots(num_segments, 1, figsize=(150, 24 * num_segments), sharex=False)
    if num_segments == 1:  # Se c'è solo un segmento, matplotlib non ritorna un array
        axes = [axes]

    for ax, segment in zip(axes, segments):
        if channel:
            ax.plot(segment['timestamp'], segment[channel.id], label='Valore Reale', color='green', linestyle='-')
            ax.plot(segment['timestamp'], segment['Predizione'], label='Predizione', color='blue', linestyle='-')

        highlight_anomalies(ax, pd.DataFrame(dataset_labels,columns=['ID', 'Timestamp-Start', 'Timestamp-End', 'Duration (ms)']))
        highlight_predicted_anomalies(ax, data_to_show, errors.anom_scores, segment)

        ax.set_xlim(segment['timestamp'].min(), segment['timestamp'].max())
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
        ax.xaxis.set_major_formatter(FuncFormatter(custom_x_asses))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, fontsize=40)  # Aumenta la dimensione del font degli assi
        plt.setp(ax.yaxis.get_majorticklabels(), fontsize=40)

    # Legenda comune con font aumentato
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=3,fontsize=55)  # dimensione del font della legenda

    text_error = None
    if errors.normalized:
        text_error = f"Normalized Prediction Error: {errors.normalized}\nNumber of Anomalies: {len(errors.E_seq)}"
    elif errors.normalized is None and errors.precision and errors.f1_score and errors.recall and errors.auroc and errors.auprc:
        text_error = f"Number of Anomalies: {len(errors.anom_scores)} -- Precision: {errors.precision:.2f}% -- Recall: {errors.recall:.2f}% -- AUROC : {errors.auroc:.2f}% -- AUPRC : {errors.auprc:.2f}%"
    else:
        raise ValueError

    plt.figtext(0.01, 0.99, text_error,fontsize=55, verticalalignment='top')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajust layout to make room for the legend

    # Salvataggio del grafico nel PDF
    with PdfPages(path_file) as export_pdf:
        export_pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    getLogger().info(f"Graphical output saved on path : {path_file}")


def converter_dataset_to_numpy(path):
    # Carica il file CSV
    df = pd.read_csv('tuo_dataset.csv')

    # Converti il DataFrame in un array NumPy
    array = df.to_numpy()

    # Salva l'array NumPy in un file .npy
    np.save('tuo_dataset.npy', array)


def load_csv(file_path, parse_dates=False, separator=";"):
    """
    Carica un file CSV utilizzando il separatore corretto e pulisce i nomi delle colonne.

    Args:
        file_path (str): Il percorso del file da caricare.

    Returns:
        DataFrame: Un DataFrame contenente i dati del file CSV.
    """
    df = pd.read_csv(file_path, sep=separator, parse_dates=parse_dates)
    df.columns = df.columns.str.strip()  # Pulizia degli spazi nei nomi delle colonne
    return df


def extract_labels_from_excel(path_file,name_folder=""):
    """
    Estrae le etichette delle anomalie da un file Excel con uno o più fogli,
    ognuno contenente colonne per 'Inizio/Fine' e 'Timestamp'.
    """
    df = refactor_excel_labels(path_file, name_folder=name_folder)
    anomalies = []

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')

    starts = df[df['Inizio/fine'] == 'i'].reset_index(drop=True)
    ends = df[df['Inizio/fine'] == 'f'].reset_index(drop=True)

    for i, (start, end) in enumerate(zip(starts['Timestamp'], ends['Timestamp'])):
        duration = (end - start).total_seconds() * 1000  # Durata in millisecondi
        anomalies.append({
            'ID': i + 1,
            'Timestamp-Start': start.strftime('%Y-%m-%d %H:%M:%S'),
            'Timestamp-End': end.strftime('%Y-%m-%d %H:%M:%S'),
            'Duration (ms)': duration
        })

    anomalies_df = pd.DataFrame(anomalies)
    return anomalies_df


def refactor_excel_labels(file_path, name_folder):
    """
    Modifica i timestamp in un file Excel aggiungendo un offset di fuso orario e salva il file modificato.
    """
    timezone_offset = -2
    getLogger().info(f"Starting Refactor Labels Data from File : {file_path} with TZ : {timezone_offset}H")
    xls = pd.ExcelFile(file_path)
    combined_df_list = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True) + pd.Timedelta(hours=timezone_offset)
        df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        combined_df_list.append(df)

    combined_df = pd.concat(combined_df_list, ignore_index=True)
    combined_df = combined_df.sort_values(by='Timestamp').reset_index(drop=True)
    getLogger().info(f"Ended Refactor Labels.")

    return combined_df


def combination_and_check_csv(df1, df2):
    combined_df = pd.concat([df1, df2])
    assert len(combined_df) == len(df1) + len(df2), "Errore nella lunghezza della concatenazione"

    return combined_df


def custom_sort_key(filename):
    # Estrae la parte numerica dopo 'rec'
    match = re.search(r'rec(\d+)', filename)
    if match:
        rec_number = int(match.group(1))
        return rec_number
    return float('inf')  # Per i file che non corrispondono al pattern, li mettiamo alla fine


def find_latest_step_file(files):
    """
    Trova il file con il numero di step più alto tra i file dati.
    """
    max_step = -1
    latest_file = None
    for file in files:
        match = re.search(r'step(\d+)\.csv$', file)  # Cerca 'step' seguito da un numero e '.csv'
        if match:
            step_number = int(match.group(1))
            if step_number > max_step:
                max_step = step_number
                latest_file = file
    return latest_file, max_step


def download_folder_from_google_drive(folder_id, destination):
    url = f'https://drive.google.com/drive/folders/{folder_id}'
    gdown.download_folder(url, output=destination, quiet=False, use_cookies=False)

def setup_dataset(dataset_path=get_correct_path("dataset/Kuka_v1"),list_samples=[]):
    """
    Carica e organizza i dati CSV da una directory strutturata contenente dati di serie temporali per varie date e frequenze di campionamento,
    salvando uno storico di ogni aggiornamento dei DataFrame per ogni chiave di frequenza di campionamento e la prima volta anche in formato Excel.

    Args:
        dataset_path (str): Il percorso base della directory del dataset.
        output_dataset (str): Il percorso base della directory di output.

    Returns:
        dict: Un dizionario annidato dove il primo livello di chiavi è la data,
              il secondo livello di chiavi è la frequenza di campionamento, e i valori sono DataFrame pandas
              contenenti tutti i dati per quella data e frequenza di campionamento.
    """
    if not os.path.exists(dataset_path):
        print("Missing Dataset Kuka V1. Downloading it ...")
        download_folder_from_google_drive("1FHKRRy2WxnG0hUodfu-mfCEzELJp3ULs?usp=sharing",get_correct_path(""))
        assert os.path.exists(dataset_path)
        print(f"Dataset Kuka V1 Downloaded in the path {dataset_path}")

    data_map = {}
    steps = {}
    sampling_data = {}
    df_calibration = None
    df_labels = None
    getLogger().info(f"Starting retrieving Kuka_v1 dataset from the path : {dataset_path}")

    for name_folder in os.listdir(dataset_path):
        getLogger().info(f"Processing the directory {name_folder}")
        if name_folder != "Extra" and not name_folder.startswith("."):
            date_dataset = None
            path_folder = os.path.join(dataset_path, name_folder)
            files = os.listdir(path_folder)
            sorted_files = sorted(files, key=custom_sort_key)
            for file in sorted_files:
                folder_type = 'test' if 'collisions' in name_folder else 'train'
                file_path = os.path.join(path_folder, folder_type, file)

                if date_dataset is not None and name_folder == "collisions" and  f"{date_dataset}_collisions_timestamp.xlsx" in file:
                    getLogger().info(f"Starting Extracting Labels Data from File : {file} from the path : {file_path}")
                    file_path = os.path.join(dataset_path, name_folder, file)
                    df_labels = extract_labels_from_excel(file_path, date_dataset)
                    getLogger().info(f"Ended Extracting Labels Data ")
                    continue

                if date_dataset is not None and name_folder == "collisions" and file == "rec6_20220811_rbtc_0.1s.csv":
                    getLogger().info(f"Starting Extracting Dataset for Calibration from File : {file} from the path : {file_path}")
                    df_calibration = load_csv(os.path.join(path_folder, file), parse_dates=['time'])
                    getLogger().info(f"Ended Extracting Dataset for Calibration from File : {file} from the path : {file_path}")
                    continue

                sample_rate_file = file.split('_')[-1].replace('.csv', '')
                if file.endswith('.csv') and not file.endswith('metadata.csv') and not file=="dataset.csv" and (len(list_samples)==0 or (len(list_samples)>0 and sample_rate_file in list_samples )):
                    date_dataset = file_path.split("_")[-3]
                    if len(sampling_data.items())==0:
                        sampling_data = {'train': {}, 'test': {}}
                        steps = {'train': {},'test': {}}

                    getLogger().info(f"Processing file: {file}")
                    sample_rate = file.split('_')[-1].replace('.csv', '')
                    df = load_csv(os.path.join(path_folder, file), parse_dates=['time'])

                    if sample_rate not in sampling_data[folder_type]:
                        sampling_data[folder_type][sample_rate] = df
                        steps[folder_type][sample_rate] = 0
                        getLogger().info(f"Step : 0 - Sample Rate : {sample_rate}")
                    else:
                        current_df = sampling_data[folder_type][sample_rate]
                        step_count = steps[folder_type][sample_rate] + 1
                        sampling_data[folder_type][sample_rate] = combination_and_check_csv(current_df,df)
                        sampling_data[folder_type][sample_rate].sort_values('time', inplace=True)
                        sampling_data[folder_type][sample_rate].reset_index(drop=True, inplace=True)
                        steps[folder_type][sample_rate] = step_count
                        getLogger().info(f"Step : {step_count} - Sample Rate : {sample_rate}")

        data_map[date_dataset] = sampling_data

    getLogger().info(f"Ended retrieving Kuka_v1 dataset ...")
    getLogger().info("Data Collected Correctly")

    return data_map, df_labels,df_calibration

# Funzione ricorsiva per copiare solo i file mancanti
def copy_missing_files(source, target):
    if not os.path.exists(target):
        os.makedirs(target)
    for item in os.listdir(source):
        src_path = os.path.join(source, item)
        dst_path = os.path.join(target, item)
        if os.path.isdir(src_path):
            copy_missing_files(src_path, dst_path)
        else:
            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)

def saveOnDrive(name_dir=""):
    pathOriginal = f"/content/OurProject/trained_models/telemanom/{name_dir}/"
    path_original_log = f"/content/OurProject/trained_models/telemanom/logs/{name_dir}.log"
    drive = "/content/drive/MyDrive/"

    if not os.path.isdir(pathOriginal):
        print(f"Path Original is wrong : {pathOriginal}")
        return
    if not os.path.isfile(path_original_log):
        print(f"Path Logs Original is wrong : {path_original_log}")
        return
    if not os.path.isdir(drive):
        print("Drive is not linked ...")
        return

    if not os.path.isdir(drive + f"MLA/"):
        os.mkdir(drive + f"MLA/")

    destination_path = os.path.join(drive,"MLA",name_dir)
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    copy_missing_files(pathOriginal, destination_path)

    destination_path_log = f"{destination_path}/{name_dir}.log"
    shutil.copy(path_original_log, destination_path_log)


def saveInfoLogger(model,config,run_id):
    # Crea il percorso del file se non esiste
    path = get_correct_path(os.path.join('trained_models/telemanom', run_id, 'info_model'))
    os.makedirs(path, exist_ok=True)

    # Specifica il file per salvare le informazioni
    file_path = os.path.join(path, 'info_model.txt')

    # Controlla se il file esiste e ha contenuto
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return

    with open(file_path, 'w') as file:
        # Scrivi le informazioni di configurazione
        file.write("###### CONFIGURATION ######\n")
        exclude = ['path_to_config', 'dictionary', 'header','dataset_path']
        for attr, value in config.__dict__.items():
            if attr not in exclude and not attr.startswith('__') and not callable(value):
                file.write(f"{attr}: {value}\n")

        file.write("\n###### MODEL INFO ######\n")
        # Assicurati che 'self.model' sia il tuo modello PyTorch
        for name, module in model.model.named_children():
            file.write(f"{name} ({module.__class__.__name__}): {module}\n")


def saveResultsMetrics(run_id, sample_rate, channel_name, precision, recall, f1_score,auprc,auroc,average=False,num_channels=0,num_anomalies_predict=0):
    path = get_correct_path(os.path.join('trained_models/telemanom', run_id, 'results'))
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, 'results_metrics.txt')
    sample_rate_header = f"#####METRICS SAMPLE_RATE: {sample_rate}######"

    if not average:
        search_text = f"\nMetrics for the channel {channel_name} found n {num_anomalies_predict} anomalies predicted"
    else:
        search_text = f"\n\nFinal Metrics for the sample_rate {sample_rate} with {num_channels} different channels"

    content = ""
    header_exists = False
    metric_exists = False

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
            header_exists = sample_rate_header in content
            metric_exists = search_text in content

    if not average:
        text = f"\nMetrics for the channel {channel_name} found n {num_anomalies_predict} anomalies predicted\nPrecision: {precision:.2f}% --- Recall: {recall:.2f}% --- F1-score: {f1_score:.2f}% --- AUROC: {auroc:.2f}% --- AUPRC: {auprc:.2f}%\n"
    else:
        text = f"\nFinal Metrics for the sample_rate {sample_rate} with {num_channels} different channels\nPrecision: {precision:.2f}% --- Recall: {recall:.2f}% --- F1-score: {f1_score:.2f}% --- AUROC: {auroc:.2f}% --- AUPRC: {auprc:.2f}%\n"

    with open(file_path, 'a') as file:
        if not header_exists:
            if os.path.getsize(file_path) > 0:
                file.write("\n\n")
            file.write(sample_rate_header)

        if not metric_exists:
            file.write(text)