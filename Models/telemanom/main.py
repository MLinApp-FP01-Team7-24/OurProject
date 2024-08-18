import os
import pandas as pd
import my_utils.verbose as verbose
from telemanom.anomaly_detector import Anomaly_Detector
import argparse
from telemanom.helpers import setup_dataset,get_correct_path,getLogger

# Configurazione del logging
logger = getLogger()


# Configurazione degli argomenti della riga di comando
parser = argparse.ArgumentParser(description='Rilevamento anomalie su dataset Kuka.')
parser.add_argument('--dataset_path', type=str, default='kuka_dataset',help='Percorso della directory dei dati.')
parser.add_argument('--config_path', type=str, default='Models/telemanom/config.yaml',help='Percorso del file di configurazione config.yaml.')
parser.add_argument('--sample_rate', type=str, nargs='+', default=[],help='Lista sample rate da usare, se non specificato tutti')
parser.add_argument('--run_id', type=str,default=None,help='Cerca nella directory trained_models/telemanom/{run_id} i parametri per partire')
parser.add_argument('--batch_size', type=int, help='Numero di valori da valutare in ogni batch.')
parser.add_argument('--window_size', type=int, help='Numero di batch consecutivi usati nel calcolo degli errori.')
parser.add_argument('--smoothing_perc', type=float, help='Determina la dimensione della finestra usata nello smoothing EWMA.')
parser.add_argument('--error_buffer', type=int, help='Numero di valori intorno a un errore che sono inclusi nella sequenza.')
parser.add_argument('--l_s', type=int, help='Numero minimo di telemetrie per fare un\'analisi.')
parser.add_argument('--n_predictions', type=int, help='Numero predizione per ogni timestamp')
parser.add_argument('--activation', type=str,help='Specifica il layer di attivazione del modello')
parser.add_argument('--validation_split', type=float, help='Determina la porzione del dataset di test per la validazione range (0,1)')
parser.add_argument('--dropout', type=float, help='Determina dropout per i vari layer del modello')
parser.add_argument('--weight_decay', type=float, help='L2 regularization')
parser.add_argument('--learning_rate', type=float, help='LR model')
parser.add_argument('--lstm_batch_size', type=int, help='lstm_batch_size')
parser.add_argument('--layer_LSTM', type=int, help='Quanti layer LSTM inserire nel modello')
parser.add_argument('--epochs', type=int, help='Numero epoche per il traning')
parser.add_argument('--p', type=float, help='Diminuzione percentuale minima tra gli errori massimi nelle sequenze anomale.')
parser.add_argument('--save_on_drive', action='store_true', help='Salva le info sul drive se collegato')
parser.add_argument('--skip_graphics', action='store_true', help='Evita creazione file grafico risultato')
parser.add_argument('--verbose', action='store_true', help='Mostra informazioni dettagliate sui dati.')
parser.add_argument('--train', action='store_true', help='Se parametro settato allora viene applicato il traning al modello')
parser.add_argument("--predict", action="store_true", help="Il modello è forzato a dare nuove predizioni, se parametro non presente allora usa quelle del run_id")
parser.add_argument('--num_channels', type=int, help='Numero canali da considerare')
parser.add_argument('--threshold',action="store_true", help='Se parametro presente viene calcolata una threshold dinamica alle anomalie finali')
args = parser.parse_args()

if __name__ == '__main__':

    list_samples = list(args.sample_rate)
    dataset, dataset_label,dataset_calibration = setup_dataset(get_correct_path(args.dataset_path),list_samples)

    for date_prediction in list(dataset.keys()):
        getLogger().info(f"Starting Time Series Analysis Detection with date {date_prediction}")
        getLogger().info("Starting Setup Anomaly Detector...")
        detector = Anomaly_Detector(dataset_labels=dataset_label, result_path='results/', config_path=get_correct_path(args.config_path),
                                    dataset=dataset, date_prediction=date_prediction, args=args,dataset_calibration = dataset_calibration)
        getLogger().info("Starting Anomaly Detector Analysis...")
        detector.run()

