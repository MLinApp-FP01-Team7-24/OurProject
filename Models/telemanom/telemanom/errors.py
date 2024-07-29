import numpy as np
import pandas as pd
import more_itertools as mit
import os
import telemanom.helpers as helpers

logger = helpers.getLogger()


class Errors:
    def __init__(self, channel, config, run_id):
        """
        Inizializza la classe Errors per gestire e analizzare gli errori di predizione in serie temporali,
        focalizzandosi in particolare sulla rilevazione di anomalie. Utilizza i dati di test e le predizioni
        generate da un modello LSTM per identificare discrepanze significative.

        Args:
            channel (obj): Oggetto della classe Channel che contiene i dati di allenamento/test (X, y)
                           per un singolo canale, dove y_test è il valore reale e y_hat è il valore predetto.
            config (obj): Oggetto di configurazione contenente i parametri come dimensione del batch,
                          dimensione della finestra per il calcolo e percentuale di lisciamento.
            run_id (str): Identificatore univoco (solitamente una data e ora) per il set di predizioni in uso.

        Attributes:
            config (obj): Configurazione importata da un file YAML esterno, usata per controllare vari aspetti
                          del processo di analisi degli errori.
            window_size (int): Numero di batch di dati precedenti considerati per il calcolo di una singola finestra
                               di errore.
            n_windows (int): Numero di finestre totali calcolate dai valori di test del canale.
            i_anom (arr): Indici delle anomalie individuate nei valori di test del canale.
            E_seq (arr of tuples): Array di tuple contenenti gli indici di inizio e fine di sequenze continue
                                   di anomalie rilevate nel dataset di test.
            anom_scores (arr): Punteggio relativo alla gravità di ciascuna sequenza di anomalie identificata in E_seq.
            e (arr): Array contenente gli errori grezzi di predizione (differenza assoluta tra predetto e reale).
            e_s (arr): Array contenente gli errori di predizione dopo il lisciamento esponenziale.
            normalized (arr): Array di errori di predizione normalizzati come percentuale dell'intervallo
                              dei valori osservati nel canale.
        """

        self.config = config
        self.window_size = self.config.window_size
        self.n_windows = int((channel.y_test.shape[0] -
                              (self.config.batch_size * self.window_size))
                             / self.config.batch_size)
        self.i_anom = np.array([])
        self.E_seq = []
        self.anom_scores = []

        # Calcolo dell'errore di predizione grezzo
        self.e = [abs(y_h-y_t[0]) for y_h, y_t in
                  zip(channel.y_hat, channel.y_test)]

        # Verifica della corrispondenza tra lunghezze delle predizioni e dei valori reali
        if not len(channel.y_hat) == len(channel.y_test):
            raise ValueError('len(y_hat) != len(y_test): {}, {}'
                             .format(len(channel.y_hat), len(channel.y_test)))


        # Applicazione del lisciamento esponenziale agli errori per mitigare i picchi di errore.
        # Il lisciamento esponenziale, o EWMA (Exponentially Weighted Moving Average), è un tipo di media mobile
        # che attribuisce pesi maggiori ai dati più recenti. Questo è particolarmente utile in serie temporali
        # dove i cambiamenti improvvisi nei valori possono portare a picchi di errore, tipici nelle previsioni
        # basate su LSTM. I picchi possono risultare da variazioni brusche nei dati che sono difficili da prevedere
        # accuratamente per il modello.
        # La formula EWMA utilizza un 'span' o finestra di lisciamento definita come una frazione del prodotto
        # della dimensione del batch e della dimensione della finestra di errore, configurata tramite il parametro
        # 'smoothing_perc'. Questo processo produce un array 'e_s' di errori smussati, dove ogni valore di errore
        # è la media ponderata dei valori precedenti con un enfasi maggiore sui valori più recenti, rendendo il
        # vettore degli errori meno sensibile ai singoli picchi e più riflessivo delle tendenze sostenute nel tempo.
        smoothing_window = int(self.config.batch_size * self.config.window_size * self.config.smoothing_perc)
        self.e_s = pd.DataFrame(self.e).ewm(span=smoothing_window) \
            .mean().values.flatten()

        # Salvataggio degli errori lisciati per ulteriori analisi
        np.save(helpers.get_correct_path(os.path.join('trained_models/telemanom', run_id, 'smoothed_errors', '{}.npy'
                             .format(channel.id))),
                np.array(self.e_s))

        # Normalizzazione degli errori di predizione rispetto all'intervallo dei valori del canale
        self.normalized = np.mean(self.e / np.ptp(channel.y_test))
        logger.info("normalized prediction error: {0:.2f}"
                    .format(self.normalized))


    def adjust_window_size(self, channel):
        """
        Decrease the historical error window size (h) if number of test
        values is limited.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        while self.n_windows < 0:
            self.window_size -= 1
            self.n_windows = int((channel.y_test.shape[0]
                                 - (self.config.batch_size * self.window_size))
                                 / self.config.batch_size)
            if self.window_size == 1 and self.n_windows < 0:
                raise ValueError('Batch_size ({}) larger than y_test (len={}). '
                                 'Adjust in config.yaml.'
                                 .format(self.config.batch_size,
                                         channel.y_test.shape[0]))

    def merge_scores(self):
        """
        If anomalous sequences from subsequent batches are adjacent they
        will automatically be combined. This combines the scores for these
        initial adjacent sequences (scores are calculated as each batch is
        processed) where applicable.
        """

        merged_scores = []
        score_end_indices = []

        for i, score in enumerate(self.anom_scores):
            if not score['start_idx']-1 in score_end_indices:
                merged_scores.append(score['score'])
                score_end_indices.append(score['end_idx'])

    def process_batches(self, channel):
        """
        Top-level function for the Error class that loops through batches
        of values for a channel.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        self.adjust_window_size(channel)

        for i in range(0, self.n_windows+1):
            prior_idx = i * self.config.batch_size
            idx = (self.config.window_size * self.config.batch_size) \
                  + (i * self.config.batch_size)
            if i == self.n_windows:
                idx = channel.y_test.shape[0]

            # 3.2 Errors and Smoothing.
            window = ErrorWindow(channel, self.config, prior_idx, idx, self, i)

            # 3.2 Threshold Calculation and Anomaly Scoring.
            window.find_epsilon()
            window.find_epsilon(inverse=True)

            #3.2 Threshold Calculation and Anomaly Scoring.
            window.compare_to_epsilon(self)
            window.compare_to_epsilon(self, inverse=True)

            if len(window.i_anom) == 0 and len(window.i_anom_inv) == 0:
                continue

            # applica la potatura (3.3 Paper)
            window.prune_anoms()
            window.prune_anoms(inverse=True)

            if len(window.i_anom) == 0 and len(window.i_anom_inv) == 0:
                continue

            window.i_anom = np.sort(np.unique(
                np.append(window.i_anom, window.i_anom_inv))).astype('int')
            window.score_anomalies(prior_idx)

            # update indices to reflect true indices in full set of values
            self.i_anom = np.append(self.i_anom, window.i_anom + prior_idx)
            self.anom_scores = self.anom_scores + window.anom_scores

        if len(self.i_anom) > 0:
            # group anomalous indices into continuous sequences
            groups = [list(group) for group in
                      mit.consecutive_groups(self.i_anom)]
            self.E_seq = [(int(g[0]), int(g[-1])) for g in groups
                          if not g[0] == g[-1]]

            # additional shift is applied to indices so that they represent the
            # position in the original data array, obtained from the .npy files,
            # and not the position on y_test (See PR #27).
            self.E_seq = [(e_seq[0] + self.config.l_s,
                           e_seq[1] + self.config.l_s) for e_seq in self.E_seq]

            self.merge_scores()


class ErrorWindow:
    def __init__(self, channel, config, start_idx, end_idx, errors, window_num):
        """
        Inizializza una finestra specifica per l'analisi degli errori di predizione, comprese le operazioni di calcolo
        delle soglie per rilevare anomalie, la potatura delle anomalie non significative e la valutazione delle sequenze
        anomale identificate. Questa classe gestisce anche gli errori invertiti, che sono calcolati invertendo gli errori
        rispetto alla loro media. L'inversione degli errori aiuta a identificare le significative diminuzioni di valore,
        che possono essere anch'esse considerate comportamenti anomali. Gli errori invertiti sono utili per rilevare
        situazioni in cui il modello supera significativamente le aspettative, il che può indicare potenziali problemi
        o cambiamenti nel comportamento del sistema monitorato.

        Args:
            channel (obj): Oggetto della classe Channel che contiene i dati di allenamento/test per X, y per un singolo canale.
            config (obj): Oggetto Config contenente i parametri di elaborazione.
            start_idx (int): Indice di inizio della finestra all'interno dell'insieme completo dei valori di test del canale.
            end_idx (int): Indice di fine della finestra all'interno dell'insieme completo dei valori di test del canale.
            errors (obj): Oggetto della classe Errors contenente tutti gli errori calcolati precedentemente.
            window_num (int): Numero corrente della finestra all'interno dei valori di test del canale.

        Attributes:
            i_anom (arr): Indici delle anomalie rilevate nella finestra per errori normali.
            i_anom_inv (arr): Indici delle anomalie rilevate nella finestra per errori invertiti.
            E_seq (arr of tuples): Sequenze di indici (inizio, fine) che identificano le anomalie continue negli errori normali.
            E_seq_inv (arr of tuples): Sequenze di indici (inizio, fine) che identificano le anomalie continue negli errori invertiti.
            non_anom_max (float): Il valore massimo di errore lisciato, al di sotto della soglia epsilon, per errori normali.
            non_anom_max_inv (float): Il valore massimo di errore lisciato, al di sotto della soglia epsilon, per errori invertiti.
            config (obj): Configurazione in uso (vedi Args).
            anom_scores (arr): Punteggio relativo alla gravità di ciascuna sequenza anomala identificata.
            window_num (int): Numero della finestra corrente.
            sd_lim (int): Numero di deviazioni standard utilizzato come default per il calcolo della soglia se non ci sono vincitori.
            sd_threshold (float): Numero di deviazioni standard utilizzato per il calcolo della migliore soglia di anomalia per gli errori normali.
            sd_threshold_inv (float): Numero di deviazioni standard utilizzato per il calcolo della migliore soglia di anomalia per gli errori invertiti.
            e_s (arr): Errori di predizione lisciati esponenzialmente nella finestra.
            e_s_inv (arr): Errori di predizione invertiti e lisciati nella finestra.
            sd_e_s (float): Deviazione standard degli errori lisciati.
            mean_e_s (float): Media degli errori lisciati.
            epsilon (float): Soglia sopra la quale un errore è considerato anomalo per gli errori normali.
            epsilon_inv (float): Soglia sopra la quale un errore è considerato anomalo per gli errori invertiti.
            y_test (arr): Valori di telemetria reali per la finestra.
            sd_values (float): Deviazione standard dei valori di telemetria.
            perc_high (float): 95° percentile dei valori di telemetria.
            perc_low (float): 5° percentile dei valori di telemetria.
            inter_range (float): Intervallo tra il 95° e il 5° percentile dei valori di telemetria.
            num_to_ignore (int): Numero di valori iniziali da ignorare quando si cerca di identificare anomalie.
        """

        # Inizializzazione delle variabili per tracciare anomalie e le loro sequenze, sia per errori normali che invertiti.
        self.i_anom = np.array([])
        self.E_seq = np.array([])
        self.non_anom_max = -1000000
        self.i_anom_inv = np.array([])
        self.E_seq_inv = np.array([])
        self.non_anom_max_inv = -1000000

        # Salvataggio della configurazione e assegnazione del numero della finestra.
        self.config = config
        self.anom_scores = []
        self.window_num = window_num

        # Impostazione del limite standard per il calcolo della soglia.
        self.sd_lim = 3.0  #12.0 (original) Valore di default per il limite di deviazione standard. Messo momentaneamente il valore 3.0 che è un valore che comunemente usato in statistica per catturare la maggior parte dei dati in una distribuzione normale
                            # 3.0: Questo valore è spesso utilizzato per identificare outlier in una distribuzione normale, poiché copre circa il 99.7% dei dati, lasciando solo lo 0.3% come potenziali outlier.
        self.sd_threshold = self.sd_lim
        self.sd_threshold_inv = self.sd_lim

        # Estrazione degli errori lisciati dalla classe Errors e calcolo degli errori invertiti.
        self.e_s = errors.e_s[start_idx:end_idx]
        self.mean_e_s = np.mean(self.e_s)
        self.sd_e_s = np.std(self.e_s)
        self.e_s_inv = np.array([self.mean_e_s + (self.mean_e_s - e) for e in self.e_s])  # Calcolo degli errori invertiti.

        # Calcolo delle soglie epsilon per rilevare anomalie, sia per errori normali che invertiti.
        # Formula dal paper : ε = μ(e_s) + z σ(e_s) , dove e_s sono gli errori smussati e z fattore di peso per la sd
        self.epsilon = self.mean_e_s + self.sd_lim * self.sd_e_s
        self.epsilon_inv = self.mean_e_s + self.sd_lim * self.sd_e_s

        # Caricamento dei valori di telemetria della finestra e calcolo delle sd(standard deviation).
        self.y_test = channel.y_test[start_idx:end_idx]
        self.sd_values = np.std(self.y_test)

        # Calcolo del 95° e del 5° percentile dei valori di telemetria per escludere gli estremi.
        # Il 95° percentile (self.perc_high) è il valore sotto cui si trova il 95% dei dati, escludendo il 5% dei valori più alti.
        # Ad esempio, se le temperature variano da 3°C, 5°C, 7°C, 8°C, 10°C, 12°C, 14°C, 15°C, 17°C, 18°C, 20°C, 22°C, e il 95° percentile è 21°C, significa che il 95% delle temperature
        # è inferiore o uguale a 21°C.
        # Il 5° percentile (self.perc_low) è il valore sotto cui si trova il 5% dei dati, escludendo il 95% dei valori più alti.
        # Se il 5° percentile è 4°C, significa che solo il 5% delle temperature è inferiore o uguale a 4°C.
        self.perc_high, self.perc_low = np.percentile(self.y_test, [95, 5])

        # L'intervallo interpercentile (self.inter_range) è calcolato come la differenza tra il 95° e il 5° percentile.
        # Questo intervallo rappresenta la variazione principale dei dati, mostrando la gamma di variazione del 90% centrale
        # dei dati. Per l'esempio delle temperature, questo intervallo sarebbe 21°C - 4°C = 17°C, indicando che la maggior parte
        # delle temperature si trova in questo range.
        # Vantaggi :
        # Riduce l'Impatto degli Outlier: Non considera i valori estremamente alti o bassi che possono essere causati da errori di misurazione o da eventi rari, non rappresentativi del normale andamento.
        # Fornisce una Misura Robusta di Variabilità: Dà una visione più realistica della dispersione dei dati, essenziale quando si analizzano variazioni o si stabiliscono soglie di normalità/anormalità.
        self.inter_range = self.perc_high - self.perc_low

        # Determinazione del numero di valori iniziali da ignorare in base alla dimensione dei dati.
        self.num_to_ignore = self.config.l_s * 2  # Ignora i valori iniziali fino ad avere abbastanza storico per l'elaborazione.
        if len(channel.y_test) < 2500:
            self.num_to_ignore = self.config.l_s
        if len(channel.y_test) < 1800:
            self.num_to_ignore = 0


    def find_epsilon(self, inverse=False):
        """
        Calcola la soglia di anomalia (epsilon) che massimizza una funzione basata sul compromesso tra
        il numero di anomalie e la riduzione della media e della deviazione standard degli errori di predizione
        quando i punti anomali vengono rimossi. Questo metodo supporta anche errori invertiti, dove gli errori
        sono calcolati come la differenza dall'errore medio invertito.

        Args:
            inverse (bool): Se True, la soglia è calcolata per gli errori invertiti.

        Dettagli dell'algoritmo:
        - Per diversi valori di 'z' (che moltiplica la deviazione standard per calcolare epsilon),
          il metodo valuta l'impatto della rimozione degli errori sopra questa soglia.
        - Calcola la percentuale di riduzione della media e della deviazione standard degli errori.
        - La soglia ottimale è quella che massimizza queste riduzioni rispetto al numero e alla lunghezza
          delle sequenze di anomalie rilevate, penalizzando configurazioni che risultano in troppe anomalie.

        Formula usata per determinare epsilon:
        ε = 𝗮𝗿𝗴𝗺𝗮𝘅(ε) = (Δμ(e_s)/μ(e_s) + Δσ(e_s)/σ(e_s)) / (|e_a| + |E_seq|^2)
        dove:
        Δμ(e_s) = μ(e_s) - μ({e_s ∈ e_s | e_s < ε})
        Δσ(e_s) = σ(e_s) - σ({e_s ∈ e_s | e_s < ε})
        |e_a| = numero di errori che superano ε
        |E_seq| = numero di sequenze continue di errori che superano ε
        """
        e_s = self.e_s if not inverse else self.e_s_inv

        max_score = -10000000 # Inizializza il punteggio massimo per la ricerca ottimale

        # Esplora diversi valori di 'z' per calcolare diverse soglie epsilon
        for z in np.arange(2.5, self.sd_lim, 0.5):
            epsilon = self.mean_e_s + (self.sd_e_s * z)

            # Filtra gli errori che sono al di sotto della soglia di epsilon.
            # Questo array 'pruned_e_s' viene usato per calcolare come cambiano la media e la deviazione standard
            # degli errori quando vengono rimossi quelli che superano la soglia.
            # È utile per valutare quanto significativi siano gli errori al di sopra della soglia rispetto a tutti gli errori.
            pruned_e_s = e_s[e_s < epsilon]  # Errori al di sotto della soglia

            # Trova gli indici degli errori che superano la soglia di epsilon.
            # Questi indici verranno usati per identificare e raggruppare le sequenze di anomalie.
            i_anom = np.argwhere(e_s >= epsilon).reshape(-1, )

            # Genera un buffer di indici attorno a ogni indice anomalo. Questo buffer è definito dalla configurazione
            # e serve a includere nei calcoli anche gli errori vicini agli errori anomali, presumendo che possano
            # essere parte di una stessa sequenza di anomalie o influenzati dagli errori maggiori.
            buffer = np.arange(1, self.config.error_buffer)
            i_anom = np.sort(np.concatenate((i_anom, np.array([i + buffer for i in i_anom]).flatten(),
                                             np.array([i - buffer for i in i_anom]).flatten())))

            # Filtra gli indici per assicurarsi che restino all'interno dei limiti validi dell'array degli errori.
            # Questo evita errori di indice fuori limite che possono verificarsi quando si aggiunge o si sottrae il buffer.
            i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]

            # Elimina gli indici duplicati che possono essere risultati dall'aggiunta dei buffer.
            # Ordinare e prendere valori unici assicura che ogni errore sia considerato una sola volta per ogni sua
            # occorrenza, mantenendo pulita la lista degli indici delle anomalie.
            i_anom = np.sort(np.unique(i_anom))

            # Verifica se ci sono indici anomali identificati
            if len(i_anom) > 0:
                # Raggruppa gli indici anomali in sequenze continue. Utilizza il modulo 'more_itertools',
                # Il modulo 'more_itertools', che offre la funzione 'consecutive_groups' per raggruppare
                # indici consecutivi. Questa funzione è utilizzata qui per identificare e raggruppare gli indici degli
                # errori che superano la soglia epsilon in sequenze continue.
                # Esempio:
                # Supponiamo che i_anom = [10, 11, 12, 15, 16, 20].
                # La funzione 'consecutive_groups' identificherà tre gruppi: [10, 11, 12], [15, 16], e [20].
                # Ogni gruppo rappresenta una sequenza di indici consecutivi, cioè errori che si presentano uno dopo l'altro.
                groups = [list(group) for group
                          in mit.consecutive_groups(i_anom)]

                # Crea una lista di tuple che rappresentano le sequenze continue di anomalie,
                # dove ogni tupla contiene l'indice di inizio e di fine di ciascuna sequenza.
                # Questo aiuta a localizzare precisamente dove le anomalie si verificano nei dati.
                E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

                # Calcola la riduzione percentuale della media degli errori. Questo calcolo mostra quanto la media
                # degli errori si riduce quando si rimuovono gli errori che superano la soglia epsilon.
                # La formula (self.mean_e_s - np.mean(pruned_e_s)) calcola la differenza tra la media degli errori originali
                # e la media degli errori che rimangono dopo aver escluso quelli superiori alla soglia.
                # Questa riduzione viene poi divisa per la media originale degli errori (self.mean_e_s) per ottenere
                # una percentuale che indica l'importanza relativa degli errori eliminati rispetto all'insieme originale.
                mean_perc_decrease = (self.mean_e_s - np.mean(pruned_e_s)) / self.mean_e_s

                # Calcola la riduzione percentuale della deviazione standard degli errori. Questo valore misura quanto
                # la dispersione (variabilità) degli errori si riduce quando si escludono quelli che superano la soglia.
                # Una riduzione significativa nella deviazione standard suggerisce che gli errori rimossi contribuivano
                # in modo sostanziale alla variabilità complessiva, indicando potenzialmente anomalie più estreme o outlier.
                sd_perc_decrease = (self.sd_e_s - np.std(pruned_e_s)) / self.sd_e_s

                # Combina le riduzioni percentuali della media e della deviazione standard in un unico 'score'.
                # Questo punteggio è poi normalizzato dividendo per la somma del quadrato del numero di sequenze di anomalie
                # e il numero totale di anomalie identificate. L'obiettivo è bilanciare il miglioramento ottenuto nella
                # riduzione degli errori rispetto al costo di identificare un numero eccessivo di punti come anomali,
                # cercando di evitare una sensibilità troppo alta che potrebbe portare a troppi falsi positivi.
                score = (mean_perc_decrease + sd_perc_decrease) / (len(E_seq) ** 2 + len(i_anom))

                # Condizioni per accettare la soglia corrente come ottimale:
                # - Il punteggio deve essere maggiore o uguale al miglior punteggio trovato finora.
                # - Il numero di sequenze di anomalie non deve superare 5, per evitare una segmentazione eccessiva.
                # La segmentazione eccessiva può ridurre l'usabilità del modello e aumentare il rischio di falsi positivi,
                # rendendo difficile per gli utenti distinguere tra anomalie reali e fluttuazioni normali.
                # Un limite di 5 sequenze aiuta a mantenere un buon equilibrio tra sensibilità e specificità,
                # concentrando l'attenzione sulle anomalie più significative
                # - Gli errori anomali non devono rappresentare più del 50% del totale degli errori, per evitare
                #   di considerare 'normali' situazioni che in realtà non lo sono.
                if score >= max_score and len(E_seq) <= 5 and len(i_anom) < (len(e_s) * 0.5):
                    max_score = score  # Aggiornamento del punteggio massimo
                    if not inverse:
                        # Imposta la nuova soglia di deviazione standard e epsilon per gli errori normali
                        self.sd_threshold = z
                        self.epsilon = self.mean_e_s + z * self.sd_e_s
                    else:
                        # Imposta la nuova soglia di deviazione standard e epsilon per gli errori invertiti
                        self.sd_threshold_inv = z
                        self.epsilon_inv = self.mean_e_s + z * self.sd_e_s

    def compare_to_epsilon(self, errors_all, inverse=False):

        """
            Confronta i valori degli errori smussati con la soglia epsilon e raggruppa gli errori consecutivi
            in sequenze. Questo metodo aiuta a identificare e valutare le sequenze di errori che potrebbero
            indicare anomalie significative.

            Args:
                errors_all (obj): Oggetto della classe Errors che contiene un elenco di tutte le anomalie
                                  precedentemente identificate nel set di test.
                inverse (bool): Se True, usa i valori degli errori invertiti per il confronto.
    """

        # Seleziona gli errori smussati normali o invertiti basati sul parametro 'inverse'.
        e_s = self.e_s if not inverse else self.e_s_inv
        # Usa la soglia epsilon appropriata in base al tipo di errore (normale o invertito).
        epsilon = self.epsilon if not inverse else self.epsilon_inv

        # Prima di procedere, verifica che gli errori siano significativi rispetto a una scala minima.
        # Questo impedisce di processare errori che sono trascurabili o troppo piccoli per essere rilevanti.
        if not (self.sd_e_s > (.05 * self.sd_values) or max(self.e_s)
                > (.05 * self.inter_range)) or not max(self.e_s) > 0.05:
            return

        # Identifica gli indici degli errori che superano sia la soglia epsilon definita
        # sia un ulteriore controllo che considera il 5% del range interpercentile.
        # Questo significa che un errore deve essere non solo superiore alla soglia epsilon,
        # ma anche significativamente più grande rispetto alla maggior parte degli errori, evitando così
        # di considerare come anomalie variazioni minori che sono normali nel contesto dei dati.
        # 'self.inter_range * 0.05' agisce come un filtro aggiuntivo per assicurarsi che solo gli errori
        # realmente significativi siano considerati, riducendo il rischio di falsi positivi in ambienti
        # con piccole fluttuazioni normali.
        i_anom = np.argwhere((e_s >= epsilon) &
                             (e_s > 0.05 * self.inter_range)).reshape(-1,)

        if len(i_anom) == 0:
            return
        # Applica un buffer agli indici anomali per catturare anche gli errori nelle immediate vicinanze
        # degli errori già identificati come anomali. Questo aiuta a identificare le sequenze di errori
        # che potrebbero essere parte della stessa anomalia.
        buffer = np.arange(1, self.config.error_buffer+1)

        # Applica un buffer agli indici anomali per catturare gli errori nelle vicinanze.
        # Consideriamo i_anom = [40, 70] e self.config.error_buffer = 3, quindi buffer = [1, 2, 3].
        # Il processo aggiunge e sottrae i valori del buffer ad ogni indice in i_anom per includere errori adiacenti:
        # Aggiunta del Buffer:
        # Per i = 40, genera [41, 42, 43]
        # Per i = 70, genera [71, 72, 73]
        # Sottrazione del Buffer:
        # Per i = 40, genera [39, 38, 37]
        # Per i = 70, genera [69, 68, 67]
        # Concatenazione e flatten trasformano le liste di liste in un unico array, e poi ordina tutti gli indici:
        # Concatena: [40, 70, 41, 42, 43, 71, 72, 73, 39, 38, 37, 69, 68, 67]
        # Ordina gli indici per assicurare l'uniformità e l'assenza di duplicati:
        # Risultato finale: [37, 38, 39, 40, 41, 42, 43, 67, 68, 69, 70, 71, 72, 73]
        # Questo array ampliato permette di considerare errori leggermente distanti dagli originali, potenzialmente parte della stessa anomalia.
        i_anom = np.sort(np.concatenate((i_anom,
                                         np.array([i + buffer for i in i_anom])
                                         .flatten(),
                                         np.array([i - buffer for i in i_anom])
                                         .flatten())))

        # Filtra gli indici per assicurarsi che restino all'interno dei limiti validi dell'array degli errori.
        # Questo evita errori di indice fuori limite che possono verificarsi quando si aggiunge o si sottrae il buffer.
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]

        # Se questa è la prima finestra temporale di analisi, è necessario ignorare gli errori iniziali.
        # Questo passaggio è importante perché all'inizio del set di dati potrebbero non esserci abbastanza
        # punti dati precedenti per formare una base solida per identificare anomalie. Ignorare gli errori iniziali
        # aiuta a evitare falsi positivi causati dalla scarsità di dati storici.
        if self.window_num == 0:
            # 'self.num_to_ignore' è una soglia definita che indica quanti valori iniziali ignorare. ( vv config.yaml row 44)
            i_anom = i_anom[i_anom >= self.num_to_ignore]

        # Per le finestre temporali successive alla prima, è importante garantire che gli errori iniziali
        # della finestra corrente non siano ignorati, poiché ora c'è una storia sufficiente da analizzare.
        # Questo controllo stabilisce che solo gli errori verso la fine della finestra corrente, dove potrebbero
        # non essere stati raccolti abbastanza dati, vengano considerati.
        else:
            # Gli indici anomali devono essere superiori al numero di elementi nell'array meno la dimensione
            # di un batch, per assicurare che la finestra corrente abbia dati sufficienti verso il suo inizio.
            i_anom = i_anom[i_anom >= len(e_s) - self.config.batch_size]

        # Una volta filtrati, gli indici anomali sono ordinati e ridotti a un insieme unico per assicurare
        # che non ci siano duplicati e che ogni anomalia sia contata una sola volta.
        i_anom = np.sort(np.unique(i_anom))

        # Calcola la posizione del batch corrente rispetto a tutti i batch moltiplicando il numero della finestra (window_num)
        # per la dimensione del batch. Questo indice rappresenta il punto di partenza del batch
        # corrente nell'array di dati globali.
        batch_position = self.window_num * self.config.batch_size

        # Genera gli indici per l'intera finestra corrente aggiungendo la posizione del batch
        # agli indici sequenziali della finestra di dati locali.
        window_indices = np.arange(0, len(e_s)) + batch_position

        # Aggiusta gli indici degli errori anomali trovati per riflettere la loro posizione corretta
        # nel dataset globale, aggiungendo la posizione del batch.
        # Quindi se prima avevamo come i_anom = [10,30] e batch_pos = 1234 --> adj_i_anom =  [1244,1264]
        adj_i_anom = i_anom + batch_position

        # Elimina gli indici già identificati come anomali in analisi precedenti dal set di indici della finestra corrente.
        # Questo aiuta a concentrarsi solo sui nuovi dati e impedisce il doppio conteggio delle anomalie.
        # Rimuove gli indici di errori già identificati come anomali dalle analisi precedenti
        # e dagli errori rilevati all'inizio di questa analisi.
        # Ciò assicura che gli errori siano conteggiati una sola volta e che la lista degli indici
        # processati rifletta solo nuove informazioni.
        window_indices = np.setdiff1d(window_indices, np.append(errors_all.i_anom, adj_i_anom))

        # Filtra gli indici per ottenere solo quelli dei valori non anomali rimanenti e trova il valore massimo tra questi.
        # Questo valore rappresenta il massimo errore ritenuto non anomalo e serve come riferimento per future decisioni.
        candidate_indices = np.unique(window_indices - batch_position)
        non_anom_max = np.max(np.take(e_s, candidate_indices))

        # Esempio: Supponiamo che e_s = [0.02, 0.03, 0.75, 0.07, 0.01, 0.20, 0.90]
        # e che epsilon = 0.05. Dopo il filtraggio, i valori come 0.75 e 0.90 sono rimossi,
        # il valore massimo tra i rimanenti (0.20) diventa non_anom_max.

        # Utilizza 'mit.consecutive_groups' per raggruppare gli indici anomali in sequenze continue.
        # Questo passaggio è critico per identificare segmenti di dati consecutivi che superano la soglia di anomalia,
        # indicando un comportamento persistente o ricorrente che devia dalla norma.
        groups = [list(group) for group in mit.consecutive_groups(i_anom)]
        E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

        # Esempio: Se i_anom = [10, 11, 12, 20, 21], questo produce E_seq = [(10, 12), (20, 21)]
        # indicando due sequenze di errori consecutivi che sono potenzialmente anomale.

        # A seconda se gli errori sono normali o invertiti, assegna i risultati alle variabili appropriate.
        # Questo include gli indici anomali, le sequenze di anomalie, e il valore massimo per gli errori non anomali,
        # che possono essere utilizzati per ulteriori analisi o per stabilire nuove soglie di allerta.
        if inverse:
            self.i_anom_inv = i_anom
            self.E_seq_inv = E_seq
            self.non_anom_max_inv = non_anom_max
        else:
            self.i_anom = i_anom
            self.E_seq = E_seq
            self.non_anom_max = non_anom_max

    def prune_anoms(self, inverse=False):
        """
        Applica una potatura alle sequenze di anomalie identificate per rimuovere quelle che non mostrano
        una separazione significativa dall'errore massimo della sequenza successiva. Questo aiuta a ridurre
        i falsi positivi mantenendo solo le anomalie significative.

        Args:
            inverse (bool): Se vero, la procedura viene applicata agli errori invertiti.
        """

        # Seleziona la sequenza di anomalie e i valori di errore adatti in base alla modalità (normale o invertita).
        E_seq = self.E_seq if not inverse else self.E_seq_inv
        e_s = self.e_s if not inverse else self.e_s_inv
        non_anom_max = self.non_anom_max if not inverse else self.non_anom_max_inv

        # Se non ci sono sequenze di anomalie, non c'è nulla da potare.
        if len(E_seq) == 0:
            return

        # Calcola i valori massimi per ogni sequenza di anomalie. Infatti viene estratto il valore delle telemtria al indice inizio telemetria errore e fine
        # E_seq_max contiene il valore massimo di errore per ogni sequenza, estratto usando gli indici di inizio e fine di ciascuna sequenza.
        E_seq_max = np.array([max(e_s[e[0]:e[1] + 1]) for e in E_seq])

        # Ordina i valori massimi in ordine decrescente per facilitare la comparazione.
        E_seq_max_sorted = np.sort(E_seq_max)[::-1]

        # Aggiunge il massimo errore non anomalo al set ordinato come riferimento.
        E_seq_max_sorted = np.append(E_seq_max_sorted, [non_anom_max])

        # Inizializza un array per tenere traccia degli indici delle sequenze da rimuovere.
        index_to_remove = np.array([])

        # Itera su ogni coppia di massimi consecutivi nel set ordinato dei massimi di errore delle sequenze.
        for i in range(0, len(E_seq_max_sorted) - 1):
            # Calcola la riduzione percentuale di errore tra due massimi consecutivi.
            reduction_percentage = (E_seq_max_sorted[i] - E_seq_max_sorted[i + 1]) / E_seq_max_sorted[i]

            # Se la riduzione non supera una soglia minima p (config.p), la sequenza è considerata non sufficientemente anomala.
            if reduction_percentage < self.config.p:
                # Aggiungi gli indici delle sequenze che corrispondono a questo massimo insufficiente.
                # Trova gli indici originali e appiattisci l'array per garantire che gli indici siano nel formato corretto.
                matching_indices = np.argwhere(E_seq_max == E_seq_max_sorted[i]).flatten()
                index_to_remove = np.append(index_to_remove, matching_indices)
            else:
                # Resetta l'array degli indici da rimuovere se la riduzione tra due valori consecutivi è sufficiente.
                index_to_remove = np.array([])

        # Assicurati che gli indici da rimuovere siano interi e ordinati in ordine inverso per una corretta rimozione.
        index_to_remove = np.unique(index_to_remove).astype(int)[::-1]

        # Elimina le sequenze di anomalie che non mostrano una riduzione sufficiente.
        if len(index_to_remove) > 0:
            E_seq = np.delete(E_seq, index_to_remove, axis=0)

        # Gestisce il caso in cui tutte le sequenze vengano eliminate.
        if len(E_seq) == 0 and inverse:
            self.i_anom_inv = np.array([])
            return
        elif len(E_seq) == 0 and not inverse:
            self.i_anom = np.array([])
            return

        # Costruisce un nuovo set di indici da mantenere basandosi sulle sequenze rimanenti.
        indices_to_keep = np.concatenate([range(e_seq[0], e_seq[-1] + 1) for e_seq in E_seq])

        # Applica una maschera per mantenere solo gli indici delle anomalie ritenute significative.
        if not inverse:
            mask = np.isin(self.i_anom, indices_to_keep)
            self.i_anom = self.i_anom[mask]
        else:
            mask_inv = np.isin(self.i_anom_inv, indices_to_keep)
            self.i_anom_inv = self.i_anom_inv[mask_inv]

    def score_anomalies(self, prior_idx):
        """
        Calculate anomaly scores based on max distance from epsilon
        for each anomalous sequence.

        Args:
            prior_idx (int): starting index of window within full set of test
                values for channel
        """

        groups = [list(group) for group in mit.consecutive_groups(self.i_anom)]

        for e_seq in groups:

            score_dict = {
                "start_idx": e_seq[0] + prior_idx + self.config.l_s,
                "end_idx": e_seq[-1] + prior_idx + self.config.l_s,
                "score": 0
            }

            score = max([abs(self.e_s[i] - self.epsilon)
                         / (self.mean_e_s + self.sd_e_s) for i in
                         range(e_seq[0], e_seq[-1] + 1)])
            inv_score = max([abs(self.e_s_inv[i] - self.epsilon_inv)
                             / (self.mean_e_s + self.sd_e_s) for i in
                             range(e_seq[0], e_seq[-1] + 1)])

            # the max score indicates whether anomaly was from regular
            # or inverted errors
            score_dict['score'] = max([score, inv_score])
            self.anom_scores.append(score_dict)