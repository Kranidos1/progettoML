
# --- Dataset / DataLoader ---
class DataLoaderParameters:

    def __init__(self):
        # Numero di campioni nel training set. Usato per estrarre un sottoinsieme dal dataset MNIST originale.
        self.train_size = 10000  
        # Numero di campioni nel test set. Usato per valutare il modello dopo l'addestramento.
        self.test_size = 2500 
        # Percentuale del training set riservata per la validazione. Utilizzato per monitorare l'overfitting.
        self.validation_perc = 10
        # Dimensioni a cui vengono ridimensionate le immagini (altezza, larghezza). None significa 28x28 di default. 
        self.resize_shape = None
        # Dimensione dell'input. Per immagini MNIST originali, è 28x28=784.
        self.input_dim = 28 * 28
        # Numero di classi per il problema di classificazione (MNIST ha 10 cifre: da 0 a 9).
        self.num_classes = 10

# --- Training ---
class TrainingParameters:

    def __init__(self):
        
        # Seed utilizzato per garantire la riproducibilità dei risultati.
        self.seed = 26
        # Lista delle dimensioni del livello nascosto sperimentate.
        self.hidden_neurons_list = [64, 128, 256, 512, 1024]
        # Numero massimo di epoche per l'addestramento.
        self.max_epochs = 25 
        # Numero di epoche senza miglioramento per attivare l'Early Stopping. 
        self.patience = 3
        # Miglioramento minimo nella validation loss richiesto per resettare il conteggio di Early Stopping.
        self.min_delta = 1e-4

        self.debug = True
        self.show_graphics = False

# --- RProp ---
class RPropParameters:

    def __init__(self):
        
        # Fattore di incremento dello step-size quando il gradiente mantiene lo stesso segno.
        self.ETA_PLUS = 1.2
        # Fattore di decremento dello step-size quando il gradiente cambia segno.
        self.ETA_MINUS = 0.5
        # Valore iniziale dello step-size per ogni parametro.
        self.STEP_INIT = 1e-2
        # Limite inferiore per lo step-size durante l'addestramento.
        self.STEP_MIN = 1e-6
        # Limite superiore per lo step-size durante l'addestramento.
        self.STEP_MAX = 50.0

# --- Modello MLP(NON UTILIZZATO, SOLO DESCRITTIVO) ---
class ModelParameters:

    def __init__(self):
        
        # Funzione di attivazione utilizzata nello strato nascosto.
        self.activation_hidden = "ReLU"
        # Funzione di perdita utilizzata per misurare l'errore tra output previsto e target.
        self.loss_function = "Cross Entropy"
        # Funzione di attivazione implicita nella cross-entropy per l'output.
        self.final_activation = "Softmax"
        # Metodo di inizializzazione dei pesi per il livello nascosto.
        self.initialization_w1 = "He Initialization"
        # Metodo di inizializzazione dei pesi per il livello di output.
        self.initialization_w2 = "Xavier Initialization"  
