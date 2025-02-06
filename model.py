import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from mnist_data_loader import MNISTDataLoader
from hyperparameters import DataLoaderParameters, TrainingParameters, RPropParameters
import time


class MLPModel:
    """
    Implementazione di un MLP (Multi-Layer Perceptron) con un singolo strato nascosto.
    Utilizza la ReLU come funzione di attivazione nel layer nascosto
    e la softmax (interna alla cross-entropy) per l'output.

    L'aggiornamento dei pesi avviene tramite RProp (Resilient Backpropagation).
    Viene inoltre impiegato un meccanismo di Early Stopping basato sui parametri
    di 'patience' e 'min_delta' forniti da TrainingParameters.
    """

    def __init__(
        self,
        hidden_layer_neurons: int,
        data_loader: MNISTDataLoader,
        data_loader_params: DataLoaderParameters,
        training_params: TrainingParameters,
        rprop_params: RPropParameters
    ):
        """
        Inizializza i parametri del modello MLP e predispone le variabili
        per l'ottimizzazione RProp e il monitoraggio delle prestazioni.

        Args:
            hidden_layer_neurons (int): Numero di neuroni nello strato nascosto.
            data_loader (MNISTDataLoader): Istanza del data loader con i dati di training, test, e validation.
            data_loader_params (DataLoaderParameters): Parametri relativi al caricamento del dataset
                (ad esempio input_dim, num_classes, ecc.).
            training_params (TrainingParameters): Parametri di training (epoche massime, early stopping, debug, ecc.).
            rprop_params (RPropParameters): Parametri di RProp (ETA_PLUS, ETA_MINUS, STEP_INIT, STEP_MIN, STEP_MAX).
        """
        self.data_loader = data_loader
        self.data_loader_params = data_loader_params
        self.training_params = training_params
        self.rprop_params = rprop_params
        self.hidden_layer_neurons = hidden_layer_neurons

        self.w1, self.b1, self.w2, self.b2 = self._initialize_parameters()
        self.params_dict = self.get_params_dict()
        self.init_history_variables()

        # Strutture dati per gestire RProp (step-size dinamico e segno ultimo gradiente)
        self.step_sizes = {}
        self.last_grad_signs = {}
        self._init_rprop_arrays()
    
    def init_history_variables(self):
        """
        Inizializza le variabili per il tracciamento della storia delle metriche
        durante l'addestramento e la validazione del modello.

        Questo metodo crea liste vuote per registrare le seguenti metriche:
        - Per il training:
            - Loss (funzione di perdita)
            - Accuratezza
            - Precisione
            - Recall
            - F1-score
        - Per la validazione:
            - Loss (funzione di perdita)
            - Accuratezza
            - Precisione
            - Recall
            - F1-score
        
        Le variabili sono utilizzate per monitorare l'evoluzione delle metriche 
        durante le epoche di addestramento e validazione.
        """
        self.loss_train_history = []
        self.acc_train_history = []
        self.loss_val_history = []
        self.acc_val_history = []
        self.precision_train_history = []
        self.recall_train_history = []
        self.f1_train_history = []
        self.precision_val_history = []
        self.recall_val_history = []
        self.f1_val_history = []


    def _init_he_weights(self, shape, name = "weight"):
        """
        Inizializza i pesi con la He Initialization (adatta a ReLU).

        Args:
            shape (tuple): (fan_in, fan_out) la forma dei pesi.
            name (str): Nome della variabile (ad es. "W1").

        Returns:
            tf.Variable: I pesi inizializzati con distribuzione normale troncata
                         secondo l'He Initialization.
        """
        fan_in = shape[0]
        stddev = tf.sqrt(2.0 / tf.cast(fan_in, tf.float32))
        return tf.Variable(
            tf.random.truncated_normal(shape, mean = 0.0, stddev = stddev),
            name = name
        )

    def _init_xavier_weights(self, shape, name = "weight"):
        """
        Inizializza i pesi con la Xavier Initialization (adatta per layer di output in softmax).

        Args:
            shape (tuple): (fan_in, fan_out) la forma dei pesi.
            name (str): Nome della variabile (ad es. "W2").

        Returns:
            tf.Variable: I pesi inizializzati con distribuzione normale troncata
                         secondo la Xavier Initialization.
        """
        fan_in, fan_out = shape
        stddev = tf.sqrt(1.0 / (fan_in + fan_out))
        return tf.Variable(
            tf.random.truncated_normal(shape, mean = 0.0, stddev = stddev),
            name = name
        )

    def _init_bias(self, shape, init_value = 0.0, name = "bias"):
        """
        Inizializza i bias con un valore costante specificato.

        Args:
            shape (tuple): Forma del vettore di bias (ad es. [num_neuroni_strato]).
            init_value (float): Valore iniziale dei bias.
            name (str): Nome della variabile (ad es. "b1").

        Returns:
            tf.Variable: Bias inizializzati a init_value.
        """
        return tf.Variable(
            tf.constant(init_value, shape = shape, dtype = tf.float32),
            name = name
        )

    def _initialize_parameters(self):
        """
        Inizializza i pesi e i bias del modello:
          - w1, b1 per input -> hidden
          - w2, b2 per hidden -> output

        Returns:
            tuple: (w1, b1, w2, b2) tutti tf.Variable.
        """
        w1 = self._init_he_weights(
            shape = [self.data_loader_params.input_dim, self.hidden_layer_neurons],
            name = "W1"
        )
        b1 = self._init_bias(
            [self.hidden_layer_neurons],
            init_value = 0.01,
            name = "b1"
        )

        w2 = self._init_xavier_weights(
            shape = [self.hidden_layer_neurons, self.data_loader_params.num_classes],
            name = "W2"
        )
        b2 = self._init_bias(
            [self.data_loader_params.num_classes],
            name = "b2"
        )

        return w1, b1, w2, b2

    def forward_pass(self, x):
        """
        Esegue la forward pass del MLP:
          z1 = x * W1 + b1
          a1 = ReLU(z1)
          z2 = a1 * W2 + b2

        Args:
            x (tf.Tensor): Tensore di input, forma [batch_size, input_dim].

        Returns:
            tf.Tensor: Logits di forma [batch_size, num_classes].
        """
        z1 = tf.matmul(x, self.w1) + self.b1
        a1 = tf.nn.relu(z1)
        #logits
        z2 = tf.matmul(a1, self.w2) + self.b2
        return z2

    def cross_entropy_loss(self, logits, labels_onehot):
        """
        Calcola la cross-entropy media tra logits e label one-hot.

        Args:
            logits (tf.Tensor): Logits di forma [batch_size, num_classes].
            labels_onehot (tf.Tensor): Label in one-hot di forma [batch_size, num_classes].

        Returns:
            tf.Tensor: Valore medio della cross-entropy sul batch.
        """
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels = labels_onehot, logits = logits)
        )

    def accuracy_fn(self, logits, labels_onehot):
        """
        Calcola l'accuracy, ovvero la percentuale di predizioni corrette.

        Args:
            logits (tf.Tensor): Logits di forma [batch_size, num_classes].
            labels_onehot (tf.Tensor): Label in one-hot di forma [batch_size, num_classes].

        Returns:
            tf.Tensor: Accuratezza media (tra 0 e 1).
        """
        preds = tf.argmax(logits, axis = 1)
        labels = tf.argmax(labels_onehot, axis = 1)
        return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

    def get_confusion_matrix(self, logits, y):
        """
        Restituisce la matrice di confusione, dati i logits e le label in one-hot.

        Args:
            logits (tf.Tensor): Logits di forma [batch_size, num_classes].
            y (tf.Tensor): Label in one-hot di forma [batch_size, num_classes].

        Returns:
            tf.Tensor: Matrice di confusione di dimensione [num_classes, num_classes].
        """
        preds = tf.argmax(logits, axis = 1)
        labels = tf.argmax(y, axis = 1)
        cm = tf.math.confusion_matrix(
            labels,
            preds,
            num_classes = self.data_loader_params.num_classes
        )
        return cm

    def _init_rprop_arrays(self):
        """
        Inizializza gli array di step-size e gli array per memorizzare
        il segno dell'ultimo gradiente, necessari per l'algoritmo RProp.
        """
        for var_name, var_value in self.params_dict.items():
            self.step_sizes[var_name] = (tf.ones_like(var_value, dtype = tf.float32)
                                         * self.rprop_params.STEP_INIT)
            self.last_grad_signs[var_name] = tf.zeros_like(var_value)

    def get_params_dict(self):
        """
        Restituisce un dizionario che mappa i nomi dei parametri ("W1", "b1", ecc.)
        alle rispettive tf.Variable.

        Returns:
            dict: { "W1": w1, "b1": b1, "W2": w2, "b2": b2 }
        """
        return {
            "W1": self.w1,
            "b1": self.b1,
            "W2": self.w2,
            "b2": self.b2
        }

    def rprop_update(self, grads):
        """
        Esegue l'aggiornamento RProp dei pesi basandosi sul segno attuale del gradiente
        e sul segno del gradiente precedente.

        Args:
            grads (dict): Dizionario {nome_parametro: gradiente}.
        """
        for key in self.params_dict.keys():
            grad = grads[key]
            grad_sign = tf.sign(grad)
            prev_sign = self.last_grad_signs[key]

            # Verifica se il segno del grad rimane costante o cambia
            sign_comparison = grad_sign * prev_sign

            # Step 1: aggiorno lo step-size
            new_step_size = tf.where(
                sign_comparison > 0,
                tf.minimum(self.step_sizes[key] * self.rprop_params.ETA_PLUS,
                           self.rprop_params.STEP_MAX),
                tf.where(
                    sign_comparison < 0,
                    tf.maximum(self.step_sizes[key] * self.rprop_params.ETA_MINUS,
                               self.rprop_params.STEP_MIN),
                    self.step_sizes[key]
                )
            )

            # Step 2: calcolo e applico l'aggiornamento ai pesi
            update = -grad_sign * new_step_size
            self.params_dict[key].assign_add(update)

            # Step 3: salvo i valori aggiornati
            self.step_sizes[key] = new_step_size
            self.last_grad_signs[key] = grad_sign

    def train(self):
        """
        Esegue il training del modello per un numero massimo di epoche
        (definito in self.training_params.max_epochs).
        Se non c'è miglioramento nella validation per un numero di epoche >= patience,
        il training si interrompe (Early Stopping).

        Returns:
            int: Numero di epoche effettivamente eseguite (<= max_epochs).
        """
        best_val_loss = float('inf')
        patience_counter = 0

        # Dati di training e validation
        x_train = self.data_loader.x_train_full
        y_train = self.data_loader.y_train_oh
        x_val = self.data_loader.x_validation_full
        y_val = self.data_loader.y_validation_oh

        # Parametri early stopping
        patience = self.training_params.patience
        min_delta = self.training_params.min_delta

        for epoch in range(self.training_params.max_epochs):
            # Calcolo dei gradienti tramite GradientTape
            with tf.GradientTape() as tape:
                logits_train = self.forward_pass(x_train)
                loss_train = self.cross_entropy_loss(logits_train, y_train)

            # Ottengo i gradienti in forma di dict
            grads_tensors = tape.gradient(loss_train, list(self.params_dict.values()))
            grads_dict = dict(zip(self.params_dict.keys(), grads_tensors))

            # Aggiorno i pesi con RProp
            self.rprop_update(grads_dict)

            # Metriche su training
            acc_train = self.accuracy_fn(logits_train, y_train)

            # Metriche su validation
            logits_val = self.forward_pass(x_val)
            loss_val = self.cross_entropy_loss(logits_val, y_val)
            acc_val = self.accuracy_fn(logits_val, y_val)

            # Precision, Recall, F1 su train e validation
            precision_train, recall_train, f1_train = self.get_metrics(logits_train, y_train)
            precision_val, recall_val, f1_val = self.get_metrics(logits_val, y_val)

            # Salvataggio metriche nei rispettivi storici
            self.loss_train_history.append(loss_train.numpy())
            self.acc_train_history.append(acc_train.numpy())
            self.loss_val_history.append(loss_val.numpy())
            self.acc_val_history.append(acc_val.numpy())
            self.precision_train_history.append(precision_train)
            self.recall_train_history.append(recall_train)
            self.f1_train_history.append(f1_train)
            self.precision_val_history.append(precision_val)
            self.recall_val_history.append(recall_val)
            self.f1_val_history.append(f1_val)

            # Stampa di debug
            print_epoch = epoch + 1
            self.debug_print(
                value = (
                    f"Epoch {print_epoch}/{self.training_params.max_epochs} | "
                    f"Train Loss: {loss_train:.4f}, Train Acc: {acc_train:.4f}, "
                    f"Train Prec: {precision_train:.4f}, Train Rec: {recall_train:.4f}, Train F1: {f1_train:.4f} | "
                    f"Val Loss: {loss_val:.4f}, Val Acc: {acc_val:.4f}, "
                    f"Val Prec: {precision_val:.4f}, Val Rec: {recall_val:.4f}, Val F1: {f1_val:.4f}"
                )
            )

            # Early Stopping
            if loss_val + min_delta < best_val_loss:
                best_val_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1
                self.debug_print(value = f"Nessun miglioramento per {patience_counter} epoca/e.")

            if patience_counter >= patience:
                self.debug_print(value = f"Early stopping attivato dopo {print_epoch} epoche.")
                break

        return epoch

    def evaluate(self):
        """
        Valuta il modello sul test set. Ritorna la loss, l'accuratezza e la
        matrice di confusione corrispondenti.

        Returns:
            tuple:
                - loss_test (float): Valore della cross-entropy sul test set.
                - acc_test (float): Accuratezza sul test set (tra 0 e 1).
                - confusion_matrix (tf.Tensor): Matrice di confusione.
        """
        x_test = self.data_loader.x_test_full
        y_test = self.data_loader.y_test_oh

        logits_test = self.forward_pass(x_test)
        loss_test = self.cross_entropy_loss(logits_test, y_test)
        acc_test = self.accuracy_fn(logits_test, y_test)
        confusion_matrix = self.get_confusion_matrix(logits_test, y_test)

        return loss_test.numpy(), acc_test.numpy(), confusion_matrix

    def get_metrics(self, logits, y):
        """
        Calcola Precision, Recall e F1-score con media macro.

        Args:
            logits (tf.Tensor): Logits di forma [batch_size, num_classes].
            y (tf.Tensor): Label in one-hot di forma [batch_size, num_classes].

        Returns:
            tuple: (precision, recall, f1) come float, media macro.
        """
        preds = tf.argmax(logits, axis = 1).numpy()
        labels = tf.argmax(y, axis = 1).numpy()

        precision = precision_score(labels, preds, average = "macro", zero_division = 0)
        recall = recall_score(labels, preds, average = "macro", zero_division = 0)
        f1 = f1_score(labels, preds, average = "macro", zero_division = 0)

        return precision, recall, f1

    def get_metrics_from_confusion_matrix(self, confusion_matrix):
        """
        Calcola Precision, Recall e F1-score medi da una matrice di confusione.

        Args:
            confusion_matrix (tf.Tensor): Matrice [num_classes, num_classes]
                                          di conteggi (TP, FP, FN, TN).

        Returns:
            tuple: (precision, recall, f1) media macro.
        """
        tp = tf.cast(tf.linalg.tensor_diag_part(confusion_matrix), tf.float32)
        fp = tf.cast(tf.reduce_sum(confusion_matrix, axis = 0), tf.float32) - tp
        fn = tf.cast(tf.reduce_sum(confusion_matrix, axis = 1), tf.float32) - tp

        EPS = 1e-7
        precision = tf.reduce_mean(tp / (tp + fp + EPS)).numpy()
        recall = tf.reduce_mean(tp / (tp + fn + EPS)).numpy()
        f1 = 2.0 * (precision * recall) / (precision + recall + EPS)

        return precision, recall, f1

    def train_and_evaluate_model(self):
        """
        Esegue l'addestramento (train) e poi valuta (evaluate) il modello sul test set.
        Visualizza infine, se richiesto, i grafici e la matrice di confusione.

        Returns:
            tuple: (test_acc, confusion_matrix, num_epoch) dove:
                - test_acc (float): Accuratezza sul test set
                - confusion_matrix (tf.Tensor): Matrice di confusione sul test set
                - num_epoch (int): Numero di epoche effettivamente eseguite
        """
        start_time = time.time()
        self.debug_print("\n============================================")
        self.debug_print(f"TRAINING CON N. HIDDEN = {self.hidden_layer_neurons}")
        self.debug_print("============================================")

        epochs_to_convergence = self.train()

        test_loss, test_acc, confusion_matrix = self.evaluate()

        end_time = time.time()
        full_computation_time = end_time - start_time

        self.debug_print(value = (
            f"\nRISULTATI FINALI per n_hidden = {self.hidden_layer_neurons} : "
            f"Test Loss = {test_loss:.4f} | Test Acc = {test_acc:.4f} | Computation Time = {full_computation_time}"
        ))

        if self.training_params.show_graphics:
            self.plot_combined_results(
                confusion_matrix = confusion_matrix
            )

        return test_acc, confusion_matrix, epochs_to_convergence

    def get_train_loss_history(self):
        """Ritorna l'andamento della loss sul training set (lista di float)."""
        return self.loss_train_history

    def get_val_loss_history(self):
        """Ritorna l'andamento della loss sul validation set (lista di float)."""
        return self.loss_val_history

    def get_train_acc_history(self):
        """Ritorna l'andamento dell'accuracy sul training set (lista di float)."""
        return self.acc_train_history

    def get_val_acc_history(self):
        """Ritorna l'andamento dell'accuracy sul validation set (lista di float)."""
        return self.acc_val_history

    def get_precision_train_history(self):
        """Ritorna l'andamento della precision sul training set (lista di float)."""
        return self.precision_train_history

    def get_recall_train_history(self):
        """Ritorna l'andamento della recall sul training set (lista di float)."""
        return self.recall_train_history

    def get_f1_train_history(self):
        """Ritorna l'andamento della F1-score sul training set (lista di float)."""
        return self.f1_train_history

    def get_precision_val_history(self):
        """Ritorna l'andamento della precision sul validation set (lista di float)."""
        return self.precision_val_history

    def get_recall_val_history(self):
        """Ritorna l'andamento della recall sul validation set (lista di float)."""
        return self.recall_val_history

    def get_f1_val_history(self):
        """Ritorna l'andamento della F1-score sul validation set (lista di float)."""
        return self.f1_val_history

    def plot_loss(self):
        """
        Traccia l'andamento della funzione di perdita (loss) per il set di training e di validazione.

        Questo metodo genera un grafico che mostra la variazione della cross-entropy loss 
        durante l'addestramento della rete neurale, con l'obiettivo di monitorare la convergenza.

        """

        plt.figure(figsize = (8, 6))
        plt.plot(self.get_train_loss_history(), label="Train Loss", linewidth=2, color="blue")
        plt.plot(self.get_val_loss_history(), label="Val Loss", linewidth=2, color="red")
        plt.title("Andamento della Cross-Entropy Loss", fontsize=16)
        plt.suptitle(f"hidden layer con {self.hidden_layer_neurons} neuroni", fontsize=12)
        plt.xlabel("Epoche", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)

        plt.show()

    def plot_accuracy(self):
        """
        Traccia l'andamento dell'accuratezza per il set di training e di validazione.

        Questo metodo genera un grafico che mostra la variazione dell'accuracy 
        durante il processo di addestramento, evidenziando la capacità della rete 
        di generalizzare sui dati non visti.

        """

        plt.figure(figsize=(8, 6))
        plt.plot(self.get_train_acc_history(), label="Train Accuracy", linewidth=2, color="blue")
        plt.plot(self.get_val_acc_history(), label="Val Accuracy", linewidth=2, color="red")
        plt.title("Andamento dell'Accuracy", fontsize=16)
        plt.suptitle(f"hidden layer con {self.hidden_layer_neurons} neuroni", fontsize=12)
        plt.xlabel("Epoche", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)

        plt.show()

    def plot_confusion_matrix(self, confusion_matrix):
        """
        Traccia la matrice di confusione per valutare le prestazioni del modello.

        Questo metodo visualizza una heatmap della matrice di confusione, che mostra 
        il numero di predizioni corrette e errate per ciascuna classe. La matrice di 
        confusione è utile per individuare eventuali bias del modello su classi specifiche.

        """

        plt.figure(figsize=(8, 6))
        labels = [str(i) for i in range(self.data_loader_params.num_classes)]
        sns.heatmap(
            confusion_matrix.numpy(),
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": 12}
        )
        plt.subtitle(f"hidden layer con {self.hidden_layer_neurons} neuroni", fontsize=12)
        plt.title(f"Matrice di Confusione", fontsize=16)
        plt.xlabel("Etichette Predette", fontsize=14)
        plt.ylabel("Etichette Reali", fontsize=14)

        plt.show()


    def plot_combined_results(self, confusion_matrix):
        """
        Mostra in un'unica figura i grafici di loss (train/val), accuracy (train/val),
        precision (train/val), recall (train/val), F1 (train/val) e la matrice di confusione.

        Args:
            confusion_matrix (tf.Tensor): Matrice di confusione finale.
            
        """
        self.plot_loss()
        self.plot_accuracy()
        self.plot_confusion_matrix(confusion_matrix = confusion_matrix)



    def debug_print(self, value):
        """
        Se il parametro di debug (training_params.debug) è True,
        stampa il messaggio passato come parametro.
        """
        if self.training_params.debug:
            print(value)
