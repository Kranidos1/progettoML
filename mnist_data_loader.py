import tensorflow as tf
from hyperparameters import DataLoaderParameters


class MNISTDataLoader:
    """
    Classe responsabile del caricamento e del preprocessing del dataset MNIST.
    Permette di estrarre un sottoinsieme di dimensioni specificate (train_size, test_size)
    e di creare un set di training, test e, se richiesto, validation.

    Attributes:
        dataset_params (DataLoaderParameters): Parametri del loader (dimensioni del set di dati,
            eventuale ridimensionamento immagini, ecc.).
        x_train_full (ndarray): Dati di training (immagini).
        y_train_full (ndarray): Label di training.
        x_test_full (ndarray): Dati di test (immagini).
        y_test_full (ndarray): Label di test.
        x_validation_full (ndarray): Dati di validation (se richiesto).
        y_validation_full (ndarray): Label di validation (se richiesto).
        y_validation_oh (ndarray): Label di validation in one-hot (se richiesto).
        input_dim (int): Dimensione di input per ciascun esempio (e.g., 784 se 28x28).
        y_train_oh (ndarray): Label di training in one-hot encoding.
        y_test_oh (ndarray): Label di test in one-hot encoding.
    """

    def __init__(self, dataset_params: DataLoaderParameters):
        """
        Inizializza il data loader, carica e preprocessa il dataset MNIST
        in base ai parametri specificati.

        Args:
            dataset_params (DataLoaderParameters): Parametri per il caricamento dei dati,
                tra cui train_size, test_size, input_dim, resize_shape e num_classes.
        """
        self.dataset_params = dataset_params

        train_dataset, test_dataset = self._get_datasets(
            train_size = self.dataset_params.train_size,
            test_size = self.dataset_params.test_size
        )

        self.x_train_full, self.y_train_full = train_dataset[0], train_dataset[1]
        self.x_test_full, self.y_test_full = test_dataset[0], test_dataset[1]

        self._normalize_data()

        self.input_dim = self.dataset_params.input_dim
        if self.dataset_params.resize_shape is not None:
            self._resize_images(resize_shape = self.dataset_params.resize_shape)

        self._flatten_inputs()

        self._labels_to_one_hot_encoding(num_classes = self.dataset_params.num_classes)

        # Creazione del dataset di validazione, se previsto (validation_perc > 0)
        self.x_validation_full = None
        self.y_validation_full = None
        self.y_validation_oh = None
        if self.dataset_params.validation_perc > 0:
            validation_start = self.dataset_params.train_size - (
                (self.dataset_params.validation_perc * self.dataset_params.train_size) // 100
            )
            assert 0 < validation_start < self.dataset_params.train_size, (
                "La suddivisione per la validation è configurata in modo non corretto."
            )
            self.x_validation_full, self.y_validation_full, self.y_validation_oh = self._get_validation_dataset(
                validation_start = validation_start
            )

    def _get_validation_dataset(self, validation_start):
        """
        Separa una frazione del training set (stabilita da validation_perc) per la validazione
        e rimuove tali campioni dal training set originario.

        Args:
            validation_start (int): Indice di partenza per la creazione del set di validazione.

        Returns:
            tuple: (x_validation_full, y_validation_full, y_validation_oh) dati di validazione.
        """
        x_validation_full = self.x_train_full[validation_start:]
        y_validation_full = self.y_train_full[validation_start:]
        y_validation_oh = self.y_train_oh[validation_start:]

        self._update_train_dataset_based_on_validation(validation_start = validation_start)

        return x_validation_full, y_validation_full, y_validation_oh

    def _update_train_dataset_based_on_validation(self, validation_start):
        """
        Aggiorna il training set rimuovendo la parte assegnata alla validazione.

        Args:
            validation_start (int): Indice di inizio dei dati di validazione.
        """
        self.x_train_full = self.x_train_full[:validation_start]
        self.y_train_full = self.y_train_full[:validation_start]
        self.y_train_oh = self.y_train_oh[:validation_start]

    def _get_datasets(self, train_size, test_size):
        """
        Carica il dataset MNIST e ne estrae subset per training e test.

        Args:
            train_size (int): Numero di campioni desiderati per il training set.
            test_size (int): Numero di campioni desiderati per il test set.

        Returns:
            tuple: (train_dataset, test_dataset), cioè ((x_train, y_train), (x_test, y_test))
                   con le dimensioni richieste.
        """
        train_dataset, test_dataset = tf.keras.datasets.mnist.load_data()
        return self._get_subsets_based_on_size(
            train_dataset = train_dataset,
            test_dataset = test_dataset,
            train_size = train_size,
            test_size = test_size
        )

    def _get_subsets_based_on_size(self, train_dataset, test_dataset, train_size, test_size):
        """
        Riduce il training set e il test set alle dimensioni desiderate.

        Args:
            train_dataset (tuple): Intero training set MNIST (x_train, y_train).
            test_dataset (tuple): Intero test set MNIST (x_test, y_test).
            train_size (int): Numero di campioni da usare per il training.
            test_size (int): Numero di campioni da usare per il test.

        Returns:
            tuple: (x_train_reduced, y_train_reduced), (x_test_reduced, y_test_reduced)
        """
        x_train = train_dataset[0][:train_size]
        y_train = train_dataset[1][:train_size]

        x_test = test_dataset[0][:test_size]
        y_test = test_dataset[1][:test_size]

        return (x_train, y_train), (x_test, y_test)

    def _normalize_data(self):
        """
        Normalizza le immagini (portando i valori di pixel da [0,255] a [0,1])
        e poi standardizza sottraendo la media e dividendo per la deviazione standard
        calcolate sul training set. La stessa trasformazione viene applicata al test set.
        """
        self.x_train_full = self.x_train_full.astype("float32") / 255.0
        self.x_test_full = self.x_test_full.astype("float32") / 255.0

        mean = self.x_train_full.mean()
        std = self.x_train_full.std()

        self.x_train_full = (self.x_train_full - mean) / std
        self.x_test_full = (self.x_test_full - mean) / std

    def _resize_images(self, resize_shape):
        """
        Ridimensiona le immagini del training e del test alla dimensione specificata.

        Args:
            resize_shape (tuple): (height, width) desiderati per le nuove immagini.
        """
        self.x_train_full = tf.image.resize(
            self.x_train_full[..., tf.newaxis], resize_shape
        ).numpy()
        self.x_test_full = tf.image.resize(
            self.x_test_full[..., tf.newaxis], resize_shape
        ).numpy()

        self.input_dim = resize_shape[0] * resize_shape[1]

    def _flatten_inputs(self):
        """
        Appiattisce ogni immagine 2D in un vettore 1D. Ad esempio, un'immagine 28x28 diventa
        un vettore di lunghezza 784.
        """
        self.x_train_full = self.x_train_full.reshape((self.x_train_full.shape[0], -1))
        self.x_test_full = self.x_test_full.reshape((self.x_test_full.shape[0], -1))

    def _labels_to_one_hot_encoding(self, num_classes):
        """
        Converte le etichette (0..9) in one-hot encoding, producendo un vettore
        di dimensione 'num_classes' con tutti 0 tranne un 1 nella posizione della classe.

        Args:
            num_classes (int): Numero di classi totali (es. 10 per MNIST).
        """
        self.y_train_oh = tf.keras.utils.to_categorical(self.y_train_full, num_classes)
        self.y_test_oh = tf.keras.utils.to_categorical(self.y_test_full, num_classes)
