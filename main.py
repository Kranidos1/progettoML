import tensorflow as tf
import numpy as np
from mnist_data_loader import MNISTDataLoader
from model import MLPModel
from settings import DataLoaderParameters, TrainingParameters, RPropParameters, ModelParameters

if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU disponibile e configurata.")
    else:
        print("GPU non trovata. Eseguo su CPU.")


    # Inizializzazione delle classi di parametri
    data_loader_params = DataLoaderParameters()
    training_params = TrainingParameters()
    rPropParameters = RPropParameters()
    #model_params = ModelParameters()

    data_loader = MNISTDataLoader(dataset_params = data_loader_params)

    # Imposta il seed per garantire la riproducibilità
    tf.random.set_seed(training_params.seed)
    np.random.seed(training_params.seed)

    # Ciclo per sperimentare con diverse dimensioni del livello nascosto
    for n_hidden in training_params.hidden_neurons_list:

        # Creazione del modello con i parametri del modello e il numero di neuroni nascosti
        model = MLPModel(
            data_loader = data_loader,
            hidden_layer_neurons = n_hidden,
            data_loader_params = data_loader_params,
            training_params = training_params,
            rprop_params = rPropParameters,
        )
        test_acc, confusion_matrix, num_epoch = model.train_and_evaluate_model()
