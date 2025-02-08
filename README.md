# 🧠 MNIST Classification with RProp and MLP

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-brightgreen.svg)](https://www.tensorflow.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0.2-orange.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10-red.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-13.2-lightblue.svg)](https://seaborn.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-yellow.svg)](https://scikit-learn.org/)

## Panoramica del Progetto

Questo progetto implementa una rete neurale **MLP (Multi-Layer Perceptron)** con un **singolo strato nascosto** per la classificazione del dataset **MNIST**.

Le principali caratteristiche sono:

- **📈 Ottimizzazione con RProp (Resilient Backpropagation):**  
  L'aggiornamento dei pesi viene eseguito con l'algoritmo RProp, che adatta dinamicamente lo step-size in base al segno del gradiente.

- **⏳ Early Stopping:**  
  Il training si interrompe se, per un numero definito di epoche (*patience*), la *validation loss* non migliora di un valore minimo (*min_delta*).

- **📊 Monitoraggio delle Metriche:**  
  Durante l'addestramento vengono tracciati e salvati gli andamenti di **loss, accuracy, precision, recall e F1-score** per i set di training e validazione.  
  Inoltre, viene calcolata la **matrice di confusione** sul test set.

## 📁 Struttura del Progetto

```plaintext
├── main.py                   
├── model.py                  
├── mnist_data_loader.py      
├── settings.py               
└── requirements.txt          
```
- **📌 `settings.py`**  
  Contiene le classi che gestiscono tutti i parametri configurabili del progetto:
  - **`DataLoaderParameters`**: Specifica le dimensioni dei sottoinsiemi di training, test e validazione, oltre al ridimensionamento delle immagini e al numero di classi.
  - **`TrainingParameters`**: Definisce gli iperparametri per il training (numero massimo di epoche, patience, min_delta, seed, lista delle dimensioni da sperimentare per il livello nascosto, attivazione della stampa di debug e dei grafici).
  - **`RPropParameters`**: Imposta i parametri dell’algoritmo RProp (fattori di incremento/decremento dello step-size, valore iniziale e limiti minimo/massimo).
  - **`ModelParameters`** *(solo descrittivo)*: Riporta alcune informazioni sul modello (funzioni di attivazione, metodo di inizializzazione dei pesi, ecc.).

- **📌 `mnist_data_loader.py`**  
  Gestisce il caricamento e il preprocessing del dataset MNIST:
  - Estrae sottoinsiemi di dimensioni specificate.
  - Normalizza i dati, esegue il ridimensionamento (se richiesto), appiattisce le immagini e converte le etichette in one-hot encoding.
  - Separa una porzione del training set per la validazione.

- **📌 `model.py`**  
  Contiene l’implementazione del modello MLP:
  - **Inizializzazione dei pesi** (con He e Xavier initialization).
  - **Forward pass**: Definizione della propagazione in avanti con ReLU e softmax.
  - **Calcolo della cross-entropy loss e dell’accuracy.**
  - **Implementazione dell’algoritmo RProp** per l’aggiornamento dei pesi.
  - **Calcolo delle metriche:** Precision, Recall, F1-score.
  - **Visualizzazione dei risultati:** Grafici di loss, accuracy, matrice di confusione, ecc.

- **📌 `main.py`**  
  Punto di ingresso del progetto:
  - Inizializza i parametri tramite le classi in `settings.py`.
  - Carica il dataset con `MNISTDataLoader`.
  - Esegue il ciclo di training sperimentando diverse dimensioni per il livello nascosto.
  - Valuta le prestazioni del modello e, se abilitata l'opzione, visualizza i grafici.
## 📊 Risultati
Il modello genera:

- 📈  Grafici di training e validation (Loss & Accuracy)
- ✅ Matrice di Confusione
- 📋 Report su Precision, Recall, F1-score
### Esempio di output:
```
Epoch 5/25 | Train Loss: 0.0767, Train Acc: 0.9820, Train Prec: 0.9820, Train Rec: 0.9817, Train F1: 0.9819 | Val Loss: 0.2092, Val Acc: 0.9430, Val Prec: 0.9421, Val Rec: 0.9411, Val F1: 0.9411
Nessun miglioramento per 2 epoche.
RISULTATI FINALI per n_hidden = 64 : Test Loss: 0.1653 | Test Acc: 95.04% | Computation Time: 45.6s
```
