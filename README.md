# Denoising di immagini mediante deep learning: rete convoluzionale U-Net addestrata con approccio Noise2Void

Questo repository fornisce una guida dettagliata passo-passo per l’addestramento e la valutazione di una rete neurale U-Net per il denoising di immagini, utilizzando l’algoritmo Noise2Void (N2V).

Il progetto si basa sull’utilizzo di:

- [BBBC006 Dataset (Broad Bioimage Benchmark Collection)](https://bbbc.broadinstitute.org/BBBC006)

- [Kodak PhotoCD](https://www.kaggle.com/datasets/sherylmehta/kodak-dataset?select=kodim12.png)

L’implementazione dell’algoritmo Noise2Void è realizzata tramite la libreria [CAREamics](https://careamics.github.io/0.1/), progettata specificamente per applicazioni di deep learning nel contesto dell’imaging biologico e microscopico.

## Obiettivi del progetto

- Introdurre e definire teoricamente il concetto di rumore nelle immagini, analizzandone le principali cause e le strategie per la sua riduzione.

- Affrontare il problema del denoising tramite deep learning, utilizzando una rete convoluzionale U-Net per l’elaborazione di immagini.

- Applicare e comprendere l’algoritmo Noise2Void (N2V) come approccio self-supervised, consentendo l’addestramento del modello in assenza di immagini pulite di riferimento.

- Valutare l’efficacia dell’approccio proposto su dataset reali e di benchmark, evidenziandone vantaggi e limiti rispetto ai metodi classici.

## Operazioni per eseguire il progetto
### Configurazione dell’ambiente virtuale con Anaconda
1. Creazione dell'ambiente Conda
```bash
conda env create -f env/conda.yml
```
2. Attivare l’ambiente
```bash
conda activate denoising
```
3. Installare le librerie aggiuntive
```bash
pip install -r env/requirements.txt
```
### Eseguire gli script
Segui i notebook Jupyter e esegui le celle per addestrare la rete U-Net con l’algoritmo Noise2Void sul dataset BBBC006 e sul dataset di Konda PhotoCD, osservando i risultati del denoising.
```bash

```
```bash

```




