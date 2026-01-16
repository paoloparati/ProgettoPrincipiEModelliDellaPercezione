### Cos’è il noise

Inisieme di alterazioni casuali dei valori di intensità dei pixel che non corrispondono all’informazione reale della scena acquisita.
Queste perturbazioni introducono una degradazione della qualità visiva e possono compromettere l’analisi automatica dell’immagine

#### Cause che creano il noise:

- Rumore del sensore: Imperfezioni dei sensori di acquisizione
- Interferenze elettroniche: Condizioni di illuminazione sfavorevoli
- Limitazioni hardware: Qualità inferiore dei dispositivi di acquisizione (bassa risoluzione)
- Condizioni ambientali: fluttuazioni di temperatura, illuminazione instabile o vibrazioni
- Compressione dei dati: algoritmi di compressione che riducono la dimensione del file ma causano artefatti

#### Classificazione del rumore

**Rumore strutturato** → configurazione regolare o pattern nell’immagine

- Rumore sale e pepe → pixel bianchi e neri casuali sparsi nell’immagine
- Rumore a bande (banding noise) → linee orizzontali o verticali regolari nell’immagine

Rumore Non Strutturato → casuale e non segue pattern

- Rumore gaussiano: casuale e segue una distribuzione normale (gaussiana)
- Rumore di Poisson: presente in immagini con scarsa illuminazione, presenta
  variazione casuali di intensità



### Denoising di immagini digitali

Ridurre il rumore senza compromettere alcune caratteristiche essenziali

- Zone uniformi devono restare omogenee
- Bordi devono essere preservati
- Le texture devono essere mantenute
- Non devono essere generati nuovi artefatti

#### Approcci Principali al Denoising

1. Filtraggio nel dominio spaziale
2. Filtraggio nel dominio delle trasformazioni
3. Metodi avanzati
4. Metodi basati sull'apprendimento automatico

##### Filtraggio del dominio spaziale

Operazioni sui pixel sfruttando correlazioni locali.

##### Filtri lineari

Si applicano le operazioni direttamente sui valori dei pixel nell’immagine, sfruttando le correlazioni locali tra un pixel e quelli appartenenti al suo intorno

- **Filtro medio**: media dei pixel circostanti
- **Filtro Gaussiano**: ponderazione centrata

##### Filtri non lineari

- Filtro mediano: valore mediano dei pixel circostanti
- Filtro bilaterale: combina distanza spaziale e intensità per preservare i bordi

##### Filtraggio nel Dominio delle Trasformazioni

Il filtraggio nel dominio delle trasformazioni non agisce direttamente sui pixel, ma segue tre fasi fondamentali:

1. Trasformazione dell’immagine in un dominio alternativo.
2. Applicazione di filtri per attenuare le componenti associate al rumore.
3. Trasformazione inversa per tornare al dominio spaziale.

Questo approccio consente una separazione più efficace tra informazione utile e rumore.

- Trasformata di Fourier (FT): Rimozione delle alte frequenze.
- Trasformata wavelet (WT): Multi-scala per separare dettagli e rumore.
- Trasformazioni adattive: Si adattano alle caratteristiche specifiche dell'immagine e del
rumore



#### Metodi Avanzati: BM3D

Adotta un approccio iterativo in due stadi costituiti dalle seguenti operazioni di base:

- Block Matching: Regioni simili nell'immagine vengono raggruppate.
- Collaborative Filtering: Applicazione di filtri nel dominio wavelet.
- Aggregation: Combina blocchi ripuliti.

#### LIMITI dei metodi classici

Nonostante l’elevata efficacia di metodi avanzati come **BM3D**, gli approcci classici di denoising presentano alcuni limiti strutturali. In particolare, essi si basano su **ipotesi**
**predefinite sul tipo di rumore** e su **parametri fissati manualmente**, risultando poco adattabili a rumori complessi o non stazionari presenti in immagini reali. Inoltre, la loro
capacità di preservare dettagli fini e texture complesse è limitata, soprattutto in presenza di livelli elevati di rumore. Dal punto di vista computazionale, algoritmi come BM3D possono
risultare onerosi, rendendo difficile l’applicazione in contesti real-time.
Questi limiti hanno motivato l’adozione di approcci basati su **reti neurali convoluzionali**, che apprendono direttamente dai dati una rappresentazione più flessibile ed efficace del rumore.

#### Apprendimento Automatico

Convolutional Neural Networks (CNN): Architettura DnCNN: predice il rumore, lo sottrae per ottenere l'immagine pulita.

- Input: immagine rumorosa
- Output previsto: rumore sintetico aggiunto
- Loss Function: differenza tra il rumore predetto dalla rete e il rumore aggiunto

Generative Adversarial Networks (GAN)

Due reti neurali che lavorano in competizione tra loro:

- Generatore: produce immagini pulite che siano il più simili possibile a quelle reali.
- Discriminatore: valuta l'immagine e cerca di distinguere tra immagini reali e false. 

#### GROUND TRUTH

**Definizione di ground truth**
La **ground truth** è il dato di riferimento considerato corretto, che rappresenta l’informazione reale attesa e viene utilizzato per addestrare e valutare le prestazioni di un algoritmo.

##### Problema dell’assenza della ground truth

In molti contesti reali, incluse le **immagini mediche**, la ground truth non è disponibile o non è acquisibile, poiché non esiste una versione perfettamente “pulita” dell’immagine o la sua
acquisizione comporterebbe costi elevati, tempi lunghi o rischi per il paziente. Questa assenza rende difficile l’addestramento supervisionato e la valutazione oggettiva dei metodi
di denoising.

##### Introduzione al self-supervised learning e Noise2Void

Per superare la mancanza di ground truth, sono stati introdotti approcci di **apprendimento self-supervised,** che apprendono direttamente dalle immagini rumorose senza necessità di
dati puliti. Un esempio è **Noise2Void**, che sfrutta la ridondanza spaziale dell’immagine per predire il valore di un pixel a partire dal suo intorno, consentendo il denoising senza immagini
di riferimento.

#### Approcci innovativi

##### Noise2Noise (N2N):

- Modello allenato su due immagini rumorose della stessa scena.
- Immagini basate sullo stesso contenuto che sono "rumorose" in modi diversi
  Vantaggio: Non richiede immagini pulite.

##### Noise2Void (N2V):

- Lavora su una sola immagine rumorosa.
- Vantaggio: Perfetto per contesti con dati limitati.

Ingredienti:

- L'Architettura: Quale modello di Rete Neurale è più adatto?

- L'Algoritmo vero e proprio: Come fa l'Algoritmo a rimuovere il rumore?

- La Loss Function: A cosa fa riferimento, se non c'è una Ground Truth?

  

#### DATASET
Motivazione della scelta del dataset

La scelta del dataset Jump Cell Painting è motivata dalla sua rappresentatività di scenari realistici tipici dell’imaging cellulare ad alta complessità, nei quali la presenza di rumore è inevitabile. Le immagini acquisite tramite microscopia fluorescente includono più canali per diverse strutture cellulari (nuclei, mitocondri, reticolo endoplasmatico, ecc.), e riflettono fedelmente la variabilità sperimentale e le condizioni reali di acquisizione.

Il dataset non fornisce immagini “pulite” prive di rumore artificiale, rendendolo ideale per sperimentare approcci self-supervised al denoising, come Noise2Void o Noise2Self, che permettono l’addestramento del modello senza necessità di immagini di riferimento completamente prive di rumore. L’utilizzo di dati reali e multicanale consente inoltre di valutare l’efficacia dei modelli di deep learning in contesti biologici complessi, confrontando i risultati con metodi tradizionali di riduzione del rumore.

Preprocessing dei dati

Il preprocessing delle immagini del dataset Jump Cell Painting comprende due fasi principali:

- Normalizzazione: i valori di intensità dei pixel, generalmente in formato TIFF a 16 bit, vengono scalati in un intervallo [0,1]. Questa operazione assicura stabilità numerica durante l’addestramento della rete neurale e facilita la convergenza dell’algoritmo di apprendimento.

- Estrazione di patch: le immagini multicanale vengono suddivise in patch di dimensioni ridotte (ad esempio 64×64 o 128×128 pixel). Questa operazione aumenta il numero di campioni disponibili per l’addestramento, riduce il carico computazionale e permette alla rete di apprendere correlazioni locali tra pixel e canali, migliorando la qualità del denoising.
