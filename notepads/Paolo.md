### PROGETTO PRINCIPI E MODELLI DELLA PERCEZIONE

#### TITOLO DEL PROGETTO: Denoising di immagini mediante deep learning: rete convoluzionale U-Net addestrata con approccio Noise2Void.



# **PARTE 2 – Modello e sperimentazione**

### **Obiettivo**

- *Dimostrare come il metodo di denoising viene implementato in pratica.*
- *Presentare esempi concreti dei risultati ottenuti.*
- *Trarre conclusioni sull’efficacia dell’approccio e sulle possibili applicazioni.*

## 2.1 Architettura U-Net

### **U‑Net per il Denoising di Immagini**

### **1. Definizione e contesto**

U‑Net è un’architettura di **rete convoluzionale fully‑convolutional (CNN)** sviluppata da **Ronneberger et al., 2015** per la **segmentazione semantica di immagini biomediche** (*U‑Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015*). L’architettura è strutturata in due parti principali:

1. **Percorso di contrazione (encoder):** costituito da blocchi convoluzionali seguiti da operazioni di downsampling (tipicamente max-pooling), che estraggono progressivamente caratteristiche gerarchiche dell’immagine. In questa fase la rete cattura informazioni globali e contesto spaziale, riducendo le dimensioni spaziali ma aumentando la profondità dei feature map.
2. **Percorso di espansione (decoder):** costituito da blocchi convoluzionali con upsampling, che ricostruiscono la risoluzione originale dell’immagine combinando le informazioni globali del bottleneck con i dettagli locali provenienti dall’encoder tramite le **skip connections**. Queste connessioni laterali trasferiscono feature map ad alta risoluzione dai corrispondenti livelli dell’encoder al decoder, preservando dettagli spaziali e contorni.

La combinazione di encoder, decoder e skip connections conferisce alla rete la caratteristica **forma a “U”**, che ottimizza sia l’apprendimento di informazioni globali sia la conservazione di dettagli locali.

Sebbene originariamente progettata per la segmentazione, la struttura di U‑Net è altamente efficace per il **denoising di immagini**. In questo contesto, la rete viene addestrata per apprendere una **mappatura tra immagini rumorose e immagini pulite**, separando il **rumore casuale** dal **segnale reale**. L’output del modello ha la stessa dimensione spaziale dell’input, ma con il rumore significativamente ridotto, mantenendo intatti dettagli strutturali e contorni importanti per l’analisi biologica.

------

### **2. Struttura e funzionamento di U‑Net**

#### **2.1 Encoder (Contrazione)**

La parte di **encoder**, o percorso di contrazione, è il cuore della U‑Net dove la rete “impara” a riconoscere le caratteristiche principali dell’immagine. L’encoder è costituito da **blocchi convoluzionali 3×3**: ogni blocco applica delle convoluzioni per estrarre informazioni dai pixel vicini e le passa attraverso una funzione di attivazione **ReLU** (Rectified Linear Unit), che introduce non-linearità permettendo alla rete di modellare relazioni più complesse tra i pixel.

Alla fine di ogni blocco, un’operazione di **max pooling 2×2** riduce le dimensioni spaziali dell’immagine, cioè il numero di righe e colonne dei pixel considerati. Questo permette alla rete di concentrarsi su **pattern globali e strutture principali**, ignorando temporaneamente dettagli locali o rumore. Parallelamente, il numero di **canali** (cioè il numero di mappe di feature generate dalle convoluzioni) aumenta, consentendo di immagazzinare più informazioni descrittive. In altre parole, la rete diventa più “profonda” nella conoscenza dell’immagine man mano che scende lungo l’encoder.

#### **2.2 Decoder (Espansione)**

Dopo aver estratto le informazioni globali e astratte dell’immagine nell’encoder, la U‑Net deve ricostruire un’immagine alla stessa risoluzione dell’input originale. Questo compito è svolto dal **decoder**, o percorso di espansione.

Nel decoder, ogni blocco esegue un’operazione di **upsampling**, che può essere una **transpose convolution** (una convoluzione “inversa” che aumenta le dimensioni) o un semplice upsampling bilineare. Lo scopo è riportare la risoluzione spaziale dell’immagine verso quella originale.

A questo punto entrano in gioco le **skip connections**, che collegano ogni livello dell’encoder con il corrispondente livello del decoder. Queste connessioni permettono di trasferire informazioni ad alta risoluzione, persino dettagli locali che erano stati persi durante il pooling dell’encoder. Successivamente, le convoluzioni (3×3 + ReLU) combinano le informazioni globali dell’encoder con i dettagli locali, permettendo al decoder di **ricostruire un’immagine pulita e dettagliata**.

#### **2.3 Skip Connections**

Le **skip connections** sono fondamentali, soprattutto nel denoising. Senza di esse, il decoder avrebbe solo le informazioni astratte dell’encoder, rischiando di perdere dettagli critici come **bordi, contorni e texture locali**. Nel contesto del denoising, le skip connections permettono alla rete di ricostruire i pixel realistici preservando le strutture biologiche e separando efficacemente il rumore dal segnale reale.

In termini semplici: immagina di scendere lungo l’encoder per capire “cosa c’è nell’immagine” e poi risalire nel decoder per disegnarla di nuovo; le skip connections sono come degli appunti presi lungo il percorso di discesa, che ti permettono di non dimenticare i dettagli mentre ricostruisci l’immagine.

#### **2.4 Bottleneck**

Al centro della U‑Net si trova il **bottleneck**, la parte più profonda dell’architettura. Qui i feature map hanno la dimensione spaziale più piccola, ma contengono informazioni molto astratte e sintetiche dell’immagine. Il bottleneck agisce come un **filtro globale del rumore**, permettendo al decoder di partire da una rappresentazione “pulita” e robusta del contenuto dell’immagine, prima di reintegrare i dettagli persi tramite le skip connections.

<img src="/home/paoloparati/Pictures/Screenshots/U-Net-with-skip-connections.jpg" style="zoom:50%;" />

------

### **3. U‑Net per il denoising**

Quando utilizziamo U‑Net per il **denoising**, la rete prende in input un’immagine **rumorosa** e produce in output un’immagine **della stessa dimensione**, ma priva di rumore. In pratica, la rete impara a distinguere il **segnale reale** (strutture biologiche importanti) dal **rumore casuale** che degrada l’immagine.

I principali vantaggi della U‑Net per il denoising sono:

- **Preservazione dei dettagli spaziali**: grazie alle skip connections, anche i bordi più sottili e le texture locali vengono mantenuti.
- **Flessibilità**: essendo completamente convoluzionale (fully convolutional), la rete può essere applicata a immagini di dimensioni diverse senza bisogno di ridimensionarle.
- **Self-supervised learning**: approcci come **Noise2Void** permettono di addestrare la rete anche senza avere immagini pulite come ground truth, mascherando casualmente alcuni pixel durante il training e calcolando la loss solo su di essi.

------

### **4. Riferimenti autorevoli**

1. **Ronneberger, O., Fischer, P., & Brox, T. (2015).** *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015. https://arxiv.org/abs/1505.04597
2. **Çiçek, Ö., Abdulkadir, A., Lienkamp, S.S., Brox, T., Ronneberger, O. (2016).** *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.* MICCAI 2016.
3. **Mathis, A., et al. (2018).** *Deep Learning for Image Denoising: U-Net Architectures and Variants.* https://www.mathstoml.com/u-net
4. **Wikipedia – U-Net.** https://en.wikipedia.org/wiki/U-Net
5. **Ronneberger, O. (2015). *U‑Net architecture with skip connections* [Diagram]**.  https://www.researchgate.net/figure/U-Net-with-skip-connections_fig3_346417821



## 2.2 Addestramento Noise2Void

### **Noise2Void per il denoising self-supervised**

**Noise2Void (N2V)**, introdotto da **Krull et al., 2019** (*Noise2Void – Learning Denoising from Single Noisy Images, CVPR 2019*), è un metodo di denoising che permette di addestrare una rete neurale senza la necessità di avere immagini pulite di riferimento (*ground truth*). L’idea centrale alla base di N2V è che ogni pixel di un’immagine rumorosa può essere stimato usando esclusivamente le informazioni provenienti dai pixel circostanti, evitando di guardare il pixel stesso durante il training.

In questo modo, la rete impara a distinguere **il segnale reale**, coerente e strutturato, dal **rumore casuale**, che è imprevedibile e distribuito in maniera aleatoria. Questo approccio è particolarmente utile in ambito biologico e medico, dove ottenere immagini perfettamente pulite può essere difficile o impossibile.

------

#### **1. Mascheramento dei pixel (Pixel Masking)**

Il concetto centrale di Noise2Void è: **“imparare a prevedere un pixel usando solo i suoi vicini”**, senza mai guardare direttamente il pixel stesso. Questo permette di distinguere il **rumore casuale** dal **segnale reale**, anche senza avere immagini pulite come riferimento.

1. **Selezione casuale dei pixel**
   - Durante l’addestramento, la rete non osserva tutti i pixel allo stesso modo.
   - Viene scelto un sottoinsieme di pixel in maniera casuale. Questi pixel saranno i **pixel target**, cioè quelli che la rete dovrà imparare a predire.
2. **Mascheramento dei pixel**
   - I pixel selezionati vengono temporaneamente **“nascosti”**.
   - Questo significa che, invece di mostrare alla rete il valore reale del pixel, si inserisce un valore alternativo, ad esempio:
     - la media dei pixel circostanti, oppure un valore casuale leggermente perturbato.
   - In questo modo la rete non può copiare semplicemente il valore reale: deve imparare a usare il contesto per predirlo.
3. **Predizione del pixel**
   - La rete (tipicamente una **U‑Net**) osserva solo i pixel **non mascherati**, cioè tutti quelli vicini al pixel nascosto.
   - Usando le informazioni spaziali dai pixel circostanti, la rete cerca di predire il **valore corretto** del pixel mascherato.
4. **Calcolo della loss**
   - Una funzione di errore (ad esempio **Mean Squared Error**) confronta la predizione della rete con il valore reale del pixel mascherato.
   - Solo i pixel mascherati contribuiscono alla loss, mentre tutti gli altri pixel non mascherati servono solo come contesto.

### **Noise2Void: Architettura e Addestramento**

Noise2Void (N2V) è un metodo **self-supervised** per il denoising di immagini che non richiede una ground truth pulita. Solitamente utilizza una **U‑Net** come backbone, sfruttando la sua capacità di catturare sia caratteristiche globali sia dettagli locali attraverso la combinazione di encoder, decoder e skip connections.

#### **2.1 Pipeline di addestramento Noise2Void**

L’addestramento si basa su un meccanismo di **mascheramento dei pixel** (pixel masking) e apprendimento contestuale:

1. **Preparazione delle immagini**
   - Normalizzazione dei valori dei pixel, tipicamente in un range [0,1], per stabilizzare il training e velocizzare la convergenza.
   - Suddivisione delle immagini in **patch** (es. 64×64 o 128×128 pixel) per aumentare il numero di campioni e ridurre l’uso di memoria GPU.
2. **Mascheramento dei pixel**
   - Selezione casuale di un sottoinsieme di pixel in ogni patch.
   - Sostituzione dei pixel selezionati con valori “fake”, ad esempio la media dei pixel circostanti o un valore perturbato. In questo modo la rete non può copiare direttamente il valore reale del pixel, ma deve inferirlo dal contesto.
3. **Forward pass**
   - La rete riceve in input la patch con pixel mascherati e produce una predizione per tutti i pixel, ma solo quelli mascherati sono considerati rilevanti per la loss.
4. **Calcolo della loss**
   - La funzione di perdita L2 (**Mean Squared Error**) viene calcolata tra i pixel mascherati predetti e i valori reali.
   - Solo i pixel mascherati contribuiscono alla loss, mentre i pixel non mascherati forniscono contesto.
5. **Backpropagation**
   - I gradienti calcolati dalla loss vengono propagati all’indietro attraverso la rete, aggiornando i pesi dei blocchi convoluzionali per minimizzare l’errore di predizione.
6. **Iterazione su tutte le patch**
   - Il processo viene ripetuto su tutte le patch di tutte le immagini del dataset.
   - In questo modo, la rete impara a distinguere **rumore casuale** da **strutture coerenti**, pur non avendo mai visto un’immagine completamente pulita.

#### **2.2 Risultato**

Al termine dell’addestramento, la rete è in grado di ricostruire immagini denoised, preservando **strutture biologiche e dettagli locali**, e riducendo efficacemente il rumore presente nei dati di imaging.

#### **Riferimenti autorevoli**

- Krull, A., Buchholz, T.O., & Jug, F. (2019). *Noise2Void – Learning Denoising from Single Noisy Images*. CVPR 2019. https://arxiv.org/abs/1811.10980
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015. https://arxiv.org/abs/1505.04597



### 2.3 Integrazione operativa tra U-Net e Noise2Void nel denoising di immagini

Il **denoising di immagini mediante deep learning** rappresenta una sfida particolarmente complessa nel contesto dell’imaging biomedico, soprattutto quando non sono disponibili immagini di riferimento “pulite” da utilizzare come ground truth. In questo scenario, la combinazione dell’architettura **U‑Net** con l’approccio **Noise2Void** permette di costruire un sistema di apprendimento **self-supervised**, in grado di separare efficacemente il segnale biologico reale dal rumore casuale sfruttando solo le proprietà strutturali intrinseche dei dati.

L’efficacia di questa strategia deriva dalla **complementarità dei due componenti**:

- **U‑Net** fornisce una struttura architetturale profondamente gerarchica, capace di modellare contemporaneamente relazioni spaziali locali, come contorni e dettagli cellulari, e relazioni globali, come la disposizione generale delle strutture biologiche. Grazie al suo percorso di contrazione (encoder) e di espansione (decoder), integrato con le skip connections, la rete può preservare le informazioni spaziali più fini anche durante la ricostruzione dell’immagine denoised.
- **Noise2Void** definisce una strategia di addestramento self-supervised che permette di allenare la rete **senza immagini pulite di riferimento**. Mascherando casualmente alcuni pixel e confrontando l’output della rete solo su questi, il modello impara a stimare il valore atteso dei pixel a partire dal contesto circostante. In questo modo, la rete distingue il **segnale strutturale coerente** dal **rumore casuale**, che non è predicibile dai pixel vicini.

La scelta di utilizzare questa combinazione è particolarmente motivata dal nostro dataset, **BBBC006**, costituito da immagini di nuclei cellulari acquisite con microscopia a fluorescenza. Queste immagini presentano **rumore reale**, generato da fenomeni fisici come fluttuazioni di fotoni, rumore elettronico e imperfezioni ottiche, che rende impossibile ottenere una ground truth pulita. Applicando U‑Net insieme a Noise2Void, possiamo **addestrare la rete direttamente sulle immagini rumorose** e ottenere un output denoised che preserva fedelmente i dettagli biologici, migliorando la visibilità di strutture cellulari e facilitando analisi quantitative successive.

### **1. Origine dei dati e natura del problema**

Il punto di partenza del presente lavoro è costituito da un insieme di immagini acquisite sperimentalmente mediante **microscopia a fluorescenza**, nello specifico il dataset **BBBC006**. Tali immagini rappresentano nuclei cellulari marcati fluorescentemente e sono caratterizzate dalla presenza di **rumore reale**, ossia rumore intrinseco al processo di acquisizione e non introdotto artificialmente in fase di simulazione.

Il rumore reale nelle immagini di microscopia deriva da diversi fenomeni fisici e strumentali. Tra le principali sorgenti si annovera il **rumore di conteggio dei fotoni (Poisson noise)**, dovuto alla natura discreta dell’emissione e della rilevazione dei fotoni, particolarmente rilevante in condizioni di bassa illuminazione. A questo si aggiunge il **rumore elettronico del sensore**, generato dai circuiti di lettura e amplificazione della camera, nonché le **variazioni di illuminazione** che possono verificarsi durante l’acquisizione, causando disomogeneità spaziali nell’intensità del segnale. Un ulteriore contributo al degrado dell’immagine è dato dalla **perdita di informazione associata a immagini parzialmente fuori fuoco**, fenomeno frequente nella microscopia ottica tridimensionale.

A differenza del rumore artificiale, che viene solitamente modellato tramite distribuzioni matematiche semplici e stazionarie (ad esempio rumore gaussiano additivo), il rumore reale presenta caratteristiche decisamente più complesse. In particolare, esso non è **perfettamente modellabile**, in quanto dipende da molteplici fattori fisici difficilmente isolabili; è spesso **non stazionario**, poiché la sua distribuzione può variare nello spazio e nell’intensità del segnale; ed è fortemente **dipendente dalle condizioni di acquisizione**, come il tempo di esposizione, il tipo di sensore e le impostazioni microscopiche.

Queste proprietà rendono il problema del denoising particolarmente complesso e limitano l’applicabilità degli approcci supervisionati classici basati sul deep learning. Tali metodi richiedono infatti la disponibilità di coppie di immagini perfettamente allineate, costituite da una versione rumorosa e dalla corrispondente **immagine pulita di riferimento (ground truth)**. Nel contesto biomedico, tuttavia, l’acquisizione di immagini realmente prive di rumore è spesso impraticabile o addirittura impossibile, poiché l’aumento del segnale comporterebbe dosi di illuminazione incompatibili con la vitalità del campione o con le condizioni sperimentali.

Alla luce di queste considerazioni, risulta evidente la necessità di adottare approcci alternativi, capaci di operare efficacemente in assenza di ground truth pulite, come nel caso delle tecniche di denoising **self-supervised**, tra cui Noise2Void, oggetto di studio nel presente progetto.

#### **2. Preprocessing e preparazione dei dati**

Prima di procedere con l’addestramento della rete neurale, è necessario effettuare una fase di **pre-processing** delle immagini, ovvero un insieme di operazioni volte a trasformare i dati grezzi acquisiti sperimentalmente in una rappresentazione numerica compatibile con il funzionamento di una rete neurale convoluzionale.

Le immagini di microscopia a fluorescenza, come quelle del dataset BBBC006, sono tipicamente acquisite in **formato a 16 bit**, con un’elevata dinamica di intensità e con distribuzioni dei pixel fortemente dipendenti dalle condizioni di acquisizione. In questa forma, i valori dei pixel non sono direttamente adatti all’addestramento di una rete neurale, poiché possono presentare scale molto diverse tra immagini differenti, rendendo instabile il processo di ottimizzazione.

Per questo motivo, il primo passo del pre-processing consiste nella **normalizzazione delle intensità**, che ha lo scopo di riportare i valori dei pixel in un intervallo numerico controllato, generalmente compreso tra 0 e 1. La normalizzazione riduce la variabilità artificiale tra immagini, migliora la stabilità numerica durante il training e facilita la convergenza dell’algoritmo di apprendimento.

Successivamente, le immagini vengono suddivise in **patch**, ovvero sotto-immagini di dimensione fissa (ad esempio 64×64 o 128×128 pixel). Questa operazione risponde a diverse esigenze: aumenta il numero di campioni disponibili per l’addestramento, riduce il carico computazionale e consente alla rete di concentrarsi su regioni locali dell’immagine, dove il rumore e le strutture biologiche si manifestano in modo più omogeneo.

Nel contesto di Noise2Void, il pre-processing assume un ruolo ancora più centrale, poiché durante la fase di training viene applicato un **mascheramento casuale dei pixel** all’interno delle patch. Alcuni pixel vengono intenzionalmente nascosti o sostituiti con valori provenienti dal loro intorno, costringendo la rete a predirne il valore basandosi esclusivamente sul contesto spaziale circostante. Questo meccanismo consente l’addestramento in assenza di immagini pulite di riferimento, sfruttando le proprietà statistiche del rumore reale.

In sintesi, il pre-processing non rappresenta un semplice passaggio preliminare, ma una fase fondamentale che condiziona direttamente l’efficacia dell’addestramento e la qualità finale del denoising ottenuto dalla combinazione U-Net + Noise2Void.

#### **2.1 Lettura e rappresentazione numerica**

Le immagini in formato TIF a 16 bit vengono inizialmente caricate in memoria sotto forma di **matrici bidimensionali di valori numerici**, in cui ciascun elemento della matrice corrisponde all’intensità luminosa associata a un singolo pixel dell’immagine. In questa rappresentazione, le coordinate della matrice identificano la posizione spaziale del pixel, mentre il valore numerico ne descrive il livello di fluorescenza misurato dal sensore.

L’utilizzo del formato a 16 bit consente di preservare un’elevata **dinamica di intensità**, permettendo di rappresentare variazioni sottili del segnale biologico che sarebbero perse in una codifica a 8 bit. Questo aspetto è particolarmente rilevante nelle immagini di microscopia a fluorescenza, dove il contrasto tra segnale e rumore può essere molto ridotto e le informazioni utili sono spesso contenute in variazioni di intensità di piccola entità.

Questa rappresentazione numerica rende possibile l’elaborazione diretta delle immagini da parte della rete neurale convoluzionale, che opera su dati quantitativi attraverso operazioni matematiche come convoluzioni, somme pesate e funzioni di attivazione. In questo modo, l’intera pipeline di denoising può essere formulata come un problema di apprendimento su matrici di numeri reali, senza introdurre approssimazioni o perdite di informazione legate a conversioni premature o compressioni del dato.

In sintesi, la lettura delle immagini come matrici bidimensionali di intensità costituisce il primo passaggio fondamentale che permette di collegare il dato sperimentale reale al modello matematico e computazionale rappresentato dalla rete U-Net, garantendo una base informativa adeguata per il successivo processo di addestramento e denoising.

### **2.2 Normalizzazione**

Una volta caricate come matrici di intensità, le immagini vengono sottoposte a una fase di **normalizzazione dei valori di pixel**, tipicamente riportando le intensità nell’intervallo continuo [0, 1]. Questo passaggio non ha lo scopo di alterare il contenuto visivo o informativo dell’immagine, ma di rendere i dati numericamente più adatti all’elaborazione da parte della rete neurale.

Dal punto di vista matematico, la normalizzazione consiste in una trasformazione lineare dei valori di intensità, che preserva le relazioni relative tra i pixel. In altre parole, se un pixel risulta più luminoso di un altro prima della normalizzazione, lo rimarrà anche dopo la trasformazione. Ciò garantisce che la struttura dell’immagine e il contrasto relativo tra le diverse regioni vengano mantenuti invariati.

L’importanza della normalizzazione risiede principalmente negli aspetti legati all’ottimizzazione della rete neurale. Riducendo la variabilità numerica dei dati in ingresso, si evita che valori di grande ampiezza producano gradienti instabili durante la fase di backpropagation. Questo contribuisce a migliorare la **stabilità numerica del processo di addestramento**, riducendo il rischio di esplosione o annullamento del gradiente.

Inoltre, lavorare con valori normalizzati consente agli algoritmi di ottimizzazione, come la discesa del gradiente e le sue varianti, di operare in modo più efficiente. In pratica, la rete converge più rapidamente verso una soluzione ottimale, richiedendo un numero inferiore di epoche di addestramento e favorendo una maggiore robustezza del modello appreso.

È fondamentale sottolineare che la normalizzazione non comporta alcuna perdita di informazione: essa agisce esclusivamente sulla scala dei valori numerici, senza modificare la distribuzione spaziale del segnale o le caratteristiche strutturali dell’immagine. Per questo motivo, rappresenta uno step standard e imprescindibile nelle pipeline di deep learning applicate al denoising di immagini biomediche.

### **2.3 Suddivisione in patch**

Dopo la normalizzazione, le immagini vengono ulteriormente processate suddividendole in **patch di dimensione fissa**, tipicamente di 64 x 64 o 128 x 128 pixel. Questo passaggio rappresenta una fase cruciale nella pipeline di addestramento della rete U‑Net per il denoising.

La suddivisione in patch offre diversi vantaggi. In primo luogo, consente di **aumentare significativamente il numero di esempi di training**, poiché da una singola immagine di grandi dimensioni possono essere generate molteplici patch sovrapposte o non sovrapposte. In secondo luogo, riduce il **carico computazionale**: elaborare piccole porzioni dell’immagine è meno oneroso in termini di memoria e tempo di calcolo rispetto a trattare l’immagine intera. Infine, permette alla rete di **apprendere pattern locali ricorrenti**, ossia strutture e dettagli che si ripetono all’interno dell’immagine, come nuclei cellulari, bordi o texture specifiche.

Dal punto di vista del modello, ogni patch viene considerata come un **campione indipendente**. Questo significa che la rete riceve in input molteplici frammenti più piccoli, aumentando la diversità dei dati osservati durante l’addestramento e migliorando la capacità del modello di generalizzare su immagini mai viste prima.

------

## **3. Principio Noise2Void: mascheramento e apprendimento self-supervised**

L’elemento distintivo dell’approccio **Noise2Void** è costituito dal **mascheramento dei pixel**, una strategia metodologica che rende possibile l’addestramento di una rete neurale **senza disporre di immagini pulite di riferimento (ground truth)** o di coppie di immagini rumorose corrispondenti. Questo principio si colloca all’interno della più ampia classe di tecniche di apprendimento **self‑supervised**, in cui il modello impara a inferire la struttura del segnale direttamente dai dati grezzi a disposizione, senza supervisione esterna esplicita.

Il problema centrale nel training di un denoiser tradizionale risiede nel fatto che una rete neurale standard, se addestrata semplicemente utilizzando un’immagine rumorosa come input e come target, tenderebbe ad apprendere una **mapping identità**: ovvero la rete imparerebbe a restituire in output esattamente ciò che riceve in input, replicando il rumore invece di rimuoverlo. Per evitare questa degenerazione, Noise2Void introduce una modifica strutturale e procedurale nello schema di addestramento basata su una **rete con “blind spot”**, ovvero una rete in cui il valore del pixel che deve essere predetto non è accessibile alla rete stessa durante la previsione. 

Operativamente, il mascheramento dei pixel funziona nel seguente modo: all’interno di ciascuna patch estratta dalle immagini rumorose, viene **selezionato casualmente uno o più pixel** (tipicamente posizionati al centro o distribuiti secondo una strategia di campionamento) e **il valore reale di tali pixel viene rimosso o sostituito** prima di essere fornito come input alla rete. La versione modificata della patch, in cui il pixel centrale è stato occultato (cioè “mascherato”), costituisce l’input effettivo per la rete durante il training, mentre **il valore originale non alterato di quel pixel mascherato viene utilizzato come target di riferimento per il calcolo della loss**. 

Da un punto di vista statistico, l’efficacia di questa procedura deriva da due ipotesi fondamentali:

- **Il rumore è indipendente tra pixel**, ovvero non presenta correlazioni spaziali significative.
- **Il segnale reale presenta dipendenze statistiche locali**, cioè la struttura dell’immagine non è casuale ma è determinata da relazioni spaziali coerenti. 

Queste ipotesi implicano che, se la rete è privata del valore del pixel da predire, essa non può semplicemente “ricordare” quel valore rumoroso; deve invece sfruttare le informazioni nei pixel circostanti per formulare una stima del valore più probabile, che rappresenta il **segnale reale** piuttosto che il rumore. Questo è l’esatto opposto di ciò che avverrebbe in un training supervisionato convenzionale: la rete non ha accesso diretto alla risposta corretta su tutta l’immagine, ma apprende a inferire il valore del pixel mascherato attraverso un’analisi contestuale delle immagini rumorose stesse. Il termine “blind spot” si riferisce proprio a questo concetto: la rete non può vedere il pixel da predire nel suo campo recettivo (receptive field) e quindi è costretta a basarsi esclusivamente sulle informazioni spaziali circostanti per stimare il suo valore. L’effetto risultante è che la rete apprende a catturare le **dipendenze statistiche del segnale reale** presenti nello spazio dell’immagine, mentre la componente di rumore, non essendo prevedibile dal contesto, viene attenuata. Dal punto di vista implementativo, Noise2Void può essere realizzato in modi differenti. Una strategia consiste nel mascherare i pixel direttamente nelle patch di input sostituendoli con valori casuali prelevati da un intorno locale, oppure con una media dei pixel circostanti; in seguito, la rete produce una previsione per l’intera patch, ma **la funzione di perdita è calcolata solamente sui pixel mascherati**, ignorando i restanti. 

Questa procedura ha un duplice effetto:

1. **Evita la regressione identità**, forzando la rete a imparare pattern contestuali piuttosto che memorizzare semplicemente i valori di ingresso.
2. **Trasforma il problema di denoising in una stima statistica basata sul contesto**, sfruttando la correlazione spaziale del segnale reale. 

### **3.1 Mascheramento dei pixel**

Per ogni patch dell’immagine, viene selezionato in modo casuale un sottoinsieme di pixel da **mascherare**. Il valore reale di questi pixel non viene fornito alla rete: può essere sostituito con un valore nullo, la media dei pixel circostanti o un valore casuale preso dal contesto locale. La patch così modificata, in cui i pixel selezionati sono nascosti, costituisce l’**input effettivo** fornito alla rete durante l’addestramento.

Il punto cruciale è che il **valore originale dei pixel mascherati rimane completamente inaccessibile alla rete**, anche indirettamente attraverso le informazioni circostanti. In questo modo la rete non può limitarsi a replicare il rumore presente nell’immagine: è costretta a stimare il valore più probabile del pixel basandosi esclusivamente sul **contesto spaziale dei pixel non mascherati circostanti**. Questo principio è alla base del funzionamento self-supervised di Noise2Void, perché permette alla rete di **apprendere la struttura reale dell’immagine e separarla dal rumore**, senza necessitare di immagini di riferimento pulite.

In termini tecnici, la funzione di perdita (loss) viene calcolata **solo sui pixel mascherati**, ignorando i restanti, costringendo così la rete a concentrarsi esclusivamente sulla previsione del segnale reale nei punti “ciechi”. Questo approccio garantisce che l’apprendimento sia guidato dal contesto locale e non dalla memorizzazione del rumore presente nei dati.

### **3.2 Calcolo della loss**

Durante l’addestramento di Noise2Void, la funzione di perdita (loss) viene calcolata **solo sui pixel mascherati**. In pratica, l’output della rete viene confrontato con i valori reali dei pixel originariamente nascosti nella patch, ignorando completamente i pixel non mascherati.

Questo meccanismo ha due effetti fondamentali:

1. La rete **non è penalizzata** per eventuali errori sui pixel non mascherati, evitando di imparare semplicemente a replicare il rumore presente nell’immagine.
2. L’apprendimento è focalizzato **esclusivamente sulla capacità di predire i pixel nascosti** a partire dal contesto locale. In altre parole, la rete deve imparare a riconoscere le strutture e i pattern reali dell’immagine per stimare correttamente il valore dei pixel mascherati, separando così il segnale reale dal rumore.

Questo approccio self-supervised consente di addestrare la rete su immagini **rumorose reali** senza necessità di una ground truth pulita, rendendo Noise2Void particolarmente adatto a dati biomedici dove immagini completamente prive di rumore sono difficili o impossibili da ottenere.

### **3.3 Fondamento statistico**

Il funzionamento di Noise2Void si basa su un’ipotesi fondamentale riguardo alle caratteristiche dell’immagine: il **segnale reale** (cioè le strutture biologiche o informazioni significative presenti nell’immagine) è **spazialmente correlato**, mentre il **rumore** è casuale e **statisticamente indipendente tra pixel**.

In altre parole, i pixel vicini tra loro contengono informazioni simili sul segnale reale, mentre il rumore varia in maniera casuale e non può essere previsto semplicemente osservando i pixel circostanti.

Grazie a questa ipotesi, la rete è incentivata a usare il **contesto spaziale dei pixel circostanti** per stimare il valore corretto di un pixel mascherato, producendo così una **stima del valore atteso del segnale reale**. Tutte le componenti casuali del rumore vengono ignorate, perché non apportano informazioni predittive.

Questo principio è ciò che permette a Noise2Void di separare efficacemente il rumore dal segnale, senza bisogno di immagini pulite come ground truth. La rete impara a “ricostruire” l’immagine basandosi solo sulle informazioni consistenti e strutturali presenti nel contesto locale.

------

## **4. Ruolo della U-Net all’interno di Noise2Void**

All’interno del framework Noise2Void, la **U-Net** funge da **modello principale** responsabile della predizione dei valori dei pixel mascherati. La rete riceve in input le patch mascherate e, grazie alla sua architettura encoder-decoder con **skip connections**, è in grado di combinare:

- le **informazioni globali** acquisite dagli strati profondi dell’encoder, che catturano le caratteristiche strutturali principali dell’immagine, e
- i **dettagli locali** preservati dalle skip connections, che permettono di mantenere contorni, bordi e texture finemente dettagliati.

Questa combinazione è cruciale nel denoising: i pixel mascherati vengono stimati utilizzando il contesto circostante, mentre il rumore casuale, non correlato spazialmente, viene ignorato.

In pratica, la U-Net agisce come un **filtro adattivo**, in grado di “ricostruire” l’immagine pulita a partire dai dati parziali, rispettando la struttura e le caratteristiche delle cellule o degli oggetti presenti.

Grazie alla **fully-convolutional design**, la rete può elaborare immagini di dimensioni variabili senza perdita di informazione, mentre la suddivisione in patch e la normalizzazione dei valori facilitano l’apprendimento e accelerano la convergenza durante l’addestramento.

### **4.1 Encoder: analisi multiscala**

Nel percorso di **contrazione** della U‑Net, noto anche come **encoder**, la rete elabora le immagini a più livelli di dettaglio in maniera gerarchica. Ogni blocco dell’encoder è costituito da **convoluzioni 3×3** seguite da funzioni di attivazione **ReLU** (*Rectified Linear Unit*). Queste operazioni hanno lo scopo di estrarre **feature locali**, cioè informazioni come **bordi, contorni, texture e piccole strutture cellulari**, che rappresentano la morfologia fine dell’immagine.

Dopo ogni blocco convoluzionale viene applicato un **max pooling 2×2**, un’operazione di downsampling che riduce le dimensioni spaziali dei feature map. Questo meccanismo permette alla rete di **ridurre progressivamente la risoluzione dell’immagine**, perdendo dettagli molto fini a livello locale, ma guadagnando **informazioni più globali e astratte** sulle strutture presenti nell’immagine. In altre parole, mentre l’encoder “comprende” l’immagine a livello macroscopico, cattura pattern complessi che descrivono la forma e l’organizzazione delle cellule, ignorando il rumore casuale che caratterizza ogni pixel.

Un altro aspetto cruciale è l’aumento del **numero di canali dei feature map** man mano che si scende lungo l’encoder. Questo significa che, sebbene la dimensione spaziale dei dati diminuisca, la profondità dei feature map aumenta, consentendo alla rete di **immagazzinare più informazioni descrittive per ogni regione dell’immagine**. Così, le feature catturate a livelli profondi contengono sia relazioni globali che dettagli sulle strutture biologiche, fondamentali per distinguere il **segnale reale** dal **rumore**.

### **4.2 Bottleneck: rappresentazione astratta**

Il **bottleneck** rappresenta il punto più profondo della U‑Net, cioè la parte centrale della “U” dove le informazioni spaziali sono al minimo e le feature map sono al massimo livello di astrazione. In questa fase, la rete non lavora più sui dettagli locali dei singoli pixel, ma su una **rappresentazione compatta e sintetica dell’immagine** nel suo insieme.

Questa rappresentazione astratta è fondamentale per il denoising: mentre le strutture biologiche coerenti – come nuclei o membrane cellulari – si manifestano come pattern ripetuti e correlati spazialmente, il rumore è **casuale e indipendente tra pixel**. Di conseguenza, nel bottleneck il rumore viene **naturalmente attenuato**, mentre le informazioni strutturali vengono mantenute e rafforzate.

In pratica, il bottleneck agisce come un **filtro globale**: elimina le fluttuazioni casuali e conserva ciò che è rilevante per la ricostruzione dell’immagine. Quando le feature di questa sezione vengono poi passate al decoder, la rete può combinare la comprensione globale dell’immagine con i dettagli spaziali preservati tramite le **skip connections**, generando un output denoised fedele all’originale.

### **4.3 Decoder e skip connections: ricostruzione informata**

Dopo la fase di **bottleneck**, l’immagine non è più rappresentata come pixel singoli, ma come **feature map compresse**, in cui le informazioni rilevanti sono condensate e il rumore è stato in gran parte filtrato. Il compito del **decoder** è quello di trasformare queste rappresentazioni astratte in un’immagine finale della stessa dimensione dell’input, pulita dal rumore.

Per fare questo, il decoder esegue una serie di **operazioni di upsampling**. L’upsampling può avvenire tramite **convoluzioni trasposte (transposed convolutions)**, che apprendono come riempire i pixel mancanti in modo ottimale, oppure tramite **interpolazione bilineare**, che ridistribuisce i valori spaziali senza aggiungere parametri. L’idea di base è semplice: man mano che la rete “risale” dal bottleneck verso l’output, la risoluzione spaziale cresce, e l’immagine ricostruita assume gradualmente la forma originale.

Qui entrano in gioco le **skip connections**, collegamenti diretti tra gli strati corrispondenti di encoder e decoder. Durante la contrazione, l’encoder riduce progressivamente la risoluzione spaziale tramite **pooling**, perdendo dettagli locali ma acquisendo informazioni globali sul contesto dell’immagine. Le skip connections permettono di trasferire **queste informazioni locali perse** direttamente al decoder. In altre parole, mentre il decoder “ricostruisce” la struttura generale dell’immagine, le skip connections forniscono **i dettagli precisi dei bordi e delle texture** che altrimenti andrebbero persi.

Dal punto di vista del **denoising**, questo meccanismo è essenziale:

1. Il decoder utilizza le feature globali del bottleneck per distinguere tra **segnale reale** e **rumore casuale**.
2. Le skip connections aggiungono le informazioni locali necessarie a **preservare i contorni cellulari, le strutture sottili e le texture biologiche**.
3. L’output finale non è solo meno rumoroso, ma **fedelmente rappresentativo della struttura originale**, evitando artefatti di levigatura che comprometterebbero l’analisi biologica.

In pratica, possiamo immaginare il processo così: il decoder “ricostruisce” l’immagine come un puzzle, usando i pezzi astratti del bottleneck per la struttura generale e aggiungendo i pezzi dettagliati delle skip connections per rendere l’immagine **realistica e fedele** all’originale. Questo permette di ottenere immagini denoised che conservano sia la coerenza globale sia i dettagli locali, rendendole utilizzabili per analisi quantitative o visualizzazione biologica.

------

## **5. Output e fase di inferenza**

Quando parliamo di U‑Net combinata con Noise2Void, dobbiamo distinguere due momenti fondamentali: l’**addestramento** e l’**inferenza**.

Durante l’addestramento, la rete non vede mai l’immagine completa come dovrebbe apparire “pulita”. Al contrario, le immagini vengono suddivise in patch e, all’interno di ciascuna patch, alcuni pixel vengono **nascosti** in modo casuale. La rete riceve quindi come input un’immagine in cui alcune informazioni sono deliberate mancate e deve imparare a **predire i valori mancanti** basandosi esclusivamente sui pixel circostanti.

Questa strategia, caratteristica di Noise2Void, permette alla rete di **apprendere senza avere a disposizione una ground truth perfetta**. La funzione di perdita viene calcolata **solo sui pixel mascherati**, quindi il modello viene incoraggiato a concentrarsi su ciò che può essere dedotto dal contesto, imparando a distinguere **segnale reale** da **rumore casuale**. Grazie all’architettura della U‑Net, con il suo bottleneck e le skip connections, il modello riesce contemporaneamente a catturare **dettagli locali** e **informazioni globali**: i dettagli fini dei pixel e la struttura complessiva dell’immagine.

Una volta che l’addestramento è completato, si passa alla fase di inferenza. Qui la situazione cambia: l’intera immagine rumorosa viene fornita come input, **senza alcun mascheramento**. La rete utilizza i pesi appresi per stimare i valori dei pixel, basandosi sulle caratteristiche statistiche del segnale che ha imparato. In pratica, sa “come dovrebbero apparire” le strutture dell’immagine e riesce a sopprimere le componenti casuali dovute al rumore, senza intaccare i dettagli reali.

Il risultato finale è un’immagine **denoised**, della stessa dimensione dell’originale, in cui i contorni sono più definiti, le texture più coerenti e le fluttuazioni casuali attenuate. Dal punto di vista tecnico, la fase di inferenza rappresenta quindi **l’applicazione pratica del modello**: la rete non impara più, ma sfrutta le informazioni accumulate per produrre un output affidabile, pronto per **analisi quantitative, visualizzazioni biologiche o ulteriori elaborazioni**, senza richiedere immagini di riferimento perfettamente pulite.

### 2.4 Sperimentazione sul dataset BBBC006



### 2.5 Risultati



