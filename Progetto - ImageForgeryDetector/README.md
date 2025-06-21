# Rilevamento Manipolazioni Immagini

Questo progetto fornisce un'applicazione basata su **PyQt5** per il rilevamento di manipolazioni su immagini, come **Copy-Move**, **Inpainting** e **Splicing**.

## Requisiti

Assicurati di avere Python 3 installato e installa le seguenti dipendenze:

```html
pip install opencv-python numpy PyQt5 matplotlib scikit-learn scikit-image
```

## Avvio del Programma

Esegui lo script con:

```html
python3 forgery_detect.py
```

Si aprirà una GUI che permette di selezionare un'immagine e applicare i diversi algoritmi di rilevamento.

## Funzionalità

L'applicazione offre tre metodi di rilevamento:

### Copy-Move Detection

Individua aree duplicate nella stessa immagine utilizzando **SIFT** (Scale-Invariant Feature Transform) e clustering **DBSCAN**:

- Converte l'immagine in scala di grigi.
- Estrae i punti di interesse con SIFT.
- Raggruppa le caratteristiche simili con DBSCAN per evidenziare le aree sospette.

### Inpainting Detection

Identifica modifiche fatte con tecniche di riempimento automatico:

- Applica un filtro **Gaussian Blur**.
- Calcola la differenza tra immagine originale e sfocata.
- Utilizza una soglia per marcare le aree sospette.

### Splicing Detection

Rileva immagini incollate da sorgenti esterne basandosi sulla componente **Cr** (rosso) e **Cb** (blu) dello spazio colore **YCrCb**:

- Estrae le caratteristiche **LBP (Local Binary Pattern)**.
- Calcola la varianza per individuare anomalie.
- Utilizza il filtro **Sobel** per evidenziare i bordi e segmentare le aree sospette.

## Interfaccia Grafica

L'applicazione presenta:

- Un pulsante per **caricare un'immagine**.
- Tre pulsanti per **eseguire le analisi**.
- Una finestra di output con l'immagine e le aree evidenziate.

## Autore

Progetto sviluppato da **Emanuel Ramaci**.
