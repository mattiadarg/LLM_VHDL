# LLM-VHDL: Modello Transformer per la Generazione e Validazione di Codice VHDL

## Descrizione del Progetto

LLM-VHDL è un progetto che mira a utilizzare un modello Transformer per generare e validare codice VHDL a partire da descrizioni testuali. Questo sistema combina tecniche di deep learning con l'analisi semantica e sintattica per supportare gli sviluppatori nella creazione di codice VHDL.

Il progetto implementa diverse funzionalità avanzate, tra cui:

- **Tokenizzazione Personalizzata**: Utilizzo di una classe `VHDLTokenizer` per gestire token specifici del linguaggio VHDL.
- **Architettura Transformer**: Comprende un encoder e un decoder con meccanismi di Multi-Head Attention.
- **Training e Valutazione**: Supporta il training con funzionalità di early stopping, ottimizzazione di iperparametri e k-fold cross-validation.
- **Augmentazione Dati**: Generazione di varianti dei dati per migliorare la capacità del modello.
- **Validazione Sintattica**: Verifica della correttezza sintattica del codice generato con il compilatore GHDL.

## Funzionalità Principali

### 1. **Tokenizzazione e Vocabolario**
- **Classe `VHDLTokenizer`**: Tokenizza il codice VHDL usando regex avanzati.
- **Classe `Vocabulary`**: Gestisce il mapping tra token e indici per il modello.

### 2. **Architettura Transformer**
- **Encoder e Decoder**: Implementano blocchi Transformer con attenzione multi-testa e codifiche posizionali sinusoidali.
- **Ottimizzazione**: Include il programma di ottimizzazione NoamOpt per un training più stabile.

### 3. **Augmentazione dei Dati**
- Generazione di varianti per descrizioni e codici VHDL:
  - Sinonimi nelle descrizioni testuali.
  - Rinominazione coerente di identificatori nel codice VHDL.

### 4. **Validazione Sintattica**
- Verifica della sintassi del codice VHDL generato tramite il compilatore GHDL.

## Prerequisiti

Assicurati di avere installati i seguenti pacchetti e strumenti:

- **Librerie Python**:
  - `torch`
  - `numpy`
  - `pandas`
  - `optuna`
  - `nltk`
  - `selenium`
- **Compilatore GHDL** per la validazione del codice VHDL.
