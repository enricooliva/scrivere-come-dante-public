# Scrivere come Dante — codice dell’esperimento

Questo repository accompagna il libro **Scrivere come Dante**, di **Enrico Oliva**: un esperimento per capire come un modello linguistico impara da un testo e ne genera di nuovo.

Il codice implementa in PyTorch un piccolo Transformer decoder-only con tokenizzazione a caratteri, attenzione causale, più teste di attenzione e generazione autoregressiva. L’esperimento usa la Divina Commedia come corpus. Non ricrea le capacità di Dante e non equivale a un moderno assistente conversazionale.

## Cosa trovi

- `train.py`: modello, addestramento, salvataggio/caricamento dei pesi e generazione.
- `config/default.json`: configurazione dell’esperimento.
- `requirements.txt`: dipendenza per questa versione a caratteri.
- `data/README.md`: preparazione del corpus.
- `checkpoints/README.md`: requisiti dei checkpoint.

**Stato:** corpus e checkpoint non sono distribuiti in questo repository. Per addestrare serve un corpus UTF-8; per generare senza addestramento serve anche un checkpoint compatibile. Il trasferimento della documentazione non costituisce una verifica di esecuzione completa.

## Installazione

Serve Python 3.10 o successivo. Dalla cartella del progetto:

```bash
git clone https://github.com/enricooliva/scrivere-come-dante-public.git
cd scrivere-come-dante-public
python -m venv .venv
```

Attiva l’ambiente su macOS/Linux con `source .venv/bin/activate`; su Windows PowerShell con `.venv\Scripts\Activate.ps1`.

```bash
python -m pip install -r requirements.txt
python train.py --help
```

## Preparare il testo

Ottieni una trascrizione della Divina Commedia da una fonte che ne consenta l’uso e salvala come `data/commedia.txt` in UTF-8. Registra fonte, edizione e trasformazioni applicate: non è ancora documentata qui la provenienza del corpus originale. Non includere automaticamente note o apparati editoriali.

Il vocabolario comprende i caratteri distinti del file. Il primo 90% del testo è usato per training e l’ultimo 10% per validation.

## Addestrare

```bash
python train.py --config config/default.json --data data/commedia.txt --checkpoint checkpoints/gpt_commedia.pt --metrics results/metrics.csv --samples results/samples.txt
```

La configurazione prevede embedding da 384 dimensioni, 6 teste, 6 blocchi e contesto di 256 caratteri. PyTorch usa CUDA se disponibile, altrimenti CPU; i tempi dipendono dall’hardware. Se il checkpoint esiste, il programma ne carica i pesi: per un nuovo esperimento usa un percorso diverso. L’ottimizzatore viene comunque creato nuovamente.

Metriche e campioni vengono aggiunti ai file indicati: usa percorsi distinti per confrontare esperimenti.

## Generare da un checkpoint

```bash
python train.py --skip-train --config config/default.json --data data/commedia.txt --checkpoint checkpoints/gpt_commedia.pt --prompt "Nel mezzo del cammin" --tokens 500 --temperature 1.0
```

Servono un checkpoint compatibile con questa architettura, la configurazione corrispondente e un corpus con lo stesso vocabolario ordinato. Il prompt deve contenere soltanto caratteri presenti nel vocabolario. I risultati sono campionati e possono contenere errori o sequenze prive di senso.

## Libro e codice

Il codice permette di studiare l’implementazione; il libro offre il percorso di spiegazione dell’esperimento e dei suoi limiti. Il testo integrale del libro non è incluso.

## Condizioni d’uso

Non è ancora presente una licenza esplicita per il codice. La disponibilità pubblica non attribuisce automaticamente una licenza di riutilizzo. Le condizioni dell’edizione digitale del corpus vanno considerate separatamente.
