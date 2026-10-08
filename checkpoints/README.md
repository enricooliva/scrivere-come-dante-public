# Checkpoint

Nessun checkpoint è attualmente distribuito qui. Per usare `--skip-train`, collocare un checkpoint compatibile in `checkpoints/gpt_commedia.pt` e usare la stessa configurazione e lo stesso vocabolario del corpus di addestramento.

Il formato atteso contiene `model_state_dict`, `chars` e `config`. La configurazione salvata non viene selezionata automaticamente: specificare il JSON corretto con `--config`.

Per una futura distribuzione tramite GitHub Release, indicare configurazione, provenienza del corpus, versione del codice e checksum SHA-256 del file effettivamente verificato.
