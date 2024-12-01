import torch
from train import Transformer, Vocabulary, generate_vhdl_code_beam_search, create_masks
import pickle
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
import math
import json

# Funzione per calcolare la perplexity
def calculate_perplexity(logits, target, ignore_index):
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=ignore_index)
    return math.exp(loss.item())

# Funzione per calcolare il BLEU Score
def calculate_bleu(reference, generated):
    reference_tokens = reference.split()
    generated_tokens = generated.split()
    return sentence_bleu([reference_tokens], generated_tokens)

# Carica il vocabolario completo
with open('vocab_full.pkl', 'rb') as vocab_file:
    vocab_full = pickle.load(vocab_file)

# Definisci le variabili necessarie per l'intero dataset
padding_idx_full = vocab_full.token2index[Vocabulary.PAD]
bos_idx_full = vocab_full.token2index[Vocabulary.BOS]
max_decoding_length = 512  # Massima lunghezza per la decodifica

# Carica i parametri migliori salvati da Optuna
with open('best_hyperparams.json', 'r') as f:
    best_params = json.load(f)

# Funzione per caricare il modello salvato con i migliori iperparametri
def load_model(model_path, config, padding_idx, bos_idx):
    model = Transformer(
        vocab_size=len(vocab_full),
        hidden_dim=config['hidden_dim'],
        ff_dim=config['ff_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        max_decoding_length=max_decoding_length,
        padding_idx=padding_idx,
        bos_idx=bos_idx,
        dropout_p=config['dropout_p']
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=True)
    model.eval()
    return model

# Caricamento del modello addestrato con i migliori iperparametri
model_path = "llm_vhdl_model_final.pth"  # Questo deve essere il percorso corretto al modello salvato
model = load_model(model_path, best_params, padding_idx_full, bos_idx_full)
print("Checksum test:", sum(p.sum().item() for p in model.parameters()))

# Richiesta input da parte dell'utente per la generazione del codice VHDL
input_text = "implements a ram memory module"
print(f"Richiesta dell'utente: {input_text}")

# Generazione del codice VHDL in base all'input dell'utente
vhdl_code = generate_vhdl_code_beam_search(model, input_text, vocab_full)
print(f"Codice VHDL Generato:\n{vhdl_code}")

# Calcolo della Perplexity
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_tokens = torch.tensor([vocab_full.encode(input_text)], device=device)
generated_tokens = torch.tensor([vocab_full.encode(vhdl_code)], device=device)

src_mask, tgt_mask = create_masks(input_tokens, generated_tokens[:, :-1], padding_idx_full)
print("src_mask: somma degli elementi True:", src_mask.sum().item())
print("tgt_mask: somma degli elementi True:", tgt_mask.sum().item())

masks = torch.load("masks.pth")
src_mask_test = masks["src_mask"]
tgt_mask_test = masks["tgt_mask"]
print("Check maschere tra addestramento e test:", torch.equal(src_mask, src_mask_test), torch.equal(tgt_mask, tgt_mask_test))


logits = model(input_tokens, generated_tokens[:, :-1], src_mask, tgt_mask)
perplexity = calculate_perplexity(logits, generated_tokens[:, 1:], padding_idx_full)
print(f"Perplexity: {perplexity}")

# Calcolo del BLEU Score
reference_code = "-- Reference code for BLEU calculation"
bleu_score = calculate_bleu(reference_code, vhdl_code)
print(f"BLEU Score: {bleu_score}")