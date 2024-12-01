import pandas as pd
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import time
import os
from datasets import load_dataset
from tqdm import tqdm

# Funzione per verificare la disponibilità della GPU
def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available. GPU will be used.")
        return torch.device("cuda")
    else:
        print("CUDA is not available. CPU will be used.")
        return torch.device("cpu")

# Carica il modello e il tokenizer
start_load_time = time.time()
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
end_load_time = time.time()
load_time = end_load_time - start_load_time

device = check_cuda()
model.to(device)

def generate_description(code_snippet):
    try:
        input_text = (
            f"Provide a clear, concise, and accurate description of the following VHDL code. "
            f"Focus on the purpose, inputs, outputs, and functionality of the code. "
            f"Avoid mentioning any other types of logic gates or irrelevant details.\n\n"
            f"Code:\n{code_snippet}\n\n"
            f"Description: The VHDL code implements "
        )

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        max_length = 2048
        if len(input_ids[0]) > max_length:
            return "Error: Token indices sequence length is longer than the specified maximum sequence length for this model"

        output = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.5,
            top_p=0.85,
            num_beams=3,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        description = tokenizer.decode(output[0], skip_special_tokens=True)
        # Estrarre solo la parte dopo "Description: "
        description_start = description.find("Description: ")
        if description_start != -1:
            description = description[description_start + len("Description: "):]
        return description
    except Exception as e:
        return f"Error generating description: {str(e)}"

# Carica il dataset VHDL
dataset = load_dataset("AWfaw/ai-hdlcoder-dataset-clean", split="train")

# Filtra le entry con size < 5000
filtered_dataset = dataset.filter(lambda x: x['size'] < 5000)

# Stima del numero di entry totali e filtrate
total_entries = len(dataset)
filtered_entries = len(filtered_dataset)

print(f"Numero totale di entry nel dataset: {total_entries}")
print(f"Numero di entry con size < 5000: {filtered_entries}")

# Prendi solo le prime 10 entry per questo esempio
filtered_dataset = filtered_dataset.select(range(20))

# Genera descrizioni per ogni codice VHDL filtrato in batch
start_time = time.time()
batch_size = 10
descriptions = []

# Aggiungi la barra di avanzamento per i batch
with tqdm(total=len(filtered_dataset), desc="Processing", unit="entry") as pbar:
    for i in range(0, len(filtered_dataset), batch_size):
        batch_end = min(i + batch_size, len(filtered_dataset))
        batch = filtered_dataset.select(range(i, batch_end))
        for entry in tqdm(batch, desc=f"Batch {i//batch_size + 1}", leave=False):
            description = generate_description(entry['content'])
            descriptions.append({**entry, 'descrizione': description})
            torch.cuda.empty_cache()  # Libera la memoria GPU
        pbar.update(batch_end - i)

end_time = time.time()

# Converti il dataset in DataFrame
df = pd.DataFrame(descriptions)

# Riformattazione dei dati
df_reformatted = df[['repo_name', 'path', 'size', 'content', 'descrizione']]

# Rimuovi le entry con errori nelle descrizioni
df_cleaned = df_reformatted[~df_reformatted['descrizione'].str.contains("Error")]

# Assicurati che la directory esista
output_file_cleaned = '/Users/mattiadargenio/Downloads/transformer-from-scratch-main/vhdl_dataset_with_descriptions_cleaned.csv'

# Salva il dataset pulito
df_cleaned.to_csv(output_file_cleaned, index=False)

print(f"Dataset pulito salvato in {output_file_cleaned}")

# Visualizza alcune righe del dataset per verificare il contenuto
print("Dataset pulito:")
print(df_cleaned.head(10))

print(f"Tempo di caricamento (modello e tokenizer): {load_time:.2f} secondi")
print(f"Tempo di esecuzione totale: {end_time - start_time:.2f} secondi")
