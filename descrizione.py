import time
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

# Avvia il tempo per il caricamento del modello e del tokenizer
start_load_time = time.time()

# Carica il modello e il tokenizer
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Calcola il tempo impiegato per il caricamento
end_load_time = time.time()
load_time = end_load_time - start_load_time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def generate_description(code_snippet):

    """
    input_text = (
        f"Explain the following VHDL code in detail. "
        f"Describe the purpose, inputs, outputs, and functionality of the code. "
        f"Provide a clear, accurate, and detailed explanation as if explaining to a VHDL engineer. "
        f"Make sure to explain the logical function performed by the code, how the inputs determine the output, "
        f"and provide an example of the expected behavior.\n\n"
        f"{code_snippet}\n\n"
        f"Description:"
    )
    """
    input_text = (
        f"Provide a clear, accurate, and detailed description of this VHDL code.\n\n "
        f"Describe the purpose, inputs, outputs, and functionality of the code.\n\n "
        f"Focus on the code implementation of the code.\n\n"        
        f"Here is the code: {code_snippet}\n\n"
        f"Description: The following vhdl code implement "
    )

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Genera il testo con parametri aggiustati
    output = model.generate(
        input_ids,
        max_new_tokens=300,  # Aumenta la lunghezza massima dei token generati
        temperature=0.3,
        top_p=0.9,
        num_beams=10,
        no_repeat_ngram_size=2,  # Aggiunto per evitare la ripetizione di frasi
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id  # Aggiunta del segnale di fine
    )

    description = tokenizer.decode(output[0], skip_special_tokens=True)

    return description


# Esempio di codice VHDL
vhdl_code = (
    "library IEEE;\n"
    "use IEEE.STD_LOGIC_1164.ALL;\n"
    "entity AND_Gate is\n"
    "    Port (\n"
    "        A : in STD_LOGIC;\n"
    "        B : in STD_LOGIC;\n"
    "        Y : out STD_LOGIC\n"
    "    );\n"
    "end AND_Gate;\n"
    "architecture Behavioral of AND_Gate is\n"
    "begin\n"
    "    Y <= A and B;\n"
    "end Behavioral;\n"
)

start_time = time.time()  # Inizia a registrare il tempo
description = generate_description(vhdl_code)
end_time = time.time()  # Ferma il tempo dopo l'esecuzione

print(description)
print("Tempo di caricamento (modello e tokenizer): {:.2f} secondi".format(load_time))
print("Tempo di esecuzione: {:.2f} secondi".format(end_time - start_time))

