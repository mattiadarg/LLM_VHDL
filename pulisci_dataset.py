import pandas as pd
import re

# Carica il dataset originale
file_path = 'vhdl_dataset_finale_nan.csv'  # sostituisci con il percorso del tuo file
original_data = pd.read_csv(file_path)

# Crea una copia del dataset per la pulizia e la standardizzazione
vhdl_data = original_data.copy()


# Funzione per applicare rigorosamente le convenzioni VHDL e rimuovere spazi indesiderati
def strict_enforce_vhdl_conventions(content):
    # Enforce the format for commonly used libraries and packages
    content = re.sub(r'\blibrary\s+ieee;', 'library IEEE;', content, flags=re.IGNORECASE)
    content = re.sub(r'\blibrary\s+work;', 'library WORK;', content, flags=re.IGNORECASE)
    content = re.sub(r'\buse\s+ieee\.std_logic_1164\.all;', 'use IEEE.std_logic_1164.ALL;', content, flags=re.IGNORECASE)
    content = re.sub(r'\buse\s+ieee\.std_logic_unsigned\.all;', 'use IEEE.std_logic_unsigned.ALL;', content, flags=re.IGNORECASE)
    content = re.sub(r'\buse\s+ieee\.numeric_std\.all;', 'use IEEE.numeric_std.ALL;', content, flags=re.IGNORECASE)

    # Preserve and enforce consistent indentation for VHDL constructs
    lines = content.split('\n')
    formatted_lines = []
    for line in lines:
        # Trim extra spaces but preserve existing indentation
        trimmed_line = line.strip()
        if trimmed_line.startswith(('entity', 'architecture', 'process', 'if', 'elsif', 'end', 'type', 'signal')):
            # Preserve a logical indentation level
            formatted_lines.append('    ' + trimmed_line)
        elif trimmed_line.startswith(('Port', 'begin')):
            # Align ports and begin statements
            formatted_lines.append('        ' + trimmed_line)
        else:
            # Keep unmodified for all other cases
            formatted_lines.append(trimmed_line)

    # Rejoin lines with appropriate spacing
    content = '\n'.join(formatted_lines)

    # Ensure consistent spacing between code sections
    content = re.sub(r'\n\s*\n', '\n\n', content)  # Replace multiple newlines with a single blank line

    return content


# Applica la funzione di standardizzazione rigorosa alla colonna 'content'
vhdl_data['content'] = vhdl_data['content'].apply(strict_enforce_vhdl_conventions)

# Salva il dataset pulito e rigorosamente standardizzato in un nuovo file CSV
cleaned_file_path = 'vhdl_dataset_finale_pulito.csv'  # sostituisci con il percorso desiderato
vhdl_data.to_csv(cleaned_file_path, index=False)

print(f"Dataset rigorosamente standardizzato e pulito salvato in {cleaned_file_path}")
