import logging
import os
import re
import subprocess
import tempfile
import torch
import torch.nn.functional as F
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from LLM_VHDL import Vocabulary


def check_vhdl_syntax(vhdl_code: str) -> bool:
    if not vhdl_code:
        print("Codice VHDL non valido o vuoto.")
        return False

    with open('temp.vhd', 'w') as f:
        f.write(vhdl_code)
    try:
        result = subprocess.run(['ghdl', '-s', 'temp.vhd'], capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            print("Errori di compilazione:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Errore durante la compilazione: {e}")
        return False

def extract_top_module_name(vhdl_code: str) -> str:
    """
    Estrae il nome del modulo top-level dall'entità VHDL.

    Args:
        vhdl_code (str): Il codice VHDL generato.

    Returns:
        str: Il nome dell'entità top-level.

    Raises:
        ValueError: Se non viene trovata alcuna entità nel codice.
    """
    # Espressione regolare per trovare tutte le entità
    pattern = re.compile(r'\bentity\s+(\w+)\s+is', re.IGNORECASE)
    entities = pattern.findall(vhdl_code)

    if not entities:
        raise ValueError("Nessuna entità trovata nel codice VHDL.")

    # Supponiamo che l'ultima entità sia il modulo top-level
    return entities[-1]


def fix_unknown_identifiers(code: str, error_output: str) -> str:
    # Trova gli identificatori sconosciuti nell'output degli errori
    unknown_identifiers = set()
    # Pattern per catturare gli identificatori sconosciuti segnalati da GHDL
    error_pattern = re.compile(r"error:.*?: (.*?): (.*)")

    for line in error_output.splitlines():
        match = error_pattern.search(line)
        if match:
            error_type = match.group(2).strip()
            if "unknown identifier" in error_type or "cannot find symbol" in error_type:
                # Estrae il nome dell'identificatore sconosciuto
                identifier = match.group(1).strip()
                unknown_identifiers.add(identifier)

    # Se non ci sono identificatori sconosciuti, restituisce il codice originale
    if not unknown_identifiers:
        return code

    # Decidi il tipo di dato per gli identificatori sconosciuti
    # In questo esempio, li dichiariamo come segnali di tipo STD_LOGIC
    # Potresti adattare questo comportamento in base al contesto

    # Trova la posizione in cui inserire le nuove dichiarazioni
    # Cerchiamo la sezione delle dichiarazioni nel blocco architecture
    architecture_match = re.search(r"ARCHITECTURE\s+\w+\s+OF\s+\w+\s+IS", code, re.IGNORECASE)
    if architecture_match:
        insert_position = architecture_match.end()
        declarations = "\n"
        for identifier in unknown_identifiers:
            # Aggiungi una dichiarazione per ciascun identificatore sconosciuto
            declarations += f"    SIGNAL {identifier} : STD_LOGIC;\n"

        # Inserisce le dichiarazioni nel codice
        code = code[:insert_position] + declarations + code[insert_position:]
    else:
        # Se non troviamo la sezione architecture, aggiungiamo alla fine (non ideale)
        declarations = "\n"
        for identifier in unknown_identifiers:
            declarations += f"    SIGNAL {identifier} : STD_LOGIC;\n"
        code += declarations

    return code


def check_and_correct_vhdl_syntax(code: str, max_iterations=5) -> str:
    iteration = 0
    while iteration < max_iterations:
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.vhd', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_name = temp_file.name

        try:
            # Esegue GHDL per analizzare il codice VHDL
            result = subprocess.run(['ghdl', '-a', temp_file_name], capture_output=True, text=True)
            if result.returncode != 0:
                # Analizza gli errori e applica correzioni specifiche
                error_output = result.stderr

                # Flag per controllare se sono state apportate modifiche
                changes_made = False

                # Salva il codice precedente per confrontare le modifiche
                previous_code = code

                # Applica le funzioni di correzione
                code = fix_entity_port_declaration(code)
                code = fix_assignments(code)
                code = fix_extra_semicolons(code)

                # Incrementa il contatore di iterazioni
                iteration += 1

                # Se il codice non è cambiato, interrompi il ciclo
                if code == previous_code:
                    break
            else:
                # Il codice è sintatticamente corretto
                return code
        finally:
            # Elimina il file temporaneo
            os.remove(temp_file_name)

    # Se dopo le iterazioni il codice non è corretto, restituisce comunque il codice
    return code


def fix_type_declaration(code: str) -> str:
    code = re.sub(r'TYPE\s+ram_type\s+IS\s+TYPE\s+ram_type\s+IS', 'TYPE ram_type IS', code, flags=re.IGNORECASE)
    return code


def fix_entity_port_declaration(code: str) -> str:
    """
    Corregge la dichiarazione delle porte di un'entità VHDL.
    """
    # Espressione regolare per catturare l'intera dichiarazione dell'entità
    entity_pattern = re.compile(
        r'(ENTITY\s+\w+\s+IS\s+PORT\s*\()(.*?)(\)\s*;\s*END\s+\w+\s*;)',
        re.DOTALL | re.IGNORECASE
    )

    match = entity_pattern.search(code)

    if not match:
        # Nessuna entità trovata, ritorna il codice originale
        return code

    entity_header = match.group(1)  # "ENTITY register_file IS PORT ("
    ports_block = match.group(2)  # Contenuto delle porte
    entity_footer = match.group(3)  # "); END register_file;"

    # Aggiunge una parentesi chiusa mancante alla fine delle porte se necessario
    if ports_block.count('(') > ports_block.count(')'):
        ports_block += ')'

    # Split delle porte sul punto e virgola
    ports = [port.strip() for port in ports_block.split(';') if port.strip()]

    # Aggiunge il punto e virgola mancante alle porte
    for i in range(len(ports)):
        if not ports[i].endswith(';') and i < len(ports) - 1:
            ports[i] += ';'

    # Ricostruisce la dichiarazione delle porte
    ports_fixed = '\n    '.join(ports)

    # Ricostruisce l'entità
    new_entity = f"{entity_header}\n    {ports_fixed}\n{entity_footer}"

    # Sostituisce l'entità nel codice
    code = entity_pattern.sub(new_entity, code)

    return code


def remove_duplicate_end_statements(code: str) -> str:
    """
    Rimuove dichiarazioni `END` duplicate nel codice VHDL,
    preservando quelle relative all'entità e all'architettura.
    """
    # Espressione regolare per catturare dichiarazioni `END` con il nome dell'entità o dell'architettura
    entity_or_arch_end_pattern = re.compile(r'\bEND\s+(\w+);\s*', re.IGNORECASE)
    matches = entity_or_arch_end_pattern.findall(code)

    if not matches:
        # Nessuna dichiarazione `END` trovata
        return code

    # Identifica l'entità e l'architettura principali
    entity_name = matches[0] if matches else None
    arch_name = matches[-1] if len(matches) > 1 else None

    # Espressione regolare per identificare le dichiarazioni `END` generiche
    generic_end_pattern = re.compile(r'\bEND\b;\s*', re.IGNORECASE)

    # Itera attraverso le linee del codice
    code_lines = code.splitlines()
    cleaned_lines = []
    for line in code_lines:
        # Ignora dichiarazioni `END` generiche duplicate
        if generic_end_pattern.fullmatch(line.strip()):
            continue
        # Mantieni dichiarazioni `END` valide con il nome dell'entità o dell'architettura
        if entity_or_arch_end_pattern.search(line):
            if entity_name in line or arch_name in line:
                cleaned_lines.append(line)
                continue
        # Mantieni tutte le altre linee
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)




def fix_assignments(code: str) -> str:
    # Aggiunge il punto e virgola mancante alle assegnazioni
    code_lines = code.split('\n')
    for i, line in enumerate(code_lines):
        stripped_line = line.strip()
        if ('<=' in stripped_line or ':=' in stripped_line) and not stripped_line.endswith(';'):
            code_lines[i] = line + ';'
    code = '\n'.join(code_lines)
    return code





def fix_end_statements(code: str) -> str:
    # Corregge 'END register_file;' in posizioni errate all'interno dell'architettura
    code = re.sub(r'END\s+register_file\s*;', 'END IF;', code, flags=re.IGNORECASE)
    # Assicura che 'END PROCESS' termini con ';'
    code = re.sub(r'END\s+PROCESS\s*(?!;)', 'END PROCESS;', code, flags=re.IGNORECASE)
    # Assicura che 'END arqregfile' termini con ';'
    code = re.sub(r'END\s+arqregfile\s*(?!;)', 'END arqregfile;', code, flags=re.IGNORECASE)
    return code


def balance_parentheses(code: str) -> str:
    """
    Bilancia le parentesi nel codice VHDL, evitando di modificare la dichiarazione dell'entità e del PORT,
    e preservando la posizione delle dichiarazioni delle librerie.
    """
    # Trova la dichiarazione dell'entità
    entity_pattern = re.compile(
        r'ENTITY\s+\w+\s+IS\s+PORT\s*\(.*?\);\s*END\s+\w+;',
        re.DOTALL | re.IGNORECASE
    )
    match = entity_pattern.search(code)
    if match:
        code_before_entity = code[:match.start()]
        entity_decl = match.group()
        code_after_entity = code[match.end():]
    else:
        # Nessuna entità trovata, processa tutto il codice
        code_before_entity = ''
        entity_decl = ''
        code_after_entity = code

    # Bilanciamo le parentesi nel code_after_entity
    code_after_entity = balance_parentheses_in_code(code_after_entity)

    # Ricostruisce il codice nell'ordine corretto
    code = code_before_entity + entity_decl + code_after_entity
    return code


def balance_parentheses_in_code(code: str) -> str:
    """
    Bilancia le parentesi nel codice VHDL, processando solo il codice passato.
    """
    # Conta il numero totale di parentesi aperte e chiuse
    total_open = code.count('(')
    total_close = code.count(')')

    # Calcola la differenza
    diff = total_open - total_close

    if diff > 0:
        # Se ci sono più parentesi aperte, aggiungiamo parentesi chiuse alla fine
        code += ')' * diff
    elif diff < 0:
        # Se ci sono più parentesi chiuse, rimuoviamo parentesi chiuse in eccesso
        for _ in range(abs(diff)):
            index = code.rfind(')')
            if index != -1:
                code = code[:index] + code[index+1:]
            else:
                break  # Non ci sono più parentesi da rimuovere
    # Restituisce il codice bilanciato
    return code


def fix_extra_semicolons(code: str) -> str:
    """
    Rimuove i punti e virgola superflui nel codice VHDL,
    evitando di eliminare quelli necessari nelle dichiarazioni e assegnazioni.
    """
    # Rimuove solo punti e virgola ridondanti consecutivi (;;)
    code = re.sub(r';\s*;', ';', code)

    # Rimuove punti e virgola prima di ')', ma solo nei contesti corretti
    # Assicura di non modificare dichiarazioni di porte o assegnazioni
    def remove_semicolon_before_parenthesis(match):
        content = match.group(1)
        if any(keyword in content.upper() for keyword in ["STD_LOGIC", "STD_LOGIC_VECTOR", "SIGNAL"]):
            return match.group(0)  # Non rimuove il ';' in dichiarazioni di segnali
        return content + ')'

    code = re.sub(r'(.*?);\s*\)', remove_semicolon_before_parenthesis, code)

    # Preserva i punti e virgola in dichiarazioni di segnali, tipi e assegnazioni
    # Assicura che ogni dichiarazione di tipo o segnale termini con `;`
    lines = code.splitlines()
    fixed_lines = []
    for line in lines:
        stripped = line.strip()
        if (stripped.startswith("SIGNAL") or stripped.startswith("TYPE") or '<=' in stripped or ':=' in stripped) and not stripped.endswith(';'):
            fixed_lines.append(line + ';')  # Aggiunge il ';' se mancante
        else:
            fixed_lines.append(line)
    return '\n'.join(fixed_lines)




def evaluate_synthesis_success(test_inputs, vhdl_code):
    # total = len(test_inputs)
    success = 0

    if check_vhdl_syntax(vhdl_code) and synthesize_vhdl_code(vhdl_code):
        success += 1

    synthesis_success_rate = success # / total
    print(f'Tasso di Successo della Sintesi: {synthesis_success_rate:.2f}')
    return synthesis_success_rate


def format_vhdl_with_VHDLFormatter(vhdl_code: str) -> str:
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Esecuzione senza interfaccia grafica
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get("https://g2384.github.io/VHDLFormatter/")

        wait = WebDriverWait(driver, 10)

        # Trova e clicca "Show More Settings"
        show_more_settings_button = wait.until(EC.element_to_be_clickable((By.ID, "settings_control")))
        show_more_settings_button.click()

        # Seleziona "Customise Indentation"
        customise_indentation_checkbox = wait.until(EC.element_to_be_clickable((By.ID, "customise_indentation")))
        customise_indentation_checkbox.click()

        # Imposta l'indentazione
        indentation_input = driver.find_element(By.ID, "customise_indentation")
        indentation_input.clear()
        indentation_input.send_keys("\t\t\t\t\t")

        # Inserisci il codice VHDL nel textarea
        textarea = wait.until(EC.presence_of_element_located((By.TAG_NAME, "textarea")))
        textarea.clear()
        textarea.send_keys(vhdl_code)

        # Clicca il pulsante "Start"
        start_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='button' and @value='Start']")))
        start_button.click()

        # Attendi l'output formattato
        output_code = wait.until(EC.presence_of_element_located((By.ID, "vhdl")))
        formatted_code = output_code.text
        return formatted_code

    except Exception as e:
        logging.error(f"Errore durante l'esecuzione di VHDLFormatter: {e}")
        return ""
    finally:
        driver.quit()


def remove_spaces_in_library_declaration(code: str) -> str:
    """
    Rimuove gli spazi extra nella dichiarazione delle librerie
    nel formato `USE IEEE . std_logic_1164 . ALL;` -> `USE IEEE.std_logic_1164.ALL;`
    """
    # Rimuove spazi attorno ai punti specificamente nelle dichiarazioni 'USE'
    code = re.sub(r'\buse\s+([\w]+)\s*\.\s*([\w]+)\s*\.\s*([\w]+)\s*;', r'use \1.\2.\3;', code, flags=re.IGNORECASE)
    code = re.sub(r'\buse\s+([\w]+)\s*\.\s*([\w]+)\s*;', r'use \1.\2;', code, flags=re.IGNORECASE)
    return code


def remove_special_tokens(vhdl_code):
    return vhdl_code.replace('< BOS >', '').replace('< EOS >', '').strip()


def remove_spaces_in_hex_declarations(vhdl_code):
    import re
    return re.sub(r'\b(x)\s+"', r'\1"', vhdl_code)


def replace_std_logic_unsigned(code: str) -> str:
    """
    Sostituisce l'uso di `std_logic_unsigned` con `numeric_std`
    """
    code = re.sub(r'\buse\s+IEEE\.STD_LOGIC_UNSIGNED\.ALL;', 'use IEEE.NUMERIC_STD.ALL;', code, flags=re.IGNORECASE)
    return code


def replace_conv_integer(code: str) -> str:
    """
    Sostituisce `conv_integer` con `to_integer(unsigned(...))` e bilancia le parentesi.
    """
    def replace_and_balance(match):
        signal = match.group(1)
        replacement = f"to_integer(unsigned({signal}))"
        # Bilancia le parentesi aggiungendo una parentesi chiusa mancante
        open_count = replacement.count('(')
        close_count = replacement.count(')')
        if open_count > close_count:
            replacement += ')' * (open_count - close_count)
        return replacement

    # Sostituzione con bilanciamento
    code = re.sub(r'conv_integer\s*\(\s*(\w+)\s*\)', replace_and_balance, code, flags=re.IGNORECASE)
    return code


def synthesize_vhdl_code(vhdl_code: str) -> bool:
    """
    Sintetizza il codice VHDL utilizzando GHDL e genera una netlist Verilog.

    Args:
        vhdl_code (str): Il codice VHDL generato.

    Returns:
        bool: True se la sintesi è riuscita e la netlist è stata creata, False altrimenti.
    """
    if not vhdl_code:
        logging.error("Codice VHDL non valido o vuoto.")
        return False

    # Controllo della presenza di 'std_logic_unsigned' nel codice VHDL
    synopsys_flag = ""
    if "std_logic_unsigned" in vhdl_code:
        synopsys_flag = "-fsynopsys"
        logging.info("Flag '-fsynopsys' abilitato a causa della presenza di 'std_logic_unsigned'.")

    # Salva il codice VHDL su un file temporaneo
    temp_vhdl = 'temp.vhd'
    with open(temp_vhdl, 'w') as f:
        f.write(vhdl_code)
    logging.info(f"Codice VHDL salvato in '{temp_vhdl}'.")

    try:
        # Estrai il nome del modulo top-level
        top_module = extract_top_module_name(vhdl_code)
        logging.info(f"Modulo top-level estratto: {top_module}")

        # Passaggio 1: Analizza (compila) il file VHDL
        analyze_cmd = ['ghdl', '-a']
        if synopsys_flag:
            analyze_cmd.append(synopsys_flag)
        analyze_cmd.append(temp_vhdl)

        logging.info(f"Esecuzione comando di analisi: {' '.join(analyze_cmd)}")
        analyze_result = subprocess.run(analyze_cmd, capture_output=True, text=True)

        if analyze_result.returncode != 0:
            logging.error("Errori durante l'analisi del codice VHDL con GHDL:")
            logging.error(analyze_result.stderr)
            return False
        else:
            logging.info("Analisi del codice VHDL completata con successo.")

        # Passaggio 2: Sintetizza l'entità top-level e genera la netlist Verilog
        synth_cmd = ['ghdl', '--synth']
        if synopsys_flag:
            synth_cmd.append(synopsys_flag)
        synth_cmd.append('--latches')  # Aggiungi l'opzione --latches prima del nome dell'entità
        synth_cmd.append('--out=vhdl')  # Specifica il formato di output
        synth_cmd.append(top_module)  # Aggiungi il nome dell'entità alla fine

        logging.info(f"Esecuzione comando di sintesi: {' '.join(synth_cmd)}")
        # Stampa di debug per synth_cmd
        logging.debug(f"synth_cmd: {synth_cmd}")

        # Esegui il comando di sintesi e cattura l'output
        synth_result = subprocess.run(synth_cmd, capture_output=True, text=True)

        if synth_result.returncode != 0:
            logging.error("Errori durante la sintesi del codice VHDL con GHDL:")
            logging.error(synth_result.stderr)
            return False
        else:
            logging.info("Sintesi GHDL completata con successo.")
            logging.debug(synth_result.stdout)
            if synth_result.stderr:
                logging.warning("Warning durante la sintesi GHDL:")
                logging.warning(synth_result.stderr)

            # Salva la netlist dall'output standard
            netlist_file = f"{top_module}.v"
            with open(netlist_file, 'w') as f:
                f.write(synth_result.stdout)
            logging.info(f"Netlist Verilog generata: {netlist_file}")
            success = True

    except ValueError as ve:
        logging.error(f"Errore nell'estrazione del modulo top-level: {ve}")
        success = False
    except FileNotFoundError:
        logging.error("GHDL non è installato o non è presente nel PATH.")
        success = False
    except Exception as ex:
        logging.error(f"Errore inaspettato durante la sintesi: {ex}")
        success = False
    finally:
        # Pulizia del file temporaneo
        if os.path.exists(temp_vhdl):
            os.remove(temp_vhdl)
            logging.info(f"File temporaneo rimosso: {temp_vhdl}")

    return success


def fix_vhdl_for_synthesis(vhdl_code: str) -> str:
    """
    Adjusts VHDL code to fix common synthesis issues:
    - Fix bit-width mismatches in comparisons
    - Ensure all signals are assigned in all paths to avoid latch inference
    """
    # Configura il logging per la funzione
    logger = logging.getLogger('fix_vhdl_for_synthesis')
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Step 1: Parse signal and port declarations to get signal bit-widths
    # Map: signal name -> (width, type)
    signals = {}
    array_signals = set()  # Inizializza l'insieme per i segnali array

    # Parse signal declarations including arrays
    signal_decl_pattern = re.compile(
        r'\bsignal\s+(\w+)\s*:\s*(\w+)\s*(.*?);',
        re.IGNORECASE
    )

    for match in signal_decl_pattern.finditer(vhdl_code):
        sig_name = match.group(1)
        sig_type = match.group(2)
        sig_details = match.group(3)
        if 'ARRAY' in sig_type.upper():
            array_signals.add(sig_name)
            signals[sig_name] = {'type': 'ARRAY'}
            logger.debug(f"Detected ARRAY signal '{sig_name}'.")
        elif 'STD_LOGIC_VECTOR' in sig_details.upper():
            vector_match = re.search(r'\(\s*(\d+)\s+(DOWNTO|TO)\s+(\d+)\s*\)', sig_details, re.IGNORECASE)
            if vector_match:
                msb = int(vector_match.group(1))
                lsb = int(vector_match.group(3))
                width = abs(msb - lsb) + 1
                signals[sig_name] = {'width': width, 'type': 'STD_LOGIC_VECTOR'}
                logger.debug(f"Detected STD_LOGIC_VECTOR signal '{sig_name}' with width {width}.")
        elif 'STD_LOGIC' in sig_details.upper():
            signals[sig_name] = {'width': 1, 'type': 'STD_LOGIC'}
            logger.debug(f"Detected STD_LOGIC signal '{sig_name}' with width 1.")

    # Parse port declarations
    port_decl_pattern = re.compile(
        r'\b(\w+)\s*:\s*(IN|OUT|INOUT)\s+(STD_LOGIC_VECTOR\s*\(\s*(\d+)\s+(DOWNTO|TO)\s+(\d+)\s*\)|STD_LOGIC)',
        re.IGNORECASE
    )

    for match in port_decl_pattern.finditer(vhdl_code):
        sig_name = match.group(1)
        direction = match.group(2)
        sig_type = match.group(3)
        if 'STD_LOGIC_VECTOR' in sig_type.upper():
            msb = int(match.group(4))
            lsb = int(match.group(6))
            width = abs(msb - lsb) + 1
            signals[sig_name] = {'width': width, 'type': 'STD_LOGIC_VECTOR', 'direction': direction.upper()}
            logger.debug(f"Detected STD_LOGIC_VECTOR port '{sig_name}' with width {width}.")
        else:
            signals[sig_name] = {'width': 1, 'type': 'STD_LOGIC', 'direction': direction.upper()}
            logger.debug(f"Detected STD_LOGIC port '{sig_name}' with width 1.")

    # Step 2: Fix comparison bit-width mismatches
    compare_pattern = re.compile(
        r'(\b\w+\b)\s*(/=|=)\s*"([01]+)"'
    )

    def replace_comparisons(match):
        signal = match.group(1)
        operator = match.group(2)
        literal = match.group(3)
        if signal in signals:
            sig_width = signals[signal]['width']
            lit_length = len(literal)
            if lit_length < sig_width:
                adjusted_literal = literal.zfill(sig_width)
                logger.info(f"Adjusted comparison for signal '{signal}': '{signal} {operator} \"{literal}\"' -> '{signal} {operator} \"{adjusted_literal}\"'")
                return f'{signal} {operator} "{adjusted_literal}"'
            elif lit_length > sig_width:
                adjusted_literal = literal[-sig_width:]
                logger.info(f"Trimmed comparison for signal '{signal}': '{signal} {operator} \"{literal}\"' -> '{signal} {operator} \"{adjusted_literal}\"'")
                return f'{signal} {operator} "{adjusted_literal}"'
        return match.group(0)

    vhdl_code = compare_pattern.sub(replace_comparisons, vhdl_code)

    # Step 3: Prevent latch inference by assigning default values at start of processes
    process_pattern = re.compile(
        r'PROCESS\s*\(([^)]*)\)\s*BEGIN\s*(.*?)END\s+PROCESS;',
        re.IGNORECASE | re.DOTALL
    )
    processes = process_pattern.findall(vhdl_code)

    for sensitivity_list, body in processes:
        # Find signals being assigned in the process
        assign_pattern = re.compile(
            r'(\b\w+\b)\s*<=',
            re.IGNORECASE
        )
        assigns = assign_pattern.findall(body)
        assigns = set(assigns)

        # Prepare default assignments
        default_assignments = ""
        for sig in assigns:
            if sig in signals:
                sig_info = signals[sig]
                if sig_info['type'] == 'STD_LOGIC_VECTOR':
                    if sig in array_signals:
                        default_val = "(others => (others => '0'))"
                    else:
                        default_val = "(others => '0')"
                elif sig_info['type'] == 'STD_LOGIC':
                    default_val = "'0'"
                elif sig_info['type'] == 'ARRAY':
                    default_val = "(others => (others => '0'))"
                else:
                    default_val = "'0'"
                default_assignments += f'    {sig} <= {default_val};\n'
                logger.debug(f"Prepared default assignment for signal '{sig}': '{sig} <= {default_val};'")

        if default_assignments:
            # Insert default assignments at the start of the process body
            body_lines = body.splitlines()
            new_body = '\n' + default_assignments + '\n'.join(body_lines) + '\n'
            # Reconstruct the process
            full_process_pattern = re.compile(
                r'(PROCESS\s*\([^)]*\)\s*BEGIN\s*)(.*?)\s*END\s+PROCESS;',
                re.IGNORECASE | re.DOTALL
            )
            replacement = r'\1' + new_body + r'END PROCESS;'
            vhdl_code, num_subs = full_process_pattern.subn(replacement, vhdl_code, count=1)
            if num_subs > 0:
                logger.info(f"Inserted default assignments in process with sensitivity list: ({sensitivity_list})")

    return vhdl_code


def insert_begin_after_architecture(code: str) -> str:
    # Inserisce 'BEGIN' dopo le dichiarazioni dell'architettura e prima delle istruzioni
    architecture_pattern = re.compile(
        r'(ARCHITECTURE\s+\w+\s+OF\s+\w+\s+IS)(.*?)(BEGIN|PROCESS)',
        re.DOTALL | re.IGNORECASE
    )

    def add_begin(match):
        header = match.group(1)
        declarations = match.group(2)
        next_keyword = match.group(3)
        return f"{header}{declarations}\nBEGIN\n"

    code = architecture_pattern.sub(add_begin, code)
    return code


def sanitize_vhdl_code(code: str) -> str:
    # Rimuove i token speciali come <BOS> e <EOS>
    code = remove_special_tokens(code)
    print("Dopo remove_special_tokens:\n", code)

    # Rimuove gli spazi in eccesso nelle dichiarazioni delle librerie
    code = remove_spaces_in_library_declaration(code)
    print("Dopo remove_spaces_in_library_declaration:\n", code)

    # Sostituisce 'std_logic_unsigned' con 'numeric_std'
    code = replace_std_logic_unsigned(code)
    print("Dopo replace_std_logic_unsigned:\n", code)

    # Rimuove gli spazi nelle dichiarazioni esadecimali
    code = remove_spaces_in_hex_declarations(code)
    print("Dopo remove_spaces_in_hex_declarations:\n", code)

    # Sostituisce 'conv_integer' con 'to_integer'
    code = replace_conv_integer(code)
    print("Dopo replace_conv_integer:\n", code)

    # Corregge la dichiarazione del tipo duplicata
    code = fix_type_declaration(code)
    print("Dopo fix_type_declaration:\n", code)

    # Corregge le assegnazioni e le chiamate di funzione
    code = fix_assignments(code)
    print("Dopo fix_assignments:\n", code)

    # Rimuove punti e virgola in eccesso
    code = fix_extra_semicolons(code)
    print("Dopo fix_extra_semicolons:\n", code)

    # Inserisce 'BEGIN' dopo l'architettura
    code = insert_begin_after_architecture(code)
    print("Dopo insert_begin_after_architecture:\n", code)

    # Bilancia le parentesi dopo le correzioni
    code = balance_parentheses(code)
    print("Dopo balance_parentheses:\n", code)

    # Rimuove dichiarazioni `END` duplicate
    code = remove_duplicate_end_statements(code)
    print("Dopo remove_duplicate_end_statements:\n", code)

    # Correggere il punto e virgola nella dichiarazione del tipo
    code = code.replace('ARRAY(0 TO 39); OF', 'ARRAY(0 TO 39) OF')
    print("Dopo correzione ARRAY(0 TO 39); OF:\n", code)

    # Correggere 'BEGIN PROCESS' in 'PROCESS'
    code = code.replace('BEGIN PROCESS', 'PROCESS')
    print("Dopo correzione 'BEGIN PROCESS':\n", code)

    # Sostituisci 'conv_integer(to_integer(unsigned(x)))' con 'to_integer(unsigned(x))'
    code = re.sub(r'conv_integer\s*\(\s*to_integer\s*\(\s*unsigned\s*\(\s*(\w+)\s*\)\s*\)\s*\)',
                  r'to_integer(unsigned(\1)))', code, flags=re.IGNORECASE)
    print("Dopo sostituzione conv_integer annidato:\n", code)

    # Corregge assegnazioni VHDL con parentesi mancanti prima di '<='
    code = re.sub(r'(reg\s*\(\s*to_integer\s*\(\s*unsigned\s*\(\s*\w+\s*\)\s*\))\s*<=', r'\1) <=', code)
    print("Dopo correzione parentesi mancanti in assegnazioni:\n", code)

    # Correggere le parentesi mancanti nelle dichiarazioni delle porte
    code_lines = code.split('\n')
    new_code_lines = []
    for line in code_lines:
        # Correggere le dichiarazioni delle porte mancanti di ')'
        if 'STD_LOGIC_VECTOR(' in line and line.strip().endswith(';'):
            if line.count('(') > line.count(')'):
                line = line.rstrip(';') + ');'
        # Correggere le chiamate di funzione mancanti di ')'
        if re.search(r'\b(to_integer|unsigned|conv_integer)\([^\)]*$', line):
            missing_parentheses = line.count('(') - line.count(')')
            line += ')' * missing_parentheses
        # Rimuovere punti e virgola extra
        line = line.replace(';;', ';')
        new_code_lines.append(line)
    code = '\n'.join(new_code_lines)
    print("Dopo correzione delle parentesi mancanti:\n", code)

    # Rimuovere parentesi chiuse extra alla fine del codice
    while code.endswith(')'):
        code = code[:-1]
    print("Dopo rimozione parentesi chiuse extra:\n", code)

    return code




def evaluate_functional_accuracy(finale):
    # total = len(test_inputs)
    success = 0

    check_and_correct_vhdl_syntax(finale)

    # Verifica sintattica prima della verifica funzionale
    if check_vhdl_syntax(finale):
        try:
            if check_functional_accuracy(finale):
                success += 1
        except Exception as e:
            print(f"Errore durante il test funzionale: {e}")

    functional_accuracy = success # / total
    print(f'Accuratezza Funzionale: {functional_accuracy:.2f}')
    return functional_accuracy


def check_functional_accuracy(vhdl_code):
    try:
        # Crea file temporanei per il codice VHDL e il testbench
        with tempfile.NamedTemporaryFile(delete=False, suffix=".vhd") as vhdl_file, \
                tempfile.NamedTemporaryFile(delete=False, suffix=".vhd") as tb_file:
            vhdl_file.write(vhdl_code.encode('utf-8'))
            vhdl_file_path = vhdl_file.name

            testbench_code = generate_testbench()
            tb_file.write(testbench_code.encode('utf-8'))
            tb_file_path = tb_file.name

        # Compila il codice VHDL
        result_compile_vhdl = subprocess.run(['ghdl', '-a', vhdl_file_path], capture_output=True, text=True)
        if result_compile_vhdl.returncode != 0:
            print(f"Errore nella compilazione del codice VHDL: {result_compile_vhdl.stderr}")
            return False

        # Compila il testbench
        result_compile_tb = subprocess.run(['ghdl', '-a', tb_file_path], capture_output=True, text=True)
        if result_compile_tb.returncode != 0:
            print(f"Errore nella compilazione del testbench: {result_compile_tb.stderr}")
            return False

        # Elabora il testbench
        result_elab_tb = subprocess.run(['ghdl', '-e', 'testbench'], capture_output=True, text=True)
        if result_elab_tb.returncode != 0:
            print(f"Errore nell'elaborazione del testbench: {result_elab_tb.stderr}")
            return False

        # Esegue la simulazione
        result_sim = subprocess.run(['ghdl', '-r', 'testbench', '--assert-level=error'], capture_output=True, text=True)
        if result_sim.returncode == 0:
            return True  # Test funzionali passati
        else:
            print(f"Errore durante la simulazione: {result_sim.stderr}")
            return False
    except Exception as e:
        print(f"Errore di sistema durante la verifica funzionale: {e}")
        return False
    finally:
        # Pulizia dei file temporanei
        if 'vhdl_file_path' in locals() and os.path.exists(vhdl_file_path):
            os.remove(vhdl_file_path)
        if 'tb_file_path' in locals() and os.path.exists(tb_file_path):
            os.remove(tb_file_path)


def generate_testbench():
    # Genera un testbench per il modulo register_file
    testbench_code = """
    LIBRARY IEEE;
    USE IEEE.std_logic_1164.ALL;
    USE IEEE.numeric_std.ALL;

    ENTITY testbench IS
    END testbench;

    ARCHITECTURE behavior OF testbench IS 

        -- Component declaration
        COMPONENT register_file
            PORT(
                Wren : IN STD_LOGIC;
                rst : IN STD_LOGIC;
                rs1 : IN STD_LOGIC_VECTOR(5 DOWNTO 0);
                rs2 : IN STD_LOGIC_VECTOR(5 DOWNTO 0);
                rd : IN STD_LOGIC_VECTOR(5 DOWNTO 0);
                data : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
                crs1 : OUT STD_LOGIC_VECTOR(31 DOWNTO 0);
                crs2 : OUT STD_LOGIC_VECTOR(31 DOWNTO 0);
                crd : OUT STD_LOGIC_VECTOR(31 DOWNTO 0)
            );
        END COMPONENT;

        -- Signals
        SIGNAL Wren : STD_LOGIC := '0';
        SIGNAL rst : STD_LOGIC := '0';
        SIGNAL rs1 : STD_LOGIC_VECTOR(5 DOWNTO 0) := (others => '0');
        SIGNAL rs2 : STD_LOGIC_VECTOR(5 DOWNTO 0) := (others => '0');
        SIGNAL rd : STD_LOGIC_VECTOR(5 DOWNTO 0) := (others => '0');
        SIGNAL data : STD_LOGIC_VECTOR(31 DOWNTO 0) := (others => '0');
        SIGNAL crs1 : STD_LOGIC_VECTOR(31 DOWNTO 0);
        SIGNAL crs2 : STD_LOGIC_VECTOR(31 DOWNTO 0);
        SIGNAL crd : STD_LOGIC_VECTOR(31 DOWNTO 0);

    BEGIN
        -- Instantiate the Unit Under Test (UUT)
        uut: register_file PORT MAP (
            Wren => Wren,
            rst => rst,
            rs1 => rs1,
            rs2 => rs2,
            rd => rd,
            data => data,
            crs1 => crs1,
            crs2 => crs2,
            crd => crd
        );

        -- Stimulus process
        stim_proc: process
        begin
            -- Test case 1: Reset the register file
            rst <= '1';
            wait for 10 ns;
            assert crs1 = x"00000000" and crs2 = x"00000000" and crd = x"00000000" 
                report "Test case 1 (Reset) failed" severity error;

            -- Test case 2: Write to register and read from rs1
            rst <= '0';
            Wren <= '1';
            rd <= "000001";
            data <= x"DEADBEEF";
            wait for 10 ns;
            Wren <= '0';
            rs1 <= "000001";
            wait for 10 ns;
            assert crs1 = x"DEADBEEF" report "Test case 2 (Write/Read rs1) failed" severity error;

            -- Test case 3: Write to a second register and read from rs2
            Wren <= '1';
            rd <= "000010";
            data <= x"CAFEBABE";
            wait for 10 ns;
            Wren <= '0';
            rs2 <= "000010";
            wait for 10 ns;
            assert crs2 = x"CAFEBABE" report "Test case 3 (Write/Read rs2) failed" severity error;

            -- Test case 4: Write to rd and verify crd
            Wren <= '1';
            rd <= "000011";
            data <= x"12345678";
            wait for 10 ns;
            Wren <= '0';
            rs1 <= "000011";
            wait for 10 ns;
            assert crd = x"12345678" report "Test case 4 (Verify crd) failed" severity error;

            -- Fine della simulazione
            wait;
        end process;

    END;
    """
    return testbench_code



if __name__ == '__main__':
    vhdl_code = """
   <BOS> library IEEE; use IEEE . STD_LOGIC_1164 . ALL; use IEEE . STD_LOGIC_UNSIGNED . ALL; entity register_file is Port( Wren: in STD_LOGIC; rst: in STD_LOGIC; rs1: in STD_LOGIC_VECTOR( 5 downto 0); rs2: in STD_LOGIC_VECTOR( 5 downto 0); rd: in STD_LOGIC_VECTOR( 5 downto 0); data: in STD_LOGIC_VECTOR( 31 downto 0); crs1: out STD_LOGIC_VECTOR( 31 downto 0); crs2: out STD_LOGIC_VECTOR( 31 downto 0); crd: out STD_LOGIC_VECTOR( 31 downto 0); end register_file; architecture ArqRegFile of register_file is type ram_type is array( 0 to 39) of std_logic_vector( 31 downto 0); signal reg: ram_type := ( others =>  x "00000000"); begin process( rst, rs1, rs2, rd, data) begin if( rst =  '0') then crs1 <=  reg( conv_integer( conv_integer( rs1)); crs2 <=  reg( conv_integer( rs2)); crd <=  reg( conv_integer( rd)); if( rd /  =  "00000" and Wren =  '1') then reg( conv_integer( rd) <=  data; end if; elsif( rst =  '1') then crs1 <=  x "00000000"; crs2 <=  x "00000000"; reg <= ( others =>  x "00000000"); end if; end process; end ArqRegFile; <EOS>
    """

    input_text = "implements a ram"

    formattato = format_vhdl_with_VHDLFormatter(vhdl_code)
    print(f"Formattato:\n{formattato}")

    # Corregge la dichiarazione delle porte dell'entità
    formattato = fix_entity_port_declaration(formattato)
    print(f"fix entity port declaration:\n{formattato}")

    sanificato = sanitize_vhdl_code(formattato)
    print(f"Sanificato:\n{sanificato}")

    fixato = fix_vhdl_for_synthesis(sanificato)
    print(f"Fixato:\n{fixato}")

    finale = check_and_correct_vhdl_syntax(fixato)
    print(f"Finale:\n{finale}")

    print("SINTESI \n")
    synthesis_success_rate = evaluate_synthesis_success(input_text, finale)
    success = synthesize_vhdl_code(finale)
    if success:
        logging.info("Sintesi completata con successo!")
    else:
        logging.error("Sintesi fallita.")

    accuratezza_funzionale = evaluate_functional_accuracy(finale)
    print(f"Accuratezza funzionale: {accuratezza_funzionale}")

