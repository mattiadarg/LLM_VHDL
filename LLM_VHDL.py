import json
import logging
import os
import re
import random
import subprocess
import tempfile
from itertools import product
from typing import Dict, List, Tuple, Optional

import optuna
from matplotlib import pyplot as plt
from nltk.corpus import wordnet
import math
from torch import nn
from torch.nn.init import xavier_uniform_
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import pickle
from optuna import Trial
from sklearn.model_selection import KFold
import torch
import numpy as np
import torch.nn.functional as F
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import certifi
import requests

# Ignoro la verifica del certificato SSL per scopi di debug, altrimenti dà errore e non riesce a richiamare
# VHDLFormatter
try:
    response = requests.get("https://googlechromelabs.github.io", verify=False)
    print("Connessione riuscita")
except requests.exceptions.SSLError as e:
    print("Errore SSL:", e)

print(certifi.where())

options = webdriver.ChromeOptions()
options.add_argument("--ignore-certificate-errors")


def set_seed(seed=42):
    """
    Imposta un seed fisso per garantire la riproducibilità dei risultati.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Imposta il seed all'inizio dello script
set_seed(42)

"""
# Scarica le risorse necessarie
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
"""
# Define the device
device = torch.device("cuda")
print(device)


# Classe per la tokenizzazione del codice VHDL
class VHDLTokenizer:
    """
    Implementa un tokenizzatore personalizzato per il codice VHDL.
    Questo tokenizzatore utilizza regex per suddividere il codice in token significativi.
    """

    def __init__(self):
        # Definizione dei pattern per i token VHDL
        self.patterns = [
            ('COMMENT', r'--.*?$'),  # Commenti
            ('STRING', r'".*?"'),  # Stringhe tra doppi apici
            ('CHARACTER', r"'.*?'"),  # Caratteri singoli tra apici
            ('NUMBER', r'\b\d+(\.\d+)?\b'),  # Numeri interi e decimali
            ('KEYWORD',
             r'\b(?:architecture|begin|end|entity|is|signal|process|if|elsif|else|then|library|use|port|in|out|std_logic|std_logic_vector|downto|to|of|type|array|constant|others)\b'),
            ('OPERATOR', r':=|<=|=>|<>|[<>]=?|[-+*/=&|]'),  # Operatori
            ('PUNCTUATION', r'[,;:\(\)\[\]]'),  # Punteggiatura
            ('IDENTIFIER', r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'),  # Identificatori
            ('WHITESPACE', r'\s+'),  # Spazi bianchi
            ('NEWLINE', r'\n'),  # Nuove righe
            ('UNKNOWN', r'.'),  # Token sconosciuti
        ]
        self.regex = re.compile('|'.join('(?P<%s>%s)' % pair for pair in self.patterns), re.MULTILINE | re.IGNORECASE)

    def tokenize(self, code: str) -> List[Tuple[str, str]]:
        """
        Tokenizza il codice VHDL in base ai pattern definiti.

        :param code: stringa contenente il codice VHDL
        :return: lista di tuple (tipo, valore) dei token
        """
        tokens = []
        for match in self.regex.finditer(code):
            kind = match.lastgroup
            value = match.group()
            if kind == 'WHITESPACE' or kind == 'COMMENT' or kind == 'NEWLINE':
                continue
            elif kind == 'KEYWORD':
                tokens.append((value.upper(), value))
            elif kind == 'IDENTIFIER':
                tokens.append(('IDENTIFIER', value))
            elif kind == 'NUMBER':
                tokens.append(('NUMBER', value))
            elif kind == 'STRING':
                tokens.append(('STRING', value))
            elif kind == 'CHARACTER':
                tokens.append(('CHARACTER', value))
            elif kind == 'OPERATOR':
                tokens.append(('OPERATOR', value))
            elif kind == 'PUNCTUATION':
                tokens.append(('PUNCTUATION', value))
            else:
                tokens.append(('UNKNOWN', value))
        return tokens


# Classe per la gestione del vocabolario
class Vocabulary:
    """
    Implementa la gestione del vocabolario per il codice VHDL.
    Consente il mapping tra token e indici e viceversa.
    """
    BOS = ('SPECIAL', '<BOS>')
    EOS = ('SPECIAL', '<EOS>')
    PAD = ('SPECIAL', '<PAD>')

    def __init__(self):
        """
        Inizializza le strutture dati per il mapping token-indice.
        """
        self.token2index = {}
        self.index2token = {}
        self.special_tokens = [self.PAD, self.BOS, self.EOS]

        for idx, (token_type, token_value) in enumerate(self.special_tokens):
            self.token2index[(token_type, token_value)] = idx
            self.index2token[idx] = (token_type, token_value)

        self.offset = len(self.special_tokens)

    def add_tokens(self, tokens: List[str]) -> None:
        """
        Aggiunge nuovi token al vocabolario.

        :param tokens: lista di stringhe rappresentanti i token
        """
        for token in tokens:
            if token not in self.token2index:
                i = len(self.token2index)
                self.token2index[token] = i
                self.index2token[i] = token

    def tokenize(self, sentence: str, add_special_tokens: bool = True) -> List[str]:
        """
        Tokenizza una stringa utilizzando un tokenizer VHDL.

        :param sentence: stringa da tokenizzare
        :param add_special_tokens: se aggiungere o meno i token <BOS> e <EOS>
        :return: lista di token
        """
        # Usa il tokenizer VHDL
        vhdl_tokenizer = VHDLTokenizer()
        tokens = vhdl_tokenizer.tokenize(sentence)
        # Estrai solo i valori dei token (escludendo i tipi)
        if add_special_tokens:
            tokens = [self.BOS] + tokens + [self.EOS]
        return tokens

    def encode(self, sentence: str, add_special_tokens: bool = True) -> List[int]:
        """
        Converte una stringa in una lista di indici basati sul vocabolario.

        :param sentence: stringa da codificare
        :param add_special_tokens: se aggiungere i token <BOS> e <EOS>
        :return: lista di indici
        """
        tokens = self.tokenize(sentence, add_special_tokens)
        # print("Tokens:", tokens)
        # print("Token2Index keys:", list(self.token2index.keys()))
        return [self.token2index[token] for token in tokens if token in self.token2index]

    def batch_encode(self, sentences: List[str], padding: bool = True, add_special_tokens: bool = False) -> List[List[int]]:
        """
        Converte un elenco di stringhe in un elenco annidato di indici di token.

        :param sentences: Una lista di stringhe da codificare in un batch    .
        :param padding: Booleano che consente il padding fino alla sequenza più lunga del batch.
        :param add_special_tokens: Booleano che consente di aggiungere un token BOS e EOS a ogni frase del batch.
        :return: Elenco annidato di sequenze tokenizzate.
        """
        tokenized_sentences = [self.encode(sentence, add_special_tokens) for sentence in sentences]
        if padding:
            max_length = max(len(tokens) for tokens in tokenized_sentences)
            tokenized_sentences = [
                tokens + [self.token2index[(self.PAD, self.PAD)]] * (max_length - len(tokens))
                for tokens in tokenized_sentences
            ]
        return tokenized_sentences

    def decode(self, token_ids: List[int]) -> str:
        """
        Converte una lista di indici in una stringa leggibile.

        :param token_ids: lista di indici
        :return: stringa decodificata
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.index2token:
                token_type, token_value = self.index2token[token_id]
                tokens.append((token_type, token_value))
            else:
                tokens.append(('UNKNOWN', '<UNK>'))

        code_parts = []
        for i, (token_type, token_value) in enumerate(tokens):
            if token_type == 'PUNCTUATION':
                code_parts.append(token_value)
            elif token_type == 'OPERATOR':
                code_parts.append(' ' + token_value + ' ')
            else:
                if i > 0 and code_parts[-1] not in [' ', '\n']:
                    code_parts.append(' ')
                code_parts.append(token_value)

        code = ''.join(code_parts)
        return code.strip()

    def __len__(self):
        return len(self.token2index)


def build_vocabulary(self, tokenized_texts: List[List[Tuple[str, str]]]):
    """
    Costruisce il vocabolario basato sui testi tokenizzati.

    :param tokenized_texts: Lista di liste di tuple (tipo, valore) dei token.
    """
    unique_tokens = set()
    for tokens in tokenized_texts:
        unique_tokens.update(tokens)

    for idx, token in enumerate(sorted(unique_tokens), start=self.offset):
        self.token2index[token] = idx
        self.index2token[idx] = token


def create_vocabulary(train_descriptions, train_vhdl, val_descriptions, val_vhdl, test_descriptions, test_vhdl):
    """
    Crea un oggetto vocabolario utilizzando i token provenienti dai dataset di training, validazione e test.

    :param train_descriptions: Liste di descrizioni del dataset di training.
    :param train_vhdl: Liste di codici VHDL del dataset di training.
    :param val_descriptions: Liste di descrizioni del dataset di validazione.
    :param val_vhdl: Liste di codici VHDL del dataset di validazione.
    :param test_descriptions: Liste di descrizioni del dataset di test.
    :param test_vhdl: Liste di codici VHDL del dataset di test.
    :return: Oggetto `Vocabulary` con i token unici di tutti i dataset.
    """
    all_words = set()

    # Inizializza il Vocabolario
    vocab = Vocabulary()

    # Aggiungi token dalle descrizioni e codici VHDL del training set
    for description in train_descriptions:
        words = vocab.tokenize(description, add_special_tokens=False)
        all_words.update(words)

    for vhdl in train_vhdl:
        words = vocab.tokenize(vhdl, add_special_tokens=False)
        all_words.update(words)

    # Aggiungi token dalle descrizioni e codici VHDL del set di validazione
    for description in val_descriptions:
        words = vocab.tokenize(description, add_special_tokens=False)
        all_words.update(words)

    for vhdl in val_vhdl:
        words = vocab.tokenize(vhdl, add_special_tokens=False)
        all_words.update(words)

    # Aggiungi token dalle descrizioni e codici VHDL del test set
    for description in test_descriptions:
        words = vocab.tokenize(description, add_special_tokens=False)
        all_words.update(words)

    for vhdl in test_vhdl:
        words = vocab.tokenize(vhdl, add_special_tokens=False)
        all_words.update(words)

    # Aggiungi tutti i token unici al vocabolario
    vocab.add_tokens(list(all_words))

    return vocab


"""file utils.py"""


def construct_future_mask(seq_len: int):
    """
    Costruisce una maschera binaria che blocca le connessioni future nella sequenza di input.

    :param seq_len: Lunghezza della sequenza di input.
    :return: Maschera binaria di forma (seq_len, seq_len).
    """
    subsequent_mask = torch.triu(torch.full((seq_len, seq_len), 1), diagonal=1)
    return subsequent_mask == 0


def construct_batches(
        corpus: List[Dict[str, str]],
        vocab: Vocabulary,
        batch_size: int,
        src_lang_key: str,
        tgt_lang_key: str,
        device: Optional[torch.device] = None,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
    """
    Costruisce batch di dati dal corpus dato.

    :param corpus: Corpus di dati come lista di dizionari con sorgente e target.
    :param vocab: Oggetto `Vocabulary` per codificare le sequenze.
    :param batch_size: Numero di sequenze per batch.
    :param src_lang_key: Chiave della lingua sorgente nel dizionario.
    :param tgt_lang_key: Chiave della lingua target nel dizionario.
    :param device: Dispositivo su cui caricare i tensori (es. GPU o CPU).
    :return: Due dizionari contenenti batch e maschere di attenzione.
    """
    pad_token_id = vocab.token2index[vocab.PAD]
    batches: Dict[str, List] = {"src": [], "tgt": []}
    masks: Dict[str, List] = {"src": [], "tgt": []}

    for i in range(0, len(corpus), batch_size):
        src_batch = torch.IntTensor(
            vocab.batch_encode(
                [pair[src_lang_key] for pair in corpus[i: i + batch_size]],
                add_special_tokens=True,
                padding=True,
            )
        )

        tgt_batch = torch.IntTensor(
            vocab.batch_encode(
                [pair[tgt_lang_key] for pair in corpus[i: i + batch_size]],
                add_special_tokens=True,
                padding=True,
            )
        )

        src_padding_mask = src_batch != pad_token_id
        future_mask = construct_future_mask(tgt_batch.shape[-1])

        if device is not None:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            src_padding_mask = src_padding_mask.to(device)
            future_mask = future_mask.to(device)
        batches["src"].append(src_batch)
        batches["tgt"].append(tgt_batch)
        masks["src"].append(src_padding_mask)
        masks["tgt"].append(future_mask)
    return batches, masks


"""Positional Encodings"""


class SinusoidEncoding(torch.nn.Module):
    """
   Implementazione delle codifiche posizionali sinusoidali per i modelli Transformer.
   """

    def __init__(self, hidden_dim, max_len=5000):
        """
        Inizializza le codifiche posizionali.

        :param hidden_dim: Dimensionalità dello spazio latente.
        :param max_len: Lunghezza massima delle sequenze.
        """
        super().__init__()

        pos_embed = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        pos_embed = pos_embed.unsqueeze(0)

        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def forward(self, x):
        """
        Aggiunge le codifiche posizionali agli embedding di input.

        :param x: Embedding di input. Forma: (batch_size, sequence_length, embedding_dim).
        :return: Embedding con codifiche posizionali aggiunte.
        """
        x = x + self.pos_embed[:, : x.size(1)]
        return x


"""Multi-Head Attention"""


class MultiHeadAttention(nn.Module):
    """
    Implementazione del meccanismo Multi-Head Attention utilizzato nei Transformer.
    Permette al modello di focalizzarsi su parti diverse della sequenza contemporaneamente.
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()

        assert hidden_dim % num_heads == 0
        self.qkv_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * num_heads * self.qkv_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.qkv_dim, hidden_dim, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Inizializza i pesi con la distribuzione di Xavier.
        """
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(
            self,
            x: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            src_padding_mask: Optional[torch.BoolTensor] = None,
            future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Calcola l'output del meccanismo Multi-Head Attention.

        :param x: Tensori di input. Forma: (batch_size, seq_len, hidden_dim).
        :param encoder_hidden_states: Stati nascosti dell'encoder per cross-attention. (opzionale)
        :param src_padding_mask: Maschera per ignorare i token di padding. Forma: (batch_size, seq_len).
        :param future_mask: Maschera per bloccare l'attenzione ai token futuri. Forma: (seq_len, seq_len).
        :return: Output trasformato. Forma: (batch_size, seq_len, hidden_dim).
        """
        batch_size, sequence_length, hidden_dim = x.size()

        if encoder_hidden_states is None:
            q, k, v = self._self_attention_projection(x)
        else:
            q, k, v = self._cross_attention_projection(encoder_hidden_states, x)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        values, attn = self.scaled_dot_product(q, k, v, src_padding_mask, future_mask)

        if values.dim() == 4:
            batch_size, num_heads, seq_length, qkv_dim = values.shape
        else:
            raise ValueError(f"Expected 4 dimensions in `values`, but got {values.dim()} dimensions.")

        # Calcola i valori di attenzione e i vettori contestualizzati
        values, attn = self.scaled_dot_product(q, k, v, src_padding_mask, future_mask)

        # Concatena i vettori delle testine e proietta l'output alla dimensione originale
        values = values.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.hidden_dim)
        output = self.o_proj(values)
        return output

    def _self_attention_projection(self, x: torch.Tensor):
        """
        Calcola Q, K, V per self-attention.

        :param x: Input tensor. Forma: (batch_size, seq_len, hidden_dim).
        :return: Tuple (Q, K, V) di forma: (batch_size, seq_len, num_heads, qkv_dim).
        """
        batch_size, sequence_length, _ = x.shape
        # Proietta Q, K, V in un unico tensor
        qkv = self.qkv_proj(x)
        # Ridimensiona e separa Q, K, V
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.qkv_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v

    def _cross_attention_projection(
            self, encoder_hidden_states: torch.Tensor, decoder_hidden_states: torch.Tensor,
    ):
        """
        Calcola Q, K, V per cross-attention.

        :param encoder_hidden_states: Stati nascosti dell'encoder. Forma: (batch_size, src_seq_len, hidden_dim).
        :param decoder_hidden_states: Stati nascosti del decoder. Forma: (batch_size, tgt_seq_len, hidden_dim).
        :return: Tuple (Q, K, V) di forma: (batch_size, seq_len, num_heads, qkv_dim).
        """
        batch_size, src_sequence_length, hidden_dim = encoder_hidden_states.shape
        batch_size, tgt_sequence_length, hidden_dim = decoder_hidden_states.shape

        w_q, w_kv = self.qkv_proj.weight.split([hidden_dim, 2 * hidden_dim])

        k, v = (
            F.linear(input=encoder_hidden_states, weight=w_kv)
            .reshape(batch_size, src_sequence_length, self.num_heads, 2 * self.qkv_dim)
            .chunk(2, dim=-1)
        )

        q = F.linear(input=decoder_hidden_states, weight=w_q).reshape(
            batch_size, tgt_sequence_length, self.num_heads, self.qkv_dim
        )

        return q, k, v

    def scaled_dot_product(self, q, k, v, src_padding_mask=None, future_mask=None):
        """
        Calcola il prodotto scalare normalizzato e applica le maschere di attenzione.

        :param q: Tensor di query. Forma: (batch_size, num_heads, seq_len, qkv_dim).
        :param k: Tensor di key. Forma: (batch_size, num_heads, seq_len, qkv_dim).
        :param v: Tensor di value. Forma: (batch_size, num_heads, seq_len, qkv_dim).
        :param src_padding_mask: Maschera per i token di padding. (opzionale)
        :param future_mask: Maschera per bloccare i token futuri. (opzionale)
        :return: Valori contestualizzati e pesi di attenzione.
        """
        # Calcola i logits di attenzione come prodotto scalare tra Q e K trasposto
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(self.qkv_dim)

        # Applica la maschera per i token di padding
        if src_padding_mask is not None:
            if src_padding_mask.dim() == 2:
                src_padding_mask = src_padding_mask.unsqueeze(1).unsqueeze(2)
            elif src_padding_mask.dim() == 3:
                src_padding_mask = src_padding_mask.unsqueeze(1)
            attn_logits = attn_logits.masked_fill(src_padding_mask == 0, float('-inf'))

        # Applica la maschera per bloccare i token futuri
        if future_mask is not None:
            # Assicuriamoci che la maschera future_mask sia di 3 dimensioni [T, T]
            if future_mask.dim() == 2:
                future_mask = future_mask.unsqueeze(0).unsqueeze(0)
            elif future_mask.dim() == 3:
                future_mask = future_mask.unsqueeze(1)

            # Espansione della maschera future_mask
            future_mask = future_mask.expand(q.size(0), q.size(1), future_mask.size(-2), future_mask.size(-1))

            # Adattamento della maschera
            future_mask = future_mask.contiguous()
            attn_logits = attn_logits.masked_fill(future_mask == 0, float('-inf'))

        # Applica softmax per calcolare i pesi di attenzione
        attn = torch.softmax(attn_logits, dim=-1)
        values = torch.matmul(attn, v)
        return values, attn

    @staticmethod
    def mask_logits(
            logits: torch.Tensor,
            src_padding_mask: Optional[torch.BoolTensor] = None,
            future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Applica maschere ai logits per ignorare alcune connessioni di attenzione.

        :param logits: Tensor contenente i logits di attenzione. Forma: (N, H, S o T, S o T).
        :param src_padding_mask: Maschera per i token di padding nei dati sorgente. Ignora i token di padding. Forma: (N, S).
        :param future_mask: Maschera per bloccare l'accesso ai token futuri durante il training del decoder. Forma: (T, T).
        :return: Tensor di logits mascherati. Forma: (N, H, S o T, S o T).
        """
        if src_padding_mask is not None:
            logits = logits.masked_fill(src_padding_mask[:, None, None, :] == 0, float("-inf"))
        if future_mask is not None:
            logits = logits.masked_fill(future_mask == 0, float("-inf"))
        return logits


"""Decoder Definition"""


class TransformerDecoder(nn.Module):
    """
    Implementa il decoder del Transformer, che elabora gli input di destinazione
    e combina l'output dell'encoder per generare sequenze tradotte.
    """
    def __init__(
            self,
            embedding: torch.nn.Embedding,
            hidden_dim: int,
            ff_dim: int,
            num_heads: int,
            num_layers: int,
            vocab_size: int,
            dropout_p: float,
            tie_output_to_embedding: Optional[bool] = True,
    ):
        """
        Inizializza il decoder con layer di attenzione e feed-forward.

        :param embedding: Embedding layer per rappresentare i token.
        :param hidden_dim: Dimensione dello spazio latente.
        :param ff_dim: Dimensione del feed-forward network.
        :param num_heads: Numero di testine di attenzione.
        :param num_layers: Numero di blocchi del decoder.
        :param vocab_size: Dimensione del vocabolario.
        :param dropout_p: Tasso di dropout per regolarizzazione.
        :param tie_output_to_embedding: Se True, condivide i pesi tra layer di output e embedding.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embed = embedding
        self.positional_encoding = SinusoidEncoding(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(hidden_dim, ff_dim, num_heads, dropout_p)
                for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Note: a linear layer multiplies the input with a transpose of the weight matrix, so no need to do that here.
        if tie_output_to_embedding:
            self.output_layer.weight = nn.Parameter(self.embed.weight)

    def _reset_parameters(self):
        """ Perform xavier weight initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(
            self,
            input_tokens: torch.IntTensor,
            encoder_hidden_states: torch.Tensor,
            src_padding_mask: Optional[torch.BoolTensor] = None,
            future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Esegue un passaggio in avanti nel decoder, dato l'output dell'encoder e gli input di destinazione.

        :param input_tokens: Token di input del decoder. Forma: (N, T).
        :param encoder_hidden_states: Stati nascosti finali dell'encoder. Forma: (N, S, E).
        :param src_padding_mask: Maschera per ignorare i token di padding. Forma: (N, S).
        :param future_mask: Maschera per bloccare i token futuri. Forma: (T, T).
        :return: Logits non normalizzati del vocabolario. Forma: (N, T, V).
        """

        x = self.embed(input_tokens) * math.sqrt(self.hidden_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_hidden_states, src_padding_mask, future_mask)

        logits = self.output_layer(x)
        return logits


class TransformerDecoderBlock(nn.Module):
    """
    Un singolo blocco del decoder Transformer, composto da:
    - Self-attention
    - Cross-attention
    - Feed-forward network
    """
    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int, dropout_p: float):
        super().__init__()

        self.cross_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.self_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, hidden_dim),
        )

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.dropout3 = nn.Dropout(p=dropout_p)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)

    def forward(
            self,
            x: torch.Tensor,
            encoder_hidden_states: torch.FloatTensor,
            src_padding_mask: Optional[torch.BoolTensor] = None,
            future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Esegue un passaggio in avanti in un blocco del decoder.

        :param x: Output del blocco precedente del decoder. Forma: (N, T, E).
        :param encoder_hidden_states: Stati nascosti finali dell'encoder. Forma: (N, S, E).
        :param src_padding_mask: Maschera per ignorare i token di padding. Forma: (N, S).
        :param future_mask: Maschera per bloccare i token futuri. Forma: (T, T).
        :return: Embedding contestualizzato aggiornato. Forma: (N, T, E).
        """

        # Self attention
        output = self.dropout1(self.self_mha.forward(x, future_mask=future_mask))
        x = self.layer_norm1(x + output)

        # Cross attention
        output = self.dropout2(
            self.cross_mha.forward(
                x,
                encoder_hidden_states=encoder_hidden_states,
                src_padding_mask=src_padding_mask,
            )
        )
        x = self.layer_norm2(x + output)

        # Feed forward layers
        output = self.dropout3(self.feed_forward(x))
        x = self.layer_norm3(x + output)
        return x


"""Encoder Definition"""


class TransformerEncoder(nn.Module):
    """
    Implementa l'encoder del Transformer, che elabora le sequenze di input e genera
    rappresentazioni contestualizzate da passare al decoder.
    """
    def __init__(
            self,
            embedding: torch.nn.Embedding,
            hidden_dim: int,
            ff_dim: int,
            num_heads: int,
            num_layers: int,
            dropout_p: float,
    ):
        super().__init__()
        self.embed = embedding
        self.hidden_dim = hidden_dim
        self.positional_encoding = SinusoidEncoding(hidden_dim, max_len=5000)
        self.dropout = nn.Dropout(p=dropout_p)
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(hidden_dim, ff_dim, num_heads, dropout_p)
                for _ in range(num_layers)
            ]
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(
            self, input_ids: torch.Tensor, src_padding_mask: Optional[torch.BoolTensor] = None
    ):
        """
        Esegue un passaggio in avanti nell'encoder.

        :param input_ids: ID dei token di input. Forma: (N, S).
        :param src_padding_mask: Maschera per ignorare i token di padding. Forma: (N, S).
        :return: Embedding contestualizzato finale. Forma: (N, S, E).
        """
        x = self.embed(input_ids) * math.sqrt(self.hidden_dim)  # (N, S, E)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block.forward(x, src_padding_mask=src_padding_mask)
        return x


class EncoderBlock(nn.Module):
    """
    Un singolo blocco dell'encoder Transformer, composto da:
    - Self-attention
    - Feed-forward network
    """
    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int, dropout_p: float):
        super().__init__()
        self.self_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, hidden_dim),
        )

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.FloatTensor, src_padding_mask: Optional[torch.BoolTensor] = None):
        """
        Esegue un passaggio in avanti in un blocco dell'encoder.

        :param x: Embedding in input o output del blocco precedente. Forma: (N, S, E).
        :param src_padding_mask: Maschera per ignorare i token di padding. Forma: (N, S).
        :return: Embedding contestualizzato intermedio. Forma: (N, S, E).
        """
        output = self.dropout1(
            self.self_mha.forward(x, src_padding_mask=src_padding_mask)
        )
        x = self.layer_norm1(x + output)

        output = self.dropout2(self.feed_forward(x))
        x = self.layer_norm2(x + output)
        return x


"""Learning Rate Scheduler"""


class NoamOpt:
    """
    Algoritmo di scheduling del learning rate per migliorare la stabilità del training
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        """
        Inizializza l'oggetto `NoamOpt`.

        :param model_size: La dimensione del modello (ad es. la dimensione del layer nascosto).
        :param factor: Fattore di scalatura per il learning rate.
        :param warmup: Numero di step iniziali durante i quali il learning rate aumenta linearmente.
        :param optimizer: L'ottimizzatore da usare (es. Adam).
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """
        Aggiorna i parametri del modello e il learning rate.
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """
        Calcola il learning rate in base allo step corrente.

        :param step: Step corrente. Se non specificato, usa `_step`.
        :return: Il learning rate calcolato.
        """
        if step is None:
            step = self._step
        return self.factor * (
                self.model_size ** (-0.5)
                * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )


def get_std_opt(model):
    """
    Crea un'istanza di `NoamOpt` con parametri standard per un Transformer.

    :param model: Il modello Transformer.
    :return: Un'istanza di `NoamOpt`.
    """
    return NoamOpt(
        model.encoder.hidden_dim,
        2,
        4000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
    )


"""Transformer Model"""


class Transformer(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            ff_dim: int,
            num_heads: int,
            num_layers: int,
            max_decoding_length: int,
            vocab_size: int,
            padding_idx: int,
            bos_idx: int,
            dropout_p: float,
            tie_output_to_embedding: Optional[bool] = None,
    ):
        """
        Inizializza il modello Transformer.

        :param hidden_dim: Dimensione dello spazio latente.
        :param ff_dim: Dimensione del feed-forward network.
        :param num_heads: Numero di testine di attenzione.
        :param num_layers: Numero di layer nell'encoder e nel decoder.
        :param max_decoding_length: Lunghezza massima della sequenza di output.
        :param vocab_size: Dimensione del vocabolario.
        :param padding_idx: Indice del token di padding.
        :param bos_idx: Indice del token di inizio sequenza.
        :param dropout_p: Tasso di dropout.
        :param tie_output_to_embedding: Se True, condivide i pesi tra l'output e l'embedding del decoder.
        """
        super().__init__()
        # Because the encoder embedding, and decoder embedding and decoder pre-softmax transformation share embeddings
        # weights, initialize one here and pass it on.
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        self.encoder = TransformerEncoder(
            self.embed, hidden_dim, ff_dim, num_heads, num_layers, dropout_p
        )
        self.decoder = TransformerDecoder(
            self.embed,
            hidden_dim,
            ff_dim,
            num_heads,
            num_layers,
            vocab_size,
            dropout_p,
            tie_output_to_embedding,
        )

        self.padding_idx = padding_idx
        self.bos_idx = bos_idx
        self.max_decoding_length = max_decoding_length
        self.hidden_dim = hidden_dim
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Esegue un passaggio in avanti nel modello Transformer.

        :param src: Input sorgente. Forma: (N, S).
        :param tgt: Input target. Forma: (N, T).
        :param src_mask: Maschera per ignorare i token di padding nella sorgente.
        :param tgt_mask: Maschera per impedire l'accesso ai token futuri nel target.
        :return: Output del decoder. Forma: (N, T, V).
        """
        enc_out = self.encoder(src, src_padding_mask=src_mask)
        dec_out = self.decoder(tgt, enc_out, src_padding_mask=src_mask, future_mask=tgt_mask)
        return dec_out


"""Training Script"""


def load_and_preprocess_data(dataset_path, subset_size: Optional[int] = None):
    """
    Carica e pre-elabora il dataset.

    :param dataset_path: Percorso al dataset CSV.
    :param subset_size: Se specificato, limita il numero di righe caricate.
    :return: Dati divisi in train, validation e test (descrizioni e contenuti VHDL).
    """
    # Usare la virgola come delimitatore
    data = pd.read_csv(dataset_path, sep=',', skip_blank_lines=True, on_bad_lines='skip')

    if subset_size is not None:
        data = data.head(subset_size)

    print(f"Numero di righe e colonne: {data.shape}")

    print(f"Numero totale di righe: {data.shape[0]}")

    # Pulizia dei dati
    data['descrizione'] = data['descrizione'].str.strip()
    data['content'] = data['content'].str.strip()

    # Shuffle dei dati prima di split
    train_data, temp_data = train_test_split(data.sample(frac=1).reset_index(drop=True), test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Aggiorna le liste
    train_descriptions = train_data['descrizione'].tolist()
    train_vhdl = train_data['content'].tolist()

    val_descriptions = val_data['descrizione'].tolist()
    val_vhdl = val_data['content'].tolist()

    test_descriptions = test_data['descrizione'].tolist()
    test_vhdl = test_data['content'].tolist()

    return train_descriptions, train_vhdl, val_descriptions, val_vhdl, test_descriptions, test_vhdl


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, ignore_index=-100):
        """
        Inizializza la loss con smoothing delle etichette.

        :param classes: Numero di classi nel vocabolario.
        :param smoothing: Livello di smoothing (default 0.1).
        :param ignore_index: Indice da ignorare nei calcoli (default -100).
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Calcola la loss con smoothing delle etichette.

        :param pred: Predizioni del modello. Forma: (N, V).
        :param target: Etichette target. Forma: (N,).
        :return: Valore della loss.
        """
        pred = pred.log_softmax(dim=-1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        mask = (target != self.ignore_index).unsqueeze(1)
        loss = torch.sum(-true_dist * pred * mask) / mask.sum()
        return loss


class VHDLDataset(Dataset):
    def __init__(self, descriptions, vhdl_codes, vocab):
        """
        Inizializza il dataset con descrizioni e codici VHDL.

        :param descriptions: Lista di descrizioni testuali.
        :param vhdl_codes: Lista di codici VHDL corrispondenti.
        :param vocab: Vocabolario per codificare le sequenze in token numerici.
        """
        self.descriptions = descriptions
        self.vhdl_codes = vhdl_codes
        self.vocab = vocab

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        """
        Restituisce una coppia (descrizione, codice VHDL) codificata in token numerici.

        :param idx: Indice del campione.
        :return: Tensori PyTorch contenenti la descrizione e il codice VHDL.
        """
        description = self.vocab.encode(self.descriptions[idx])
        vhdl_code = self.vocab.encode(self.vhdl_codes[idx])
        return torch.tensor(description), torch.tensor(vhdl_code)


# Funzione di padding per uniformare la lunghezza delle sequenze
def pad_sequence(seq, max_len, pad_value=0):
    """
    Applica padding alla sequenza per uniformarla a una lunghezza fissa.

    :param seq: Sequenza di token.
    :param max_len: Lunghezza massima desiderata.
    :param pad_value: Valore usato per il padding (default: 0).
    :return: Sequenza con padding.
    """
    return seq + [pad_value] * (max_len - len(seq))


def collate_fn(batch):
    """
    Funzione per il DataLoader: applica padding a ogni batch.

    :param batch: Lista di coppie (descrizione, codice VHDL).
    :return: Tensori PyTorch con padding per descrizioni e codici VHDL.
    """
    descriptions, vhdl_codes = zip(*batch)

    max_len_desc = max(len(d) for d in descriptions)
    max_len_vhdl = max(len(v) for v in vhdl_codes)

    descriptions_padded = [pad_sequence(d.tolist(), max_len_desc) for d in descriptions]
    vhdl_codes_padded = [pad_sequence(v.tolist(), max_len_vhdl) for v in vhdl_codes]

    return torch.tensor(descriptions_padded), torch.tensor(vhdl_codes_padded)


def create_masks(src, tgt, pad_idx):
    """
    Crea le maschere per l'attenzione durante il training.

    :param src: Tensore sorgente (descrizioni).
    :param tgt: Tensore target (codici VHDL).
    :param pad_idx: Indice del token di padding.
    :return: Maschera per il sorgente e maschera per il target.
    """
    src_mask = (src != pad_idx).unsqueeze(-2).to(device)
    tgt_mask = (tgt != pad_idx).unsqueeze(-2).to(device)
    tgt_mask = tgt_mask & (
            1 - torch.triu(torch.ones((1, tgt.size(-1), tgt.size(-1)), device=tgt.device), diagonal=1)).bool()
    return src_mask, tgt_mask


# Dataset e DataLoader con padding
class PaddedVHDLDataset(Dataset):
    def __init__(self, descriptions, vhdl_codes, vocab, max_len_description, max_len_vhdl):
        """
        Inizializza il dataset con padding pre-calcolato.

        :param descriptions: Lista di descrizioni testuali.
        :param vhdl_codes: Lista di codici VHDL corrispondenti.
        :param vocab: Vocabolario per codificare le sequenze.
        :param max_len_description: Lunghezza massima delle descrizioni.
        :param max_len_vhdl: Lunghezza massima dei codici VHDL.
        """
        self.descriptions = descriptions
        self.vhdl_codes = vhdl_codes
        self.vocab = vocab
        self.max_len_description = max_len_description
        self.max_len_vhdl = max_len_vhdl

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        """
        Restituisce una coppia (descrizione, codice VHDL) con padding applicato.

        :param idx: Indice del campione.
        :return: Tensori PyTorch con padding.
        """
        description = self.vocab.encode(self.descriptions[idx])
        vhdl_code = self.vocab.encode(self.vhdl_codes[idx])
        description = pad_sequence(description, self.max_len_description)
        vhdl_code = pad_sequence(vhdl_code, self.max_len_vhdl)
        return torch.tensor(description), torch.tensor(vhdl_code)


def train_and_save_model(model, train_loader, val_loader, criterion, optimizer, padding_idx, epochs=1000, save_path="llm_vhdl_model.pth", patience=20):
    """
    Addestra il modello e salva i pesi con la migliore validazione.

    :param model: Modello Transformer da addestrare.
    :param train_loader: DataLoader per il training set.
    :param val_loader: DataLoader per il validation set.
    :param criterion: Funzione di perdita.
    :param optimizer: Ottimizzatore.
    :param padding_idx: Indice del padding nel vocabolario.
    :param epochs: Numero massimo di epoche.
    :param save_path: Percorso per salvare il modello.
    :param patience: Numero massimo di epoche senza miglioramenti per l'early stopping.
    :return: Liste delle perdite di training e validazione.
    """
    model = model.to(device)
    best_loss = float('inf')
    epochs_without_improvement = 0

    # Liste per salvare le perdite
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for descriptions, vhdl_codes in train_loader:
            descriptions = descriptions.to(device)
            vhdl_codes = vhdl_codes.to(device)

            optimizer.zero_grad()
            src_mask, tgt_mask = create_masks(descriptions, vhdl_codes[:, :-1], padding_idx)
            outputs = model(descriptions, vhdl_codes[:, :-1], src_mask, tgt_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), vhdl_codes[:, 1:].contiguous().view(-1))

            loss.backward()
            # Applica il gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss = evaluate(model, val_loader, criterion, padding_idx)
        val_losses.append(val_loss)

        print(f'Epoca {epoch + 1}/{epochs}, Perdita di Training: {train_loss}, Perdita di Validazione: {val_loss}')

        # Aggiorna lo scheduler con la perdita di validazione
        # scheduler.step(val_loss)

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"Modello salvato all'epoca {epoch + 1} con perdita di validazione: {val_loss}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping all'epoca {epoch + 1}. Nessun miglioramento per {patience} epoche consecutive.")
            break

    # Visualizza le curve di apprendimento
    # plot_learning_curves(train_losses, val_losses)
    return train_losses, val_losses


# Funzione di valutazione
def evaluate(model, test_loader, criterion, padding_idx):
    """
    Valuta il modello sul test set.

    :param model: Modello Transformer.
    :param test_loader: DataLoader per il test set.
    :param criterion: Funzione di perdita.
    :param padding_idx: Indice del padding nel vocabolario.
    :return: Perdita media sul test set.
    """
    model.eval()  # Imposta il modello in modalità di valutazione
    total_loss = 0
    with torch.no_grad():  # Disabilita il calcolo dei gradienti
        for descriptions, vhdl_codes in test_loader:
            descriptions = descriptions.to(device)
            vhdl_codes = vhdl_codes.to(device)
            src_mask, tgt_mask = create_masks(descriptions, vhdl_codes[:, :-1], padding_idx)
            outputs = model(descriptions, vhdl_codes[:, :-1], src_mask, tgt_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), vhdl_codes[:, 1:].contiguous().view(-1))
            total_loss += loss.item()
    # model.train()  # Ripristina il modello in modalità di addestramento
    return total_loss / len(test_loader)


def check_vhdl_syntax(vhdl_code: str) -> bool:
    """
    Verifica la sintassi di un codice VHDL utilizzando il compilatore GHDL.

    :param vhdl_code: Stringa contenente il codice VHDL.
    :return: True se il codice è sintatticamente corretto, False altrimenti.
    """
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


def generate_vhdl_code(model, input_text, vocab, max_length=512):
    """
    Genera codice VHDL a partire da una descrizione testuale. Utilizza una strategia greedy:
    per ogni passo, seleziona il token con la probabilità più alta (argmax) come prossimo token.

    :param model: Modello Transformer.
    :param input_text: Testo di input (descrizione).
    :param vocab: Vocabolario per codificare e decodificare.
    :param max_length: Lunghezza massima della sequenza generata.
    :return: Codice VHDL generato.
    """
    model.eval()

    # Determina il dispositivo usato dai parametri del modello
    device = next(model.parameters()).device

    # Tokenizzazione della descrizione
    input_tokens = vocab.encode(input_text)
    input_tensor = torch.tensor([input_tokens], device=device)

    # Creazione della maschera
    src_mask = (input_tensor != vocab.token2index[Vocabulary.PAD]).unsqueeze(-2)

    # Inizializzazione del decodificatore
    output_tokens = [vocab.token2index[Vocabulary.BOS]]

    for i in range(max_length):
        output_tensor = torch.tensor([output_tokens], device=device)
        tgt_mask = (output_tensor != vocab.token2index[Vocabulary.PAD]).unsqueeze(-2)
        tgt_mask = tgt_mask & (
                1 - torch.triu(torch.ones((1, output_tensor.size(-1), output_tensor.size(-1)), device=device),
                               diagonal=1)).bool()

        # Generazione dei token successivi
        output = model(input_tensor, output_tensor, src_mask, tgt_mask)

        vocab_size = len(vocab.token2index)
        next_token_logits = output[:, -1, :]

        # Applica softmax per ottenere le probabilità
        probabilities = F.softmax(next_token_logits, dim=-1)

        # Assicurati che le dimensioni delle probabilità corrispondano al vocabolario
        probabilities = probabilities[:, :vocab_size]

        next_token = output.argmax(dim=-1)[:, -1].item()

        # Aggiungi il token generato all'output
        output_tokens.append(next_token)

        # Stop se viene generato il token di fine sequenza (EOS)
        if next_token == vocab.token2index[Vocabulary.EOS]:
            break

    print("Raw Output Tokens:", output_tokens)

    invalid_token_ids = [tid for tid in output_tokens if tid not in vocab.index2token]
    if invalid_token_ids:
        print("Token IDs non validi trovati:", invalid_token_ids)

    vhdl_code = vocab.decode(output_tokens)
    print("Token corrispondenti:", [vocab.index2token.get(t, "<UNK>") for t in output_tokens])

    """if check_vhdl_syntax(vhdl_code):
        print("Il codice generato è sintatticamente corretto.")
    else:
        print("Il codice generato contiene errori sintattici.")"""
    return vhdl_code


def generate_vhdl_code_beam_search(model, input_text, vocab, max_length=512, beam_width=5):
    """
    Genera codice VHDL utilizzando la ricerca a fascio. Utilizza una strategia di beam search:
    esplora i beam_width percorsi migliori (sequenze parziali) in ogni passo, mantenendo un equilibrio tra esplorazione e sfruttamento.

    :param model: Modello Transformer.
    :param input_text: Testo di input (descrizione).
    :param vocab: Vocabolario per codificare e decodificare.
    :param max_length: Lunghezza massima della sequenza generata.
    :param beam_width: Numero di sequenze mantenute in ogni iterazione.
    :return: Codice VHDL generato.
    """
    model.eval()
    device = next(model.parameters()).device

    # Tokenizzazione della descrizione
    input_tokens = vocab.encode(input_text)
    input_tensor = torch.tensor([input_tokens], device=device)

    # Creazione della maschera
    src_mask = (input_tensor != vocab.token2index[Vocabulary.PAD]).unsqueeze(-2)

    # Inizializzazione del beam
    beams = [([vocab.token2index[Vocabulary.BOS]], 0.0)]  # (sequence, score)

    for _ in range(max_length):
        all_candidates = []
        for seq, score in beams:
            if seq[-1] == vocab.token2index[Vocabulary.EOS]:
                all_candidates.append((seq, score))
                continue
            output_tensor = torch.tensor([seq], device=device)
            tgt_mask = (output_tensor != vocab.token2index[Vocabulary.PAD]).unsqueeze(-2)
            tgt_mask = tgt_mask & ~torch.triu(
                torch.ones((1, output_tensor.size(-1), output_tensor.size(-1)), device=device), diagonal=1).bool()

            # Generazione dei token successivi
            output = model(input_tensor, output_tensor, src_mask, tgt_mask)
            next_token_logits = output[:, -1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            top_log_probs, top_indices = log_probs.topk(beam_width)

            for log_prob, idx in zip(top_log_probs[0], top_indices[0]):
                candidate_seq = seq + [idx.item()]
                candidate_score = score + log_prob.item()
                all_candidates.append((candidate_seq, candidate_score))

        # Ordina tutti i candidati per punteggio e seleziona i migliori k
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        beams = ordered[:beam_width]

        # Verifica se tutte le sequenze hanno raggiunto <EOS>
        if all(seq[-1] == vocab.token2index[Vocabulary.EOS] for seq, _ in beams):
            break

    # Seleziona la sequenza con il punteggio più alto
    best_seq, best_score = beams[0]
    print("Best Sequence Score:", best_score)
    vhdl_code = vocab.decode(best_seq)
    if vhdl_code is None:
        print("Errore: decodifica del codice non riuscita.")
        return None

    # Verifica la sintassi VHDL
    """if check_vhdl_syntax(vhdl_code):
        print("Il codice generato è sintatticamente corretto.")
    else:
        print("Il codice generato contiene errori sintattici.")"""
    return vhdl_code


def augment_data(descriptions: List[str], vhdl_codes: List[str], augmentation_factor: int = 1) -> Tuple[List[str], List[str]]:
    """
    Aumenta i dati nel dataset fornendo varianti delle descrizioni e dei codici VHDL.
    Effettua modifiche sulle descrizioni (utilizzando sinonimi) e sui codici VHDL (rinominando gli identificatori).

    :param descriptions: Lista di descrizioni (testo).
    :param vhdl_codes: Lista di codici VHDL associati.
    :param augmentation_factor: Numero di varianti da creare per ogni coppia di dati.
    :return: Tuple con la lista aumentata di descrizioni e codici VHDL.
    """
    augmented_descriptions = []
    augmented_vhdl_codes = []

    # Assicurati di aver scaricato le risorse di WordNet
    # nltk.download('wordnet')

    # Definisci una lista di parole chiave VHDL da non sostituire
    vhdl_keywords = {
        "entity", "architecture", "signal", "port", "begin", "end", "process",
        "if", "then", "else", "elsif", "case", "when", "is", "use", "library",
        "package", "function", "procedure", "type", "constant", "variable",
        "generic", "in", "out", "inout", "buffer", "std_logic",
        "std_logic_vector", "integer", "std_logic_1164", "numeric_std", "rising_edge"
    }

    # Funzione per ottenere sinonimi da WordNet
    def get_synonyms(word):
        synonyms = set()
        for syn in wordnet.synsets(word, lang='ita'):
            for lemma in syn.lemmas(lang='ita'):
                synonym = lemma.name().replace('_', ' ')
                if synonym != word:
                    synonyms.add(synonym)
        return list(synonyms)

    # Lista di identificatori nel codice VHDL da rinominare
    def extract_identifiers(vhdl_code):
        # Regex per identificare variabili, segnali e entità
        pattern = r'\b(?!\b' + r'\b|\b'.join(vhdl_keywords) + r'\b)\w+\b'
        identifiers = set(re.findall(pattern, vhdl_code))
        return identifiers

    for description, vhdl_code in zip(descriptions, vhdl_codes):
        # Numero di aumentazioni per ciascun esempio
        for _ in range(augmentation_factor):
            new_vhdl_code = vhdl_code
            new_description = description

            # Rinominazione coerente degli identificatori nel codice VHDL
            identifiers = extract_identifiers(new_vhdl_code)
            identifier_map = {}
            for identifier in identifiers:
                # Genera un nuovo nome di variabile casuale
                new_identifier = identifier + '_' + str(random.randint(1, 1000))
                identifier_map[identifier] = new_identifier

                # Sostituisci nel codice VHDL
                new_vhdl_code = re.sub(r'\b{}\b'.format(re.escape(identifier)), new_identifier, new_vhdl_code)

            # Sostituzione di sinonimi nella descrizione
            words = re.findall(r'\w+|\W+', new_description)
            new_words = []

            for word in words:
                # Controlla se la parola è alfanumerica (cioè non punteggiatura)
                if re.match(r'\w+', word):
                    lower_word = word.lower()
                    synonyms = get_synonyms(lower_word)
                    if synonyms:
                        # Scegli un sinonimo casualmente
                        synonym = random.choice(synonyms)
                        # Mantieni la capitalizzazione originale
                        if word[0].isupper():
                            synonym = synonym.capitalize()
                        new_words.append(synonym)
                    else:
                        new_words.append(word)
                else:
                    # Mantieni la punteggiatura
                    new_words.append(word)

            # Ricostruisci la descrizione
            new_description = ''.join(new_words)

            augmented_descriptions.append(new_description)
            augmented_vhdl_codes.append(new_vhdl_code)

    # Restituisci sia i dati originali che quelli aumentati
    return descriptions + augmented_descriptions, vhdl_codes + augmented_vhdl_codes


def define_search_space(trial: Trial):
    """
    Definisce lo spazio di ricerca per l'ottimizzazione degli iperparametri.

    :param trial: Istanza del trial Optuna per la ricerca degli iperparametri.
    :return: Dizionario degli iperparametri suggeriti.
    """
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    ff_dim = trial.suggest_categorical('ff_dim', [512, 1024, 2048])
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    num_layers = trial.suggest_categorical('num_layers', [2, 4, 6])
    dropout_p = trial.suggest_float('dropout_p', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    return {
        'hidden_dim': hidden_dim,
        'ff_dim': ff_dim,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'dropout_p': dropout_p,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }


def objective(
        trial: Trial,
        descriptions,
        vhdl_codes,
        vocab,
        max_len_description,
        max_len_vhdl,
        padding_idx,
        bos_idx,
        max_decoding_length,
        k=5
):
    """
    Funzione obiettivo per l'ottimizzazione basata su Optuna.

    :param trial: Istanza del trial Optuna.
    :param descriptions: Lista di descrizioni (testo).
    :param vhdl_codes: Lista di codici VHDL associati.
    :param vocab: Oggetto del vocabolario per tokenizzazione e decodifica.
    :param max_len_description: Lunghezza massima delle descrizioni.
    :param max_len_vhdl: Lunghezza massima dei codici VHDL.
    :param padding_idx: Indice del padding nel vocabolario.
    :param bos_idx: Indice del token di inizio sequenza.
    :param max_decoding_length: Lunghezza massima durante la decodifica.
    :param k: Numero di fold per la cross-validation.
    :return: Perdita media di validazione.
    """
    # Definisci lo spazio di ricerca e campiona gli iperparametri
    hyperparams = define_search_space(trial)
    print(f"Testing hyperparameters: {hyperparams}")

    # Esegui la k-fold cross-validation con gli iperparametri campionati
    avg_val_loss = perform_k_fold_cross_validation(
        descriptions=descriptions,
        vhdl_codes=vhdl_codes,
        vocab=vocab,
        hyperparams=hyperparams,
        trial=trial,
        padding_idx=padding_idx,
        bos_idx=bos_idx,
        max_len_description=max_len_description,
        max_len_vhdl=max_len_vhdl,
        max_decoding_length=max_decoding_length,  # Passa correttamente questo parametro
        k=k,
        epochs=100,
        patience=20
    )
    return avg_val_loss


# Funzione di cross-validation
def perform_k_fold_cross_validation(
        descriptions: List[str],
        vhdl_codes: List[str],
        vocab: Vocabulary,
        hyperparams: Dict[str, any],
        padding_idx: int,
        bos_idx: int,
        max_len_description: int,
        max_len_vhdl: int,
        max_decoding_length: int,
        k: int = 5,
        epochs: int = 200,
        patience: int = 20
) -> float:
    """
    Esegue k-fold cross-validation per valutare le prestazioni del modello con gli iperparametri forniti.

    :param descriptions: Lista di descrizioni (testo).
    :param vhdl_codes: Lista di codici VHDL associati.
    :param vocab: Oggetto del vocabolario per tokenizzazione e decodifica.
    :param hyperparams: Dizionario contenente gli iperparametri del modello.
    :param padding_idx: Indice del padding nel vocabolario.
    :param bos_idx: Indice del token di inizio sequenza.
    :param max_len_description: Lunghezza massima delle descrizioni.
    :param max_len_vhdl: Lunghezza massima dei codici VHDL.
    :param max_decoding_length: Lunghezza massima durante la decodifica.
    :param k: Numero di fold per la cross-validation.
    :param epochs: Numero di epoche per l'addestramento.
    :param patience: Numero massimo di epoche senza miglioramento prima dell'interruzione anticipata.
    :return: Perdita media di validazione.
    """
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    all_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(descriptions)):
        print(f"  Fold {fold + 1}/{k}")

        # Suddivisione dei dati per il fold corrente
        train_descriptions = [descriptions[i] for i in train_idx]
        train_vhdl = [vhdl_codes[i] for i in train_idx]
        val_descriptions = [descriptions[i] for i in val_idx]
        val_vhdl = [vhdl_codes[i] for i in val_idx]

        # Creazione dei dataset e dei DataLoader per il fold corrente
        train_dataset = PaddedVHDLDataset(train_descriptions, train_vhdl, vocab, max_len_description, max_len_vhdl)
        val_dataset = PaddedVHDLDataset(val_descriptions, val_vhdl, vocab, max_len_description, max_len_vhdl)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

        # Inizializzazione del modello per il fold corrente con gli iperparametri specificati
        model = Transformer(
            vocab_size=len(vocab),
            hidden_dim=hyperparams['hidden_dim'],
            ff_dim=hyperparams['ff_dim'],
            num_heads=hyperparams['num_heads'],
            num_layers=hyperparams['num_layers'],
            max_decoding_length=max_decoding_length,
            padding_idx=padding_idx,
            bos_idx=bos_idx,
            dropout_p=hyperparams['dropout_p']
        ).to(device)

        # Definizione del criterio di perdita, ottimizzatore e scheduler
        criterion = LabelSmoothingLoss(classes=len(vocab), smoothing=0.1, ignore_index=padding_idx)
        optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams['learning_rate'],
                                      weight_decay=hyperparams['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=20)

        # Addestramento del modello sul fold corrente
        train_and_save_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            padding_idx=padding_idx,
            epochs=epochs,
            save_path=f"llm_vhdl_model_fold_{fold + 1}.pth",
            patience=patience
        )

        # Valutazione sul set di validazione dopo l'addestramento
        val_loss = evaluate(model, val_loader, criterion, padding_idx)
        all_val_losses.append(val_loss)
        print(f"Fold {fold + 1} Val Loss: {val_loss}")

    # Calcola la media delle perdite di validazione
    avg_val_loss = sum(all_val_losses) / k
    print(f"  Average Val Loss: {avg_val_loss}")
    return avg_val_loss


def plot_learning_curves(train_losses, val_losses):
    """
    Traccia le curve di apprendimento per le perdite di training e validazione.

    :param train_losses: Lista delle perdite del set di training durante le epoche.
    :param val_losses: Lista delle perdite del set di validazione durante le epoche.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Perdita di Training')
    plt.plot(epochs, val_losses, label='Perdita di Validazione')
    plt.xlabel('Epoche')
    plt.ylabel('Perdita')
    plt.title('Curve di Apprendimento')
    plt.legend()
    plt.grid(True)
    plt.show()


def define_grid_search_space():
    """
    Definisce lo spazio di ricerca per la grid search degli iperparametri.

    :return: Dizionario contenente i possibili valori per ciascun iperparametro.
    """
    return {
        'hidden_dim': [128, 256, 512],
        'ff_dim': [512, 1024, 2048],
        'num_heads': [2, 4, 8],
        'num_layers': [2, 4, 6],
        'dropout_p': [0.1, 0.3, 0.5],
        'learning_rate': [1e-5, 1e-4, 1e-3],
        'weight_decay': [1e-5, 1e-4, 1e-3]
    }


def perform_grid_search(
        descriptions,
        vhdl_codes,
        vocab,
        max_len_description,
        max_len_vhdl,
        padding_idx,
        bos_idx,
        max_decoding_length,
        k=5
):
    """
    Esegue una grid search su uno spazio di iperparametri per identificare la migliore combinazione.

    :param descriptions: Lista di descrizioni (testo).
    :param vhdl_codes: Lista di codici VHDL associati.
    :param vocab: Oggetto del vocabolario per tokenizzazione e decodifica.
    :param max_len_description: Lunghezza massima delle descrizioni.
    :param max_len_vhdl: Lunghezza massima dei codici VHDL.
    :param padding_idx: Indice del padding nel vocabolario.
    :param bos_idx: Indice del token di inizio sequenza.
    :param max_decoding_length: Lunghezza massima durante la decodifica.
    :param k: Numero di fold per la cross-validation.
    :return: Tuple contenente i migliori iperparametri e la perdita media di validazione associata.
    """
    search_space = define_grid_search_space()
    all_hyperparams = list(product(*search_space.values()))
    all_keys = list(search_space.keys())

    best_loss = float('inf')
    best_params = None

    for i, hyperparam_values in enumerate(all_hyperparams):
        # Costruisce il dizionario degli iperparametri
        hyperparams = dict(zip(all_keys, hyperparam_values))
        print(f"Testing combination {i + 1}/{len(all_hyperparams)}: {hyperparams}")

        # Calcola la perdita media di validazione per questa combinazione di iperparametri
        avg_val_loss = perform_k_fold_cross_validation(
            descriptions=descriptions,
            vhdl_codes=vhdl_codes,
            vocab=vocab,
            hyperparams=hyperparams,
            padding_idx=padding_idx,
            bos_idx=bos_idx,
            max_len_description=max_len_description,
            max_len_vhdl=max_len_vhdl,
            max_decoding_length=max_decoding_length,
            k=k,
            epochs=100,
            patience=20
        )

        print(f"Average Val Loss for combination {i + 1}/{len(all_hyperparams)}: {avg_val_loss}")

        # Se la perdita media di validazione è migliore, aggiorna i migliori iperparametri
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_params = hyperparams

    print("Best hyperparameters:", best_params)
    print("Best average validation loss:", best_loss)

    return best_params, best_loss


def simple_vhdl_formatter(code: str) -> str:
    """
    Formatta il codice VHDL, migliorando leggibilità e conformità agli standard.

    :param code: Codice VHDL grezzo da formattare.
    :return: Codice VHDL formattato.
    """
    # Separare i token <BOS> e <EOS>
    code = code.strip()
    bos_token = ''
    eos_token = ''
    if code.startswith('<BOS>'):
        bos_token = '<BOS>'
        code = code[len('<BOS>'):].strip()
    if code.endswith('<EOS>'):
        eos_token = '<EOS>'
        code = code[:-len('<EOS>')].strip()

    # Converte le parole chiave in minuscolo
    keywords = [
        'library', 'use', 'entity', 'is', 'port', 'in', 'out', 'std_logic', 'std_logic_vector',
        'architecture', 'of', 'begin', 'end', 'process', 'if', 'then', 'elsif', 'else', 'signal',
        'type', 'array', 'others', 'downto', 'to', 'and', 'or', 'not', 'when', 'case', 'loop',
        'wait', 'until', 'generate', 'constant', 'variable', 'function', 'procedure', 'package',
        'body', 'assert', 'report', 'severity'
    ]
    for kw in keywords:
        code = re.sub(r'\b' + kw + r'\b', kw.lower(), code, flags=re.IGNORECASE)

    # Rimozione degli spazi attorno ai punti nelle dichiarazioni 'use'
    code = re.sub(r'\buse\s+([\w]+)\s*\.\s*([\w]+)\s*\.\s*([\w]+)\s*;', r'use \1.\2.\3;', code)
    code = re.sub(r'\buse\s+([\w]+)\s*\.\s*([\w]+)\s*;', r'use \1.\2;', code)

    # Correzione delle virgolette nei valori esadecimali
    code = re.sub(r'x\s*\'([0-9A-Fa-f]+)\'', r'x"\1"', code)

    # Normalizza gli spazi nel codice
    code = re.sub(r'\s+', ' ', code)

    # Correzione degli operatori con spazi indesiderati
    def fix_operator_spaces(code):
        operators = ['<=', '=>', '/=', ':=', '=']
        for op in operators:
            pattern = r'\s*'.join(re.escape(c) for c in op)
            code = re.sub(r'\s*' + pattern + r'\s*', f' {op} ', code)
        return code

    code = fix_operator_spaces(code)

    # Aggiunge nuove linee dopo ';' e parole chiave specifiche
    code = re.sub(r';', ';\n', code)
    code = re.sub(
        r'\b(begin|architecture|entity|process|if|elsif|else|then|port|is|case|when|loop|signal|type|package|function|procedure)\b',
        r'\n\1', code, flags=re.IGNORECASE)
    code = re.sub(r'\)\s*;', ');\n', code)

    # Gestione speciale di 'end' per evitare nuove linee tra 'end' e il contesto
    code = re.sub(r'\bend\s+(\w+)\s*;', r'\nend \1;', code, flags=re.IGNORECASE)
    code = re.sub(r'\bend\b\s*;', r'\nend;', code, flags=re.IGNORECASE)

    # Rimozione di spazi extra attorno a parentesi e virgole
    code = re.sub(r'\s*\(\s*', '(', code)
    code = re.sub(r'\s*\)\s*', ')', code)
    code = re.sub(r'\s*,\s*', ', ', code)

    # Dividi il codice in linee
    lines = code.strip().split('\n')
    formatted_lines = []
    indent_level = 0
    indent_string = '    '  # 4 spazi per livello di indentazione

    # Parole chiave che aumentano o diminuiscono l'indentazione
    increase_indent = [
        'entity', 'architecture', 'process', 'begin', 'if', 'then', 'else', 'elsif', 'case',
        'when', 'for', 'loop', 'generate', 'package', 'function', 'procedure'
    ]
    decrease_indent = [
        'end', 'else', 'elsif', 'when'
    ]

    for i, line in enumerate(lines):
        line = line.strip()

        # Salta le linee vuote
        if not line:
            continue

        # Determina se deve diminuire l'indentazione
        if any(line.lower().startswith(kw) for kw in decrease_indent):
            indent_level = max(indent_level - 1, 0)

        # Gestisce le parentesi chiuse allineate
        if line.endswith(');') or line.endswith(')'):
            formatted_lines.append(f"{indent_string * indent_level}{line}")
            # Dopo una parentesi chiusa, non aumentare l'indentazione
            continue

        # Applica l'indentazione
        formatted_lines.append(f"{indent_string * indent_level}{line}")

        # Determina se deve aumentare l'indentazione
        if any(line.lower().startswith(kw) for kw in increase_indent):
            indent_level += 1

        # Gestisce il caso di 'end' seguito da una parola chiave
        if re.match(r'^end\b', line, re.IGNORECASE):
            indent_level = max(indent_level - 1, 0)

    # Ricostruisce il codice formattato
    formatted_code = '\n'.join(formatted_lines)

    # Rimuove linee vuote multiple
    formatted_code = re.sub(r'\n+', '\n', formatted_code)

    # Rimuove spazi extra prima di punti e virgola
    formatted_code = re.sub(r'\s+;', ';', formatted_code)

    # Rimozione di spazi extra attorno alle parentesi
    formatted_code = re.sub(r'\(\s+', '(', formatted_code)
    formatted_code = re.sub(r'\s+\)', ')', formatted_code)

    # Rimuove spazi extra prima di virgole
    formatted_code = re.sub(r'\s+,', ',', formatted_code)
    formatted_code = re.sub(r',\s+', ', ', formatted_code)

    # Aggiunge i token <BOS> e <EOS> se presenti
    if bos_token:
        formatted_code = bos_token + '\n' + formatted_code
    if eos_token:
        formatted_code = formatted_code + '\n' + eos_token

    return formatted_code.strip()


def generate_multiple_vhdl_codes(model, input_text, vocab, k, max_length=512):
    """
    Genera più codici VHDL dato un testo di input utilizzando campionamento casuale.

    :param model: Modello Transformer addestrato per generare codice VHDL.
    :param input_text: Testo descrittivo di input.
    :param vocab: Vocabolario per tokenizzazione e decodifica.
    :param k: Numero di varianti di codice da generare.
    :param max_length: Lunghezza massima della sequenza generata.
    :return: Lista di varianti di codice VHDL generate.
    """
    model.eval()
    device = next(model.parameters()).device

    # Tokenizzazione
    input_tokens = vocab.encode(input_text)
    input_tensor = torch.tensor([input_tokens], device=device)
    src_mask = (input_tensor != vocab.token2index[Vocabulary.PAD]).unsqueeze(-2)

    generated_codes = []
    for _ in range(k):
        # Inizializza l'input del decodificatore con il token BOS
        output_tokens = [vocab.token2index[Vocabulary.BOS]]
        for i in range(max_length):
            output_tensor = torch.tensor([output_tokens], device=device)
            tgt_mask = (output_tensor != vocab.token2index[Vocabulary.PAD]).unsqueeze(-2)
            tgt_mask = tgt_mask & (
                    1 - torch.triu(
                torch.ones((1, output_tensor.size(-1), output_tensor.size(-1)), device=device), diagonal=1
            )
            ).bool()

            # Genera le probabilità del prossimo token
            output = model(input_tensor, output_tensor, src_mask, tgt_mask)
            next_token_logits = output[:, -1, :]
            probabilities = F.softmax(next_token_logits, dim=-1).squeeze()

            # Campiona il prossimo token
            next_token = torch.multinomial(probabilities, num_samples=1).item()
            output_tokens.append(next_token)

            if next_token == vocab.token2index[Vocabulary.EOS]:
                break

        vhdl_code = vocab.decode(output_tokens)
        generated_codes.append(vhdl_code)

    return generated_codes


def evaluate_pass_at_k(model, test_inputs, vocab, k):
    """
    Valuta la capacità del modello di generare almeno un codice VHDL sintatticamente corretto su k tentativi.

    :param model: Modello addestrato.
    :param test_inputs: Input di test (descrizioni testuali).
    :param vocab: Vocabolario utilizzato per la tokenizzazione.
    :param k: Numero di codici generati per ogni input.
    :return: pass@k, indicatore del successo su k tentativi.
    """
    success = 0

    # Genera k codici VHDL per l'input fornito
    generated_codes = generate_multiple_vhdl_codes(model, input_text, vocab, k)

    # Verifica la correttezza sintattica dei codici generati
    passed = False
    for code in generated_codes:
        formattato = format_vhdl_with_VHDLFormatter(code)
        sanificato = sanitize_vhdl_code(formattato)
        fixato = fix_vhdl_for_synthesis(sanificato)
        finale = check_and_correct_vhdl_syntax(fixato)
        if check_vhdl_syntax(finale):
            passed = True
            break  # Almeno un codice corretto trovato

    if passed:
        success += 1

    pass_at_k = success
    print(f'pass@{k}: {pass_at_k:.2f}')
    return pass_at_k


def evaluate_syntactic_correctness(finale):
    """
    Valuta se un codice VHDL è sintatticamente corretto.

    :param finale: Codice VHDL da valutare.
    :return: Indicatore di successo (1 se corretto, 0 altrimenti).
    """
    success = 0

    if check_vhdl_syntax(finale):
        success += 1

    syntactic_correctness = success
    print(f'Correttezza Sintattica: {syntactic_correctness:.2f}')
    return syntactic_correctness


def check_vhdl_syntax(vhdl_code: str) -> bool:
    """
    Verifica la sintassi di un codice VHDL utilizzando GHDL.

    :param vhdl_code: Codice VHDL da verificare.
    :return: True se il codice è sintatticamente corretto, False altrimenti.
    """
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

    :param vhdl_code: Codice VHDL da analizzare.
    :return: Nome dell'entità top-level.
    :raises ValueError: Se nessuna entità è trovata.
    """
    # Espressione regolare per trovare tutte le entità
    pattern = re.compile(r'\bentity\s+(\w+)\s+is', re.IGNORECASE)
    entities = pattern.findall(vhdl_code)

    if not entities:
        raise ValueError("Nessuna entità trovata nel codice VHDL.")

    # Supponiamo che l'ultima entità sia il modulo top-level
    return entities[-1]


def fix_unknown_identifiers(code: str, error_output: str) -> str:
    """
    Risolve identificatori sconosciuti nel codice VHDL dichiarandoli come segnali.

    :param code: Codice VHDL originale.
    :param error_output: Output degli errori generato da GHDL.
    :return: Codice VHDL corretto.
    """
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
    """
    Corregge iterativamente errori di sintassi VHDL analizzando con GHDL.
    """
    iteration = 0
    while iteration < max_iterations:
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.vhd', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_name = temp_file.name

        try:
            # Analizza il codice con GHDL
            result = subprocess.run(['ghdl', '-a', temp_file_name], capture_output=True, text=True)
            if result.returncode != 0:
                error_output = result.stderr

                # Controlla errori specifici
                if "is expected instead of ';'" in error_output:
                    # Trova il punto con la parentesi mancante
                    lines = code.split('\n')
                    for i, line in enumerate(lines):
                        if re.search(r'\b(conv_integer|to_integer)\([^)]*$', line):
                            missing_parentheses = line.count('(') - line.count(')')
                            if missing_parentheses > 0:
                                lines[i] += ')' * missing_parentheses
                    code = '\n'.join(lines)

                # Incrementa il contatore di iterazioni
                iteration += 1
            else:
                # Codice corretto
                return code
        finally:
            os.remove(temp_file_name)

    # Restituisce il codice anche se non corretto dopo tutte le iterazioni
    return code



def fix_type_declaration(code: str) -> str:
    """
    Corregge dichiarazioni duplicate di tipi nel codice VHDL.
    """
    code = re.sub(r'TYPE\s+ram_type\s+IS\s+TYPE\s+ram_type\s+IS', 'TYPE ram_type IS', code, flags=re.IGNORECASE)
    return code


def fix_entity_port_declaration(code: str) -> str:
    """
    Corregge la dichiarazione delle porte in un'entità VHDL, assicurando che le parentesi e i punti e virgola siano corretti.

    :param code: Codice VHDL contenente errori di sintassi nelle porte.
    :return: Codice VHDL con la dichiarazione delle porte corretta.
    """
    # Trova l'inizio della dichiarazione delle porte
    entity_start_match = re.search(r'ENTITY\s+(\w+)\s+IS\s+PORT\s*\(', code, re.IGNORECASE)
    if not entity_start_match:
        # Nessuna entità trovata, ritorna il codice originale
        return code

    entity_name = entity_start_match.group(1)
    start_index = entity_start_match.end()  # Posizione dopo '('
    end_index = start_index

    # Usa una pila per gestire le parentesi
    paren_stack = ['(']  # Abbiamo trovato la prima '('

    while paren_stack and end_index < len(code):
        char = code[end_index]
        if char == '(':
            paren_stack.append('(')
        elif char == ')':
            paren_stack.pop()
        end_index += 1

    if paren_stack:
        # Parentesi non bilanciate, non possiamo correggere, ritorna il codice originale
        return code

    # Estrai la dichiarazione delle porte
    ports_str = code[start_index:end_index - 1]  # Esclude l'ultima ')'

    # Processa le porte
    port_lines = ports_str.strip().split(';')
    port_lines = [line.strip() for line in port_lines if line.strip()]

    # Assicura che ogni linea di porta termini con ';' tranne l'ultima
    for i in range(len(port_lines) - 1):
        if not port_lines[i].endswith(';'):
            port_lines[i] += ';'

    # Ricostruisci la dichiarazione delle porte
    ports_fixed = ';\n    '.join(port_lines)

    # Ricostruisci l'entità
    entity_decl = f'ENTITY {entity_name} IS PORT (\n    {ports_fixed}\n);\nEND {entity_name};'

    # Rimuovi eventuali dichiarazioni duplicate di 'END entity_name;' nel codice successivo
    code_after_entity = code[end_index:].strip()
    code_after_entity = re.sub(r'\bEND\s+' + re.escape(entity_name) + r'\s*;\s*', '', code_after_entity,
                               flags=re.IGNORECASE)

    # Ricostruisci il codice completo
    code = code[:entity_start_match.start()] + entity_decl + '\n' + code_after_entity

    return code


def fix_assignments(code: str) -> str:
    """
    Corregge errori comuni nelle assegnazioni, come parentesi chiuse in eccesso e mancati punti e virgola.

    :param code: Codice VHDL con assegnazioni errate.
    :return: Codice VHDL corretto.
    """
    # Rimuove parentesi chiuse in eccesso nelle assegnazioni
    code = re.sub(r';\s*\)', ';', code)
    # Assicura che ogni assegnazione termini con ';'
    code_lines = code.split('\n')
    for i, line in enumerate(code_lines):
        if '<=' in line and not line.strip().endswith(';'):
            code_lines[i] = line.strip() + ';'
    code = '\n'.join(code_lines)
    return code


def fix_end_statements(code: str) -> str:
    """
    Corregge errori nelle dichiarazioni `END`, come mancati punti e virgola o errata associazione ai blocchi.

    :param code: Codice VHDL contenente errori di `END`.
    :return: Codice VHDL corretto.
    """
    # Corregge 'END register_file;' in posizioni errate all'interno dell'architettura
    code = re.sub(r'END\s+register_file\s*;', 'END IF;', code, flags=re.IGNORECASE)
    # Assicura che 'END PROCESS' termini con ';'
    code = re.sub(r'END\s+PROCESS\s*(?!;)', 'END PROCESS;', code, flags=re.IGNORECASE)
    # Assicura che 'END arqregfile' termini con ';'
    code = re.sub(r'END\s+arqregfile\s*(?!;)', 'END arqregfile;', code, flags=re.IGNORECASE)
    return code


def balance_parentheses(code: str) -> str:
    """
    Corregge errori di parentesi non bilanciate nelle linee del codice VHDL.

    :param code: Codice VHDL con parentesi sbilanciate.
    :return: Codice VHDL con parentesi bilanciate.
    """
    code_lines = code.split('\n')
    for i, line in enumerate(code_lines):
        open_parens = line.count('(')
        close_parens = line.count(')')
        if open_parens > close_parens:
            missing_parens = open_parens - close_parens
            code_lines[i] = line + ')' * missing_parens
    code = '\n'.join(code_lines)
    return code


def fix_extra_semicolons(code: str) -> str:
    """
    Rimuove punti e virgola in eccesso o posizionati in modo errato nel codice VHDL.

    :param code: Codice VHDL con errori nei punti e virgola.
    :return: Codice VHDL corretto.
    """
    # Rimuove punti e virgola dopo THEN, ELSE, ELSIF, BEGIN
    code = re.sub(r'\b(THEN|ELSE|ELSIF\b.*?\)|BEGIN)\s*;', r'\1', code, flags=re.IGNORECASE)
    # Rimuove punti e virgola multipli consecutivi
    code = re.sub(r';\s*;', ';', code)
    return code


def evaluate_synthesis_success(vhdl_code):
    """
    Valuta se un codice VHDL è sintetizzabile, verificando sia la sintassi che la sintesi.

    :param vhdl_code: Codice VHDL da verificare.
    :return: Tasso di successo della sintesi.
    """
    success = 0

    if check_vhdl_syntax(vhdl_code) and synthesize_vhdl_code(vhdl_code):
        success += 1

    synthesis_success_rate = success
    print(f'Tasso di Successo della Sintesi: {synthesis_success_rate:.2f}')
    return synthesis_success_rate


def format_vhdl_with_VHDLFormatter(vhdl_code: str) -> str:
    """
    Formatta il codice VHDL utilizzando il servizio online VHDLFormatter.

    :param vhdl_code: Codice VHDL non formattato.
    :return: Codice VHDL formattato.
    """
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
    """
    Rimuove i token speciali all'inizio e alla fine del codice generato
    """
    return vhdl_code.replace('< BOS >', '').replace('< EOS >', '').strip()


def remove_spaces_in_hex_declarations(vhdl_code):
    """
    Rimuove spazi superflui nelle dichiarazioni esadecimali nel codice VHDL.
    """
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
    Sostituisce `conv_integer` con `to_integer(unsigned(...))`
    """
    # Trova tutte le occorrenze di conv_integer(variable) e le sostituisce con to_integer(unsigned(variable))
    code = re.sub(r'conv_integer\s*\(\s*(\w+)\s*\)', r'to_integer(unsigned(\1))', code, flags=re.IGNORECASE)
    return code


def synthesize_vhdl_code(vhdl_code: str) -> bool:
    """
    Sintetizza un codice VHDL utilizzando GHDL e genera una netlist in formato Verilog.

    Questa funzione verifica la sintassi del codice VHDL fornito, lo compila con GHDL
    e successivamente tenta di sintetizzarlo in una netlist Verilog. La funzione tiene conto di eventuali
    flag specifici, come `-fsynopsys`, nel caso in cui il codice utilizzi librerie deprecate come `std_logic_unsigned`.

    :param vhdl_code: Codice VHDL formattato.
    :return: Booleano che indica se il codice è stato sintetizzato o no.
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
    Corregge il codice VHDL per risolvere problemi comuni di sintesi:
    - Allinea le larghezze dei bit nei confronti logici.
    - Evita l'inferenza di latch assegnando valori di default ai segnali non assegnati in tutti i percorsi.

    :param vhdl_code: Codice VHDL.
    :return: Codice VHDL pronto per la sintesi.
    """
    logger = logging.getLogger('fix_vhdl_for_synthesis')
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

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
                logger.info(
                    f"Adjusted comparison for signal '{signal}': '{signal} {operator} \"{literal}\"' -> '{signal} {operator} \"{adjusted_literal}\"'")
                return f'{signal} {operator} "{adjusted_literal}"'
            elif lit_length > sig_width:
                adjusted_literal = literal[-sig_width:]
                logger.info(
                    f"Trimmed comparison for signal '{signal}': '{signal} {operator} \"{literal}\"' -> '{signal} {operator} \"{adjusted_literal}\"'")
                return f'{signal} {operator} "{adjusted_literal}"'
        return match.group(0)

    vhdl_code = compare_pattern.sub(replace_comparisons, vhdl_code)

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
            body_lines = body.splitlines()
            new_body = '\n' + default_assignments + '\n'.join(body_lines) + '\n'
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
    """
    Inserisce la parola chiave `BEGIN` dopo le dichiarazioni dell'architettura nel codice VHDL.

    :param vhdl_code: Codice VHDL.
    :return: Codice VHDL modificato.
    """
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
    """
    Sanifica il codice VHDL applicando una serie di correzioni:
    - Rimuove token speciali.
    - Bilancia parentesi e corregge errori di sintassi comuni.
    """
    # Correzioni di base già esistenti
    code = remove_special_tokens(code)
    code = remove_spaces_in_library_declaration(code)
    code = replace_std_logic_unsigned(code)
    code = fix_assignments(code)
    code = fix_extra_semicolons(code)

    # Nuovo bilanciamento delle parentesi con maggiore precisione
    code_lines = code.split('\n')
    balanced_code_lines = []
    for line in code_lines:
        # Corregge parentesi mancanti in funzioni come conv_integer
        if re.search(r'\b(conv_integer|to_integer)\([^)]*$', line):
            missing_parentheses = line.count('(') - line.count(')')
            if missing_parentheses > 0:
                line += ')' * missing_parentheses
        # Rimuove punti e virgola extra
        line = line.replace(';;', ';')
        balanced_code_lines.append(line)
    code = '\n'.join(balanced_code_lines)

    # Rimuove parentesi extra alla fine del codice
    while code.endswith(')'):
        code = code[:-1]

    return code



def evaluate_functional_accuracy(finale):
    """
    Valuta l'accuratezza funzionale del codice VHDL.

    :param: Codice VHDL da verificare.
    :return: Accuratezza funzionale del codice.
    """
    success = 0

    # Verifica sintattica prima della verifica funzionale
    if check_vhdl_syntax(finale):
        try:
            if check_functional_accuracy(finale):
                success += 1
        except Exception as e:
            print(f"Errore durante il test funzionale: {e}")

    functional_accuracy = success
    print(f'Accuratezza Funzionale: {functional_accuracy:.2f}')
    return functional_accuracy


def check_functional_accuracy(vhdl_code):
    """
    Verifica l'accuratezza funzionale del codice VHDL utilizzando GHDL.

    :param: Codice VHDL da verificare.
    :return: Booleano che restituisce True se i test passano.
    """
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
    """
    Genera un testbench per il modulo register_file (esempio)
    """
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


def generate_testbench2():
    testbeanch_code = """
    LIBRARY IEEE;
    USE IEEE.STD_LOGIC_1164.ALL;
    USE IEEE.NUMERIC_STD.ALL;
    
    ENTITY DataMemory_tb IS
    END DataMemory_tb;
    
    ARCHITECTURE behavior OF DataMemory_tb IS
    
        -- Component Declaration
        COMPONENT DataMemory
            PORT (
                enableMem : IN STD_LOGIC;
                reset : IN STD_LOGIC;
                cRD : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
                address : IN STD_LOGIC_VECTOR(31 DOWNTO 0);
                wrEnMem : IN STD_LOGIC;
                datoToWr : OUT STD_LOGIC_VECTOR(31 DOWNTO 0)
            );
        END COMPONENT;
    
        -- Signals to connect to the DUT (Device Under Test)
        SIGNAL enableMem_tb : STD_LOGIC := '0';
        SIGNAL reset_tb : STD_LOGIC := '0';
        SIGNAL cRD_tb : STD_LOGIC_VECTOR(31 DOWNTO 0) := (OTHERS => '0');
        SIGNAL address_tb : STD_LOGIC_VECTOR(31 DOWNTO 0) := (OTHERS => '0');
        SIGNAL wrEnMem_tb : STD_LOGIC := '0';
        SIGNAL datoToWr_tb : STD_LOGIC_VECTOR(31 DOWNTO 0);
    
        -- Clock signal
        SIGNAL clk_tb : STD_LOGIC := '0';
    
    BEGIN
        -- Instantiate the Unit Under Test (UUT)
        uut: DataMemory
            PORT MAP (
                enableMem => enableMem_tb,
                reset => reset_tb,
                cRD => cRD_tb,
                address => address_tb,
                wrEnMem => wrEnMem_tb,
                datoToWr => datoToWr_tb
            );
    
        -- Clock generation process (50 MHz clock)
        clk_process : PROCESS
        BEGIN
            clk_tb <= '0';
            WAIT FOR 10 ns;
            clk_tb <= '1';
            WAIT FOR 10 ns;
        END PROCESS;
    
        -- Stimulus process
        stimulus_process: PROCESS
        BEGIN
            -- Test case 1: Reset the memory
            enableMem_tb <= '1';
            reset_tb <= '1';
            WAIT FOR 20 ns;
            reset_tb <= '0';
    
            -- Test case 2: Write data to address 5
            address_tb <= STD_LOGIC_VECTOR(TO_UNSIGNED(5, 32));
            cRD_tb <= X"0000000A";  -- Writing value 10
            wrEnMem_tb <= '1';
            WAIT FOR 20 ns;
    
            -- Test case 3: Read data from address 5
            wrEnMem_tb <= '0';
            WAIT FOR 20 ns;
    
            -- Test case 4: Write data to another address
            address_tb <= STD_LOGIC_VECTOR(TO_UNSIGNED(10, 32));
            cRD_tb <= X"00000014";  -- Writing value 20
            wrEnMem_tb <= '1';
            WAIT FOR 20 ns;
    
            -- Test case 5: Read data from address 10
            wrEnMem_tb <= '0';
            address_tb <= STD_LOGIC_VECTOR(TO_UNSIGNED(10, 32));
            WAIT FOR 20 ns;
    
            -- Test case 6: Read data from an uninitialized address
            address_tb <= STD_LOGIC_VECTOR(TO_UNSIGNED(15, 32));
            WAIT FOR 20 ns;
    
            -- End simulation
            WAIT;
        END PROCESS;
    
    END behavior;
    """
    return testbeanch_code


if __name__ == "__main__":
    # Percorso del dataset completo
    dataset_path = 'vhdl_dataset_finale_nan.csv'

    # Caricamento e preprocessamento del sottoinsieme per ricerca iperparametri
    subset_size = 50  # Numero di esemplari per la ricerca degli iperparametri
    train_descriptions_subset, train_vhdl_subset, val_descriptions_subset, val_vhdl_subset, test_descriptions_subset, test_vhdl_subset = load_and_preprocess_data(dataset_path, subset_size=subset_size)

    # Applicazione la data augmentation al sottoinsieme
    augmentation_factor = 0
    train_descriptions_subset, train_vhdl_subset = augment_data(train_descriptions_subset, train_vhdl_subset, augmentation_factor=augmentation_factor)

    print(f"Numero di righe dopo la data augmentation (sottoinsieme): {len(train_descriptions_subset)}")

    # Creazione del vocabolario basato sul sottoinsieme
    vocab_subset = create_vocabulary(train_descriptions_subset, train_vhdl_subset, val_descriptions_subset, val_vhdl_subset, test_descriptions_subset, test_vhdl_subset)

    # Salvataggio il vocabolario del sottoinsieme
    with open('vocab_subset.pkl', 'wb') as vocab_file:
        pickle.dump(vocab_subset, vocab_file)

    # Massima lunghezza delle sequenze nel sottoinsieme
    max_len_description_subset = max(len(vocab_subset.encode(desc)) for desc in train_descriptions_subset)
    max_len_vhdl_subset = max(len(vocab_subset.encode(vhdl)) for vhdl in train_vhdl_subset)

    # Definizione le variabili necessarie per il sottoinsieme
    max_decoding_length = 512
    padding_idx_subset = vocab_subset.token2index[Vocabulary.PAD]
    bos_idx_subset = vocab_subset.token2index[Vocabulary.BOS]

    # Creazione del dataset e DataLoader per l'ottimizzazione
    full_descriptions_subset = train_descriptions_subset
    full_vhdl_subset = train_vhdl_subset

    # Definizione dei parametri fissi per l'ottimizzazione
    k = 3  # Numero di folds
    n_trials = 100  # Numero di trial Optuna
    timeout = 600  # Tempo massimo in secondi (opzionale)

    # Creazione dello studio Optuna
    study = optuna.create_study(direction='minimize')  # Minimizzare la perdita

    # Esecuzione dell'ottimizzazione su un sottoinsieme ridotto
    study.optimize(
        lambda trial: objective(
            trial,
            descriptions=full_descriptions_subset,
            vhdl_codes=full_vhdl_subset,
            vocab=vocab_subset,
            max_len_description=max_len_description_subset,
            max_len_vhdl=max_len_vhdl_subset,
            padding_idx=padding_idx_subset,
            bos_idx=bos_idx_subset,
            max_decoding_length=max_decoding_length,
            k=k
        ),
        n_trials=n_trials,
        timeout=timeout
    )

    # Migliori iperparametri trovati
    print("Best hyperparameters: ", study.best_params)
    print("Best average validation loss: ", study.best_value)


    # Esegui la grid search
    best_params, best_loss = perform_grid_search(
        descriptions=full_descriptions_subset,
        vhdl_codes=full_vhdl_subset,
        vocab=vocab_subset,
        max_len_description=max_len_description_subset,
        max_len_vhdl=max_len_vhdl_subset,
        padding_idx=padding_idx_subset,
        bos_idx=bos_idx_subset,
        max_decoding_length=max_decoding_length,
        k=2  # Numero di folds
    )

    # Salva i migliori iperparametri
    with open('best_hyperparams_grid_search.json', 'r') as f:
        # json.dump(best_params, f)
        best_params = json.load(f)

    # Caricamento intero dataset e applicazione la data augmentation
    subset_sizef = 400
    train_descriptions_full, train_vhdl_full, val_descriptions_full, val_vhdl_full, test_descriptions_full, test_vhdl_full = load_and_preprocess_data(dataset_path, subset_size=subset_sizef)

    # Applica la data augmentation all'intero dataset
    train_descriptions_full, train_vhdl_full = augment_data(train_descriptions_full, train_vhdl_full, augmentation_factor=augmentation_factor)

    print(f"Numero di righe dopo la data augmentation (full dataset): {len(train_descriptions_full)}")

    # Creazione del vocabolario basato sull'intero dataset
    vocab_full = create_vocabulary(train_descriptions_full, train_vhdl_full, val_descriptions_full, val_vhdl_full, test_descriptions_full, test_vhdl_full)

    # Salvataggio il vocabolario completo
    with open('vocab_full.pkl', 'wb') as vocab_file:
        pickle.dump(vocab_full, vocab_file)

    # Massima lunghezza delle sequenze nell'intero dataset
    max_len_description_full = max(len(vocab_full.encode(desc)) for desc in train_descriptions_full)
    max_len_vhdl_full = max(len(vocab_full.encode(vhdl)) for vhdl in train_vhdl_full)

    # Definisci le variabili necessarie per l'intero dataset
    padding_idx_full = vocab_full.token2index[Vocabulary.PAD]
    bos_idx_full = vocab_full.token2index[Vocabulary.BOS]

    # Creazione del DataLoader per il set di validazione finale
    val_dataset_final = PaddedVHDLDataset(val_descriptions_full, val_vhdl_full, vocab_full, max_len_description_full, max_len_vhdl_full)
    val_loader_final = DataLoader(val_dataset_final, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # 6. Addestramento del modello finale con i migliori iperparametri su tutto il dataset
    final_model = Transformer(
        vocab_size=len(vocab_full),
        hidden_dim=best_params['hidden_dim'],
        ff_dim=best_params['ff_dim'],
        num_heads=best_params['num_heads'],
        num_layers=best_params['num_layers'],
        max_decoding_length=max_decoding_length,
        padding_idx=padding_idx_full,
        bos_idx=bos_idx_full,
        dropout_p=best_params['dropout_p']
    ).to(device)

    # Creazione del DataLoader completo per l'addestramento finale
    final_dataset = PaddedVHDLDataset(train_descriptions_full, train_vhdl_full, vocab_full, max_len_description_full, max_len_vhdl_full)
    final_loader = DataLoader(final_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # Definizione del criterio, ottimizzatore e scheduler per il modello finale
    # final_criterion = LabelSmoothingLoss(classes=len(vocab_full), smoothing=0.1, ignore_index=padding_idx_full)
    # final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    # final_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='min', factor=0.3, patience=20, min_lr=1e-5)

    final_criterion = torch.nn.CrossEntropyLoss(ignore_index=padding_idx_full)
    # final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_params['learning_rate'])

    torch.cuda.empty_cache()

    # Addestramento del modello finale con il set completo
    train_losses, val_losses = train_and_save_model(
        final_model,
        final_loader,
        val_loader_final,
        final_criterion,
        final_optimizer,
        padding_idx=padding_idx_full,
        epochs=400,
        save_path="llm_vhdl_model_final.pth",
        patience=200
    )

    # Visualizzazione le curve di apprendimento dopo l'addestramento finale
    plot_learning_curves(train_losses, val_losses)

    # Generazione del codice VHDL per un input specifico
    input_text = "implements a ram memory."
    vhdl_code = generate_vhdl_code(final_model, input_text, vocab_full)
    print(f"Codice VHDL Generato:\n{vhdl_code}")
    # exit(0)
    formattato = format_vhdl_with_VHDLFormatter(vhdl_code)
    print(f"Formattato:\n{formattato}")

    sanificato = sanitize_vhdl_code(formattato)
    print(f"Sanificato:\n{sanificato}")

    fixato = fix_vhdl_for_synthesis(sanificato)
    print(f"Fixato:\n{fixato}")

    finale = check_and_correct_vhdl_syntax(fixato)
    print(f"Finale:\n{finale}")

    print("SINTESI \n")
    synthesis_success_rate = evaluate_synthesis_success(finale)
    success = synthesize_vhdl_code(finale)
    if success:
        logging.info("Sintesi completata con successo!")
    else:
        logging.error("Sintesi fallita.")

    # PASS AT K
    pass_at_k = evaluate_pass_at_k(final_model, input_text, vocab_full, 5)
    print(f"Pass at k:{pass_at_k}")

    # CORRETTEZZA SINTATTICA
    correttezza_sintattica = evaluate_syntactic_correctness(finale)
    print(f"Correttezza sintattica: {correttezza_sintattica}")

    # ACCURATEZZA FUNZIONALE
    accuratezza_funzionale = evaluate_functional_accuracy(finale)
    print(f"Accuratezza funzionale: {accuratezza_funzionale}")
