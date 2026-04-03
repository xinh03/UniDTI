import os
import random
import numpy as np
import torch
import dgl
import logging

CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9,
    "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17,
    "R": 18, "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25,
}

CHARPROTLEN = 25

# 药物序列词汇表（与dataloader.py中一致）
SMILES_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@#()[]=+-*/\\.")
SMILES_TO_IDX = {c: i+1 for i, c in enumerate(SMILES_CHARS)}
UNKNOWN_SMILES_IDX = len(SMILES_TO_IDX) + 1
SMILES_VOCAB_SIZE = UNKNOWN_SMILES_IDX + 1

IUPAC_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()-[]=#+")
IUPAC_TO_IDX = {c: i+1 for i, c in enumerate(IUPAC_CHARS)}
UNKNOWN_IUPAC_IDX = len(IUPAC_TO_IDX) + 1
IUPAC_VOCAB_SIZE = UNKNOWN_IUPAC_IDX + 1

def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def graph_collate_func(x):
    d, p, y = zip(*x)
    d = dgl.batch(d)
    return d, torch.tensor(np.array(p)), torch.tensor(y)

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)

def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in /"
                f"sequence category encoding, skip and treat as " f"padding."
            )
    return encoding

def smiles_to_indices(sequence, max_length=200):
    """Convert SMILES string to indices"""
    encoding = np.zeros(max_length, dtype=np.int64)
    for idx, char in enumerate(sequence[:max_length]):
        if char in SMILES_TO_IDX:
            encoding[idx] = SMILES_TO_IDX[char]
        else:
            encoding[idx] = UNKNOWN_SMILES_IDX
    return encoding

def iupac_to_indices(sequence, max_length=200):
    """Convert IUPAC string to indices"""
    encoding = np.zeros(max_length, dtype=np.int64)
    for idx, char in enumerate(sequence[:max_length]):
        if char in IUPAC_TO_IDX:
            encoding[idx] = IUPAC_TO_IDX[char]
        else:
            encoding[idx] = UNKNOWN_IUPAC_IDX
    return encoding