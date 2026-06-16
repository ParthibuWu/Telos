from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .composition import CANONICAL
from .fasta_io import ProcessedFastaRecord


def kmer_to_fcgr_position(kmer: str) -> Tuple[int, int]:
    x = 0
    y = 0
    for ch in kmer:
        x *= 2
        y *= 2
        if ch == "A":
            pass
        elif ch == "C":
            y += 1
        elif ch == "G":
            x += 1
            y += 1
        elif ch == "T":
            x += 1
        else:
            raise ValueError(f"Non-canonical base found in k-mer: {kmer}")
    return y, x


def sequence_to_fcgr(seq: str, k: int = 6, normalize: bool = True) -> np.ndarray:
    s = (seq or "").strip().upper()
    grid_size = 2 ** k
    fcgr = np.zeros((grid_size, grid_size), dtype=float)

    if len(s) < k:
        return fcgr

    valid_kmers = 0
    for i in range(len(s) - k + 1):
        kmer = s[i:i + k]
        if not all(ch in CANONICAL for ch in kmer):
            continue
        row, col = kmer_to_fcgr_position(kmer)
        fcgr[row, col] += 1
        valid_kmers += 1

    if normalize and valid_kmers > 0:
        fcgr = fcgr / valid_kmers

    return fcgr


def records_to_fcgr_matrix(
    records: List[ProcessedFastaRecord],
    k: int = 6,
    normalize: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    features = []
    ids = []
    for record in records:
        fcgr = sequence_to_fcgr(record.clean_sequence, k=k, normalize=normalize)
        features.append(fcgr.flatten())
        ids.append(record.record_id)

    X = np.array(features)
    return X, ids