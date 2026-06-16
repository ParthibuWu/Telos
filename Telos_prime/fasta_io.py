from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, TextIO

from Bio import SeqIO

from .composition import atgc_content, clean_atgc, gc_fraction


@dataclass(frozen=True)
class FastaRecord:
    record_id: str
    description: str
    sequence: str


@dataclass(frozen=True)
class ProcessedFastaRecord:
    record_id: str
    description: str
    raw_sequence: str
    clean_sequence: str
    raw_length: int
    clean_length: int
    gc_fraction: Optional[float]
    gc_percent: Optional[float]
    A: int
    T: int
    G: int
    C: int


def read_fasta(path: str) -> Iterator[FastaRecord]:
    with open(path, "r", encoding="utf-8") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            yield FastaRecord(
                record_id=record.id,
                description=getattr(record, "description", ""),
                sequence=str(record.seq),
            )


def read_fasta_stream(handle: TextIO) -> Iterator[FastaRecord]:
    for record in SeqIO.parse(handle, "fasta"):
        yield FastaRecord(
            record_id=record.id,
            description=getattr(record, "description", ""),
            sequence=str(record.seq),
        )


def process_fasta_record(
    record: FastaRecord,
    treat_u_as_t: bool = True,
) -> ProcessedFastaRecord:
    clean_seq = clean_atgc(record.sequence, treat_u_as_t=treat_u_as_t)
    comp = atgc_content(record.sequence, treat_u_as_t=treat_u_as_t)
    gc = gc_fraction(record.sequence, treat_u_as_t=treat_u_as_t)

    return ProcessedFastaRecord(
        record_id=record.record_id,
        description=record.description,
        raw_sequence=record.sequence,
        clean_sequence=clean_seq,
        raw_length=len(record.sequence),
        clean_length=len(clean_seq),
        gc_fraction=gc,
        gc_percent=None if gc is None else 100 * gc,
        A=comp.counts["A"],
        T=comp.counts["T"],
        G=comp.counts["G"],
        C=comp.counts["C"],
    )


def process_fasta_file(
    path: str,
    max_records: Optional[int] = None,
    treat_u_as_t: bool = True,
) -> List[ProcessedFastaRecord]:
    processed: List[ProcessedFastaRecord] = []
    for i, record in enumerate(read_fasta(path), start=1):
        if max_records is not None and i > max_records:
            break
        processed.append(process_fasta_record(record, treat_u_as_t=treat_u_as_t))
    return processed


def records_to_summary_rows(records: List[ProcessedFastaRecord]) -> List[Dict]:
    rows = []
    for record in records:
        rows.append(
            {
                "ID": record.record_id,
                "Description": record.description,
                "Raw_length": record.raw_length,
                "Clean_length": record.clean_length,
                "GC_percent": record.gc_percent,
                "A": record.A,
                "T": record.T,
                "G": record.G,
                "C": record.C,
                "Sequence": record.clean_sequence,
            }
        )
    return rows