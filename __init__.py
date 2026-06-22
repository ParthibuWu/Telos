# Telos_prime/__init__.py

from .composition import (
    CANONICAL,
    BaseComposition,
    clean_atgc,
    atgc_content,
    gc_fraction,
)

from .fasta_io import (
    FastaRecord,
    ProcessedFastaRecord,
    read_fasta,
    read_fasta_stream,
    process_fasta_record,
    process_fasta_file,
    records_to_summary_rows,
)

# New clustering module
from .clustering import (
    run_kmeans,
    run_dbscan,
    plot_dbscan_results,
    plot_k_distance,
)
