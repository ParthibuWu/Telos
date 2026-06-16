from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
from sklearn.decomposition import PCA

from .fasta_io import process_fasta_file, ProcessedFastaRecord
from .Chaos__game import records_to_fcgr_matrix


def load_records_from_paths(
    paths: List[str],
    max_records: int | None = None,
    treat_u_as_t: bool = True,
) -> List[ProcessedFastaRecord]:
    """
    Load and process records from one or more FASTA files.
    Each file may contain one or many sequences; all are pooled
    together into a single list for downstream comparison.
    """
    all_records: List[ProcessedFastaRecord] = []
    for path in paths:
        all_records.extend(
            process_fasta_file(
                path,
                max_records=max_records,
                treat_u_as_t=treat_u_as_t,
            )
        )
    return all_records


def run_pca(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 0,
) -> Tuple[np.ndarray, PCA]:

    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(X)
    return coords, pca


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """
    Euclidean distance matrix between sequences in PCA space.
    """
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def print_summary(
    ids: List[str],
    pca: PCA,
    coords: np.ndarray,
) -> None:
    print(f"Number of sequences: {len(ids)}")
    print(f"Components kept: {pca.n_components_}")
    print("Explained variance ratio per component:")
    for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"  PC{i}: {ratio:.4f}")
    print(f"Cumulative explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print()
    print("PC1/PC2 coordinates:")
    for record_id, coord in zip(ids, coords):
        pc1 = coord[0] if coord.shape[0] > 0 else float("nan")
        pc2 = coord[1] if coord.shape[0] > 1 else float("nan")
        print(f"  {record_id:>20s}  PC1={pc1: .4f}  PC2={pc2: .4f}")
    print()

    dist = pairwise_distances(coords)
    print("Pairwise Euclidean distance in PCA space:")
    header = "".join(f"{rid[:12]:>14s}" for rid in ids)
    print(" " * 14 + header)
    for i, rid in enumerate(ids):
        row = "".join(f"{dist[i, j]:14.4f}" for j in range(len(ids)))
        print(f"{rid[:12]:>14s}{row}")


def plot_pca(
    ids: List[str],
    coords: np.ndarray,
    out_path: str = "pca_plot.png",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coords[:, 0], coords[:, 1])
    for record_id, (x, y) in zip(ids, coords[:, :2]):
        ax.annotate(record_id, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("FCGR-based PCA of sequences")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare FASTA sequences via FCGR features + PCA."
    )
    parser.add_argument("fasta_files", nargs="+", help="One or more FASTA file paths.")
    parser.add_argument("-k", type=int, default=6, help="K-mer size for FCGR (default: 6).")
    parser.add_argument(
        "--components", type=int, default=2, help="Number of PCA components to keep."
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Max records to read per file (default: all).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable FCGR normalization (raw counts instead of frequencies).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a PC1 vs PC2 scatter plot to pca_plot.png.",
    )
    parser.add_argument(
        "--plot-out",
        default="pca_plot.png",
        help="Output path for the plot (default: pca_plot.png).",
    )
    args = parser.parse_args()

    records = load_records_from_paths(args.fasta_files, max_records=args.max_records)

    if len(records) < 2:
        raise SystemExit(
            f"Need at least 2 sequences to compare; found {len(records)}."
        )

    X, ids = records_to_fcgr_matrix(
        records,
        k=args.k,
        normalize=not args.no_normalize,
    )

    coords, pca = run_pca(X, n_components=args.components)

    print_summary(ids, pca, coords)

    if args.plot:
        plot_pca(ids, coords, out_path=args.plot_out)


if __name__ == "__main__":
    main()