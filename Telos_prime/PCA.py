from __future__ import annotations

import argparse
from typing import List, Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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
    """
    Fit PCA and return coordinates and the fitted PCA object.
    """
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(X)
    return coords, pca


def run_kmeans(
    coords: np.ndarray,
    n_clusters: Optional[int] = None,
    scale: bool = True,
    random_state: int = 0,
    auto_k_max: int = 10,
) -> Tuple[np.ndarray, np.ndarray, int, Optional[np.ndarray]]:
    """
    Perform K-means clustering on PCA coordinates.
    
    Parameters
    ----------
    coords : np.ndarray, shape (N, K_pca)
        PCA-reduced data.
    n_clusters : int or None
        Number of clusters. If None, we automatically choose the best K
        using silhouette score for K in 2..auto_k_max.
    scale : bool
        Whether to scale the features before clustering.
    random_state : int
        Random seed for reproducibility.
    auto_k_max : int
        Maximum K to test when auto-selecting.
    
    Returns
    -------
    labels : np.ndarray, shape (N,)
        Cluster labels for each sample.
    centroids : np.ndarray, shape (n_clusters, K_pca) or (n_clusters, K_pca_scaled)
        Cluster centroids (in the same space as the input coords, i.e., after scaling
        if scale=True, else original PCA space).
    chosen_K : int
        The actual number of clusters used.
    silhouette_scores : np.ndarray or None
        If auto-selection was used, returns the silhouette scores for each K tested;
        otherwise None.
    """
    if scale:
        scaler = StandardScaler()
        data = scaler.fit_transform(coords)
    else:
        data = coords

    if n_clusters is None:
        # Auto-select K using silhouette score
        best_k = 2
        best_score = -1
        scores = []
        for k in range(2, min(auto_k_max, data.shape[0]) + 1):
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = km.fit_predict(data)
            score = silhouette_score(data, labels)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_k = k
        n_clusters = best_k
        silhouette_scores = np.array(scores)
    else:
        silhouette_scores = None

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(data)
    centroids = km.cluster_centers_   # these are in the scaled space if scale=True

    # If we scaled, we might want to return centroids in the original PCA space
    # for plotting on the same axes as the data. We can store them separately.
    if scale:
        # Inverse transform centroids back to original PCA space
        centroids_orig = scaler.inverse_transform(centroids)
    else:
        centroids_orig = centroids

    return labels, centroids_orig, n_clusters, silhouette_scores


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
    cluster_labels: Optional[np.ndarray] = None,
    centroids: Optional[np.ndarray] = None,
) -> None:
    """
    Print PCA summary, and if clustering was performed, also print cluster assignments
    and centroids.
    """
    print(f"Number of sequences: {len(ids)}")
    print(f"Components kept: {pca.n_components_}")
    print("Explained variance ratio per component:")
    for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"  PC{i}: {ratio:.4f}")
    print(f"Cumulative explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print()

    print("PC1/PC2 coordinates:")
    for i, (record_id, coord) in enumerate(zip(ids, coords)):
        pc1 = coord[0] if coord.shape[0] > 0 else float("nan")
        pc2 = coord[1] if coord.shape[0] > 1 else float("nan")
        if cluster_labels is not None:
            print(f"  {record_id:>20s}  PC1={pc1: .4f}  PC2={pc2: .4f}  cluster={cluster_labels[i]}")
        else:
            print(f"  {record_id:>20s}  PC1={pc1: .4f}  PC2={pc2: .4f}")
    print()

    if cluster_labels is not None and centroids is not None:
        print("Cluster centroids in PCA space:")
        for k, cent in enumerate(centroids):
            cent_str = ", ".join(f"{c:.4f}" for c in cent[:2])  # show only first 2 dims
            print(f"  Cluster {k}: ({cent_str})")
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
    cluster_labels: Optional[np.ndarray] = None,
    centroids: Optional[np.ndarray] = None,
) -> None:
    """
    Plot PC1 vs PC2, optionally colored by cluster and with centroids marked.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))

    if cluster_labels is not None:
        # Color by cluster
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7)
        ax.legend(*scatter.legend_elements(), title="Clusters")
        # Plot centroids if provided
        if centroids is not None:
            ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red', edgecolors='white', linewidth=2, label='Centroids')
            ax.legend()
    else:
        ax.scatter(coords[:, 0], coords[:, 1])

    # Annotate points with IDs
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
        description="Compare FASTA sequences via FCGR features + PCA, with optional K-means clustering."
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
    # NEW arguments for K-means
    parser.add_argument(
        "--kmeans",
        type=int,
        default=None,
        help="Number of clusters for K-means (if not set, no clustering).",
    )
    parser.add_argument(
        "--kmeans-auto",
        action="store_true",
        help="Automatically determine best K using silhouette score (ignores --kmeans).",
    )
    parser.add_argument(
        "--scale-pca",
        action="store_true",
        help="Scale PCA features before clustering (recommended for K-means).",
    )
    parser.add_argument(
        "--kmeans-plot",
        action="store_true",
        help="If set, the PCA plot will be colored by cluster (requires --plot).",
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

    # ----- K-means clustering -----
    cluster_labels = None
    centroids = None
    if args.kmeans_auto:
        print("Auto-selecting K via silhouette score (2..10)...")
        cluster_labels, centroids, chosen_K, sil_scores = run_kmeans(
            coords, n_clusters=None, scale=args.scale_pca, random_state=0, auto_k_max=10
        )
        print(f"Chosen K = {chosen_K}")
        if sil_scores is not None:
            for k, score in enumerate(sil_scores, start=2):
                print(f"  K={k}: silhouette = {score:.4f}")
    elif args.kmeans is not None:
        print(f"Running K-means with K={args.kmeans}...")
        cluster_labels, centroids, chosen_K, _ = run_kmeans(
            coords, n_clusters=args.kmeans, scale=args.scale_pca, random_state=0
        )
        print(f"Clustering completed.")
    else:
        # No clustering
        cluster_labels = None
        centroids = None

    print_summary(ids, pca, coords, cluster_labels, centroids)

    if args.plot:
        plot_pca(
            ids,
            coords,
            out_path=args.plot_out,
            cluster_labels=cluster_labels if args.kmeans_plot else None,
            centroids=centroids if args.kmeans_plot else None,
        )


if __name__ == "__main__":
    main()
