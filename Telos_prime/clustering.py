from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


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
    coords : (N, K_pca)
        PCA-reduced data.
    n_clusters : int or None
        Number of clusters. If None, auto-select using silhouette score.
    scale : bool
        Whether to scale features before clustering.
    random_state : int
        Random seed.
    auto_k_max : int
        Maximum K to test when auto-selecting.

    Returns
    -------
    labels : (N,)
        Cluster labels.
    centroids : (n_clusters, K_pca)
        Cluster centroids in the original PCA space.
    chosen_K : int
        Number of clusters used.
    silhouette_scores : array or None
        Silhouette scores for each K tested if auto; else None.
    """
    if scale:
        scaler = StandardScaler()
        data = scaler.fit_transform(coords)
    else:
        data = coords

    if n_clusters is None:
        best_k = 2
        best_score = -1
        scores = []
        max_k = min(auto_k_max, data.shape[0] - 1)
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = km.fit_predict(data)
            score = silhouette_score(data, labels)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_k = k
        n_clusters = best_k
        sil_scores = np.array(scores)
    else:
        sil_scores = None

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(data)
    centroids_scaled = km.cluster_centers_

    if scale:
        centroids_orig = scaler.inverse_transform(centroids_scaled)
    else:
        centroids_orig = centroids_scaled

    return labels, centroids_orig, n_clusters, sil_scores


def run_dbscan(
    coords: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    scale: bool = True,
) -> Tuple[np.ndarray, int, int, DBSCAN]:
    """
    Perform DBSCAN clustering on PCA coordinates.

    Parameters
    ----------
    coords : (N, K_pca)
        PCA-reduced data.
    eps : float
        Neighborhood radius.
    min_samples : int
        Minimum points to form a dense region.
    scale : bool
        Whether to scale features before clustering.

    Returns
    -------
    labels : (N,)
        Cluster labels (-1 for noise).
    n_clusters : int
        Number of clusters (excluding noise).
    n_noise : int
        Number of noise points.
    dbscan_model : DBSCAN
        Fitted DBSCAN object (useful for core sample indices).
    """
    if scale:
        scaler = StandardScaler()
        data = scaler.fit_transform(coords)
    else:
        data = coords

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    return labels, n_clusters, n_noise, dbscan


def plot_dbscan_results(
    coords: np.ndarray,
    labels: np.ndarray,
    dbscan_model: DBSCAN,
    ids: Optional[List[str]] = None,
    title: str = "DBSCAN Clustering",
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Enhanced DBSCAN visualization: core points (large), boundary points (small),
    noise (hollow squares). Points are colored by cluster.
    """
    # Determine point types
    core_mask = np.zeros(len(coords), dtype=bool)
    core_mask[dbscan_model.core_sample_indices_] = True
    noise_mask = (labels == -1)
    boundary_mask = ~(core_mask | noise_mask)

    fig, ax = plt.subplots(figsize=figsize)

    unique_clusters = sorted(set(labels) - {-1})
    if not unique_clusters:
        # All noise
        ax.scatter(coords[:, 0], coords[:, 1], s=60, facecolors='none',
                   edgecolors='grey', marker='s', label='Noise (-1)')
        ax.set_title("All points are noise")
        fig.tight_layout()
        return fig

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

    # Plot each cluster
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = (labels == cluster_id)
        color = colors[i % len(colors)]

        # Core points
        core_in_cluster = cluster_mask & core_mask
        if np.any(core_in_cluster):
            ax.scatter(
                coords[core_in_cluster, 0],
                coords[core_in_cluster, 1],
                s=120,
                c=[color],
                marker='o',
                edgecolors='k',
                linewidth=0.5,
                label=f'Cluster {cluster_id} (core)',
                alpha=0.9,
            )

        # Boundary points
        boundary_in_cluster = cluster_mask & boundary_mask
        if np.any(boundary_in_cluster):
            ax.scatter(
                coords[boundary_in_cluster, 0],
                coords[boundary_in_cluster, 1],
                s=40,
                c=[color],
                marker='o',
                edgecolors='k',
                linewidth=0.5,
                label=f'Cluster {cluster_id} (boundary)',
                alpha=0.6,
            )

    # Noise points
    if np.any(noise_mask):
        ax.scatter(
            coords[noise_mask, 0],
            coords[noise_mask, 1],
            s=60,
            facecolors='none',
            edgecolors='grey',
            linewidth=1.5,
            marker='s',
            label='Noise (-1)',
            alpha=0.8,
        )

    # Annotations
    if ids is not None:
        for idx, record_id in enumerate(ids):
            ax.annotate(
                record_id,
                (coords[idx, 0], coords[idx, 1]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
                alpha=0.7,
            )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_k_distance(
    coords: np.ndarray,
    min_samples: int = 5,
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    K-distance plot for DBSCAN eps selection.
    The 'elbow' is the optimal eps.
    """
    # Compute distances to the min_samples-th nearest neighbor
    neigh = NearestNeighbors(n_neighbors=min_samples)
    neigh.fit(coords)
    distances, _ = neigh.kneighbors(coords)
    # distances[:, -1] is the distance to the farthest neighbor among the min_samples
    k_dist = distances[:, -1]

    # Sort descending for elbow plot
    sorted_dist = np.sort(k_dist)[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, len(sorted_dist) + 1), sorted_dist, marker='o', markersize=4)
    ax.set_xlabel("Points (sorted by k-distance)")
    ax.set_ylabel(f"{min_samples}-th nearest neighbor distance")
    ax.set_title(f"K-distance plot (min_samples={min_samples}) - Look for the 'elbow'")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
# ----------------------------------------------------------------------
# NEW: Biological Insight Visualizations
# ----------------------------------------------------------------------

def get_genomic_coordinates(records: List[ProcessedFastaRecord]) -> List[Tuple[int, int]]:
    """
    Extract genomic start and end positions from each record.
    Expects record ID to contain coordinates like "NC_12345:1000-2000" or similar.
    If not found, we assign artificial coordinates based on cumulative sequence lengths.
    """
    import re
    coords = []
    cumulative = 0
    for rec in records:
        # Try to parse from record_id (format: "contig:start-end")
        match = re.search(r':(\d+)-(\d+)', rec.record_id)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
        else:
            # Assign artificial positions based on sequence length
            start = cumulative
            end = cumulative + len(rec.clean_sequence)
            cumulative = end
        coords.append((start, end))
    return coords


def plot_genomic_track(
    records: List[ProcessedFastaRecord],
    cluster_labels: np.ndarray,
    ids: List[str],
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Plot a 1D barcode track showing cluster assignments along the genome.
    Each sequence fragment is a horizontal bar from its start to end position,
    colored by its cluster label. Noise (label=-1) is shown in grey.
    """
    coords = get_genomic_coordinates(records)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine unique clusters (excluding noise for color mapping)
    unique_clusters = sorted(set(cluster_labels) - {-1})
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    color_map = {cl: colors[i % len(colors)] for i, cl in enumerate(unique_clusters)}
    color_map[-1] = 'grey'  # noise
    
    # Plot each sequence as a horizontal line segment
    for i, (start, end) in enumerate(coords):
        label = cluster_labels[i]
        color = color_map.get(label, 'black')
        ax.hlines(y=label, xmin=start, xmax=end, colors=color, linewidth=2, alpha=0.7)
    
    ax.set_xlabel("Genomic Position (bp)")
    ax.set_ylabel("Cluster ID")
    ax.set_title("Genome-Coordinate Mapping of Clusters")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cluster_kmer_heatmap(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    k: int,
    top_n: int = 30,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Heatmap of average k-mer frequencies for each cluster.
    Selects top_n most variable k-mers across all sequences.
    Requires get_kmer_labels from tsne_analysis.
    """
    # Import here to avoid circular import if needed
    from .tsne_analysis import get_kmer_labels
    
    # Compute variance per feature
    variances = np.var(X, axis=0)
    top_indices = np.argsort(variances)[-top_n:][::-1]
    
    # Get cluster labels (exclude noise if present)
    unique_clusters = sorted(set(cluster_labels) - {-1})
    if not unique_clusters:
        # All noise – plot nothing
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No clusters found (all noise)", ha='center', va='center')
        return fig
    
    # Compute mean FCGR per cluster for the selected features
    cluster_means = []
    for cl in unique_clusters:
        mask = (cluster_labels == cl)
        cluster_means.append(np.mean(X[mask][:, top_indices], axis=0))
    cluster_means = np.array(cluster_means)
    
    # Get k-mer labels
    labels_kmer = get_kmer_labels(k)
    selected_labels = [labels_kmer[i] for i in top_indices]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cluster_means, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(selected_labels)))
    ax.set_xticklabels(selected_labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(unique_clusters)))
    ax.set_yticklabels([f"Cluster {cl}" for cl in unique_clusters])
    ax.set_xlabel("k-mer")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Cluster-specific k-mer enrichment (top {top_n} most variable)")
    plt.colorbar(im, ax=ax, label='Mean Frequency')
    fig.tight_layout()
    return fig
