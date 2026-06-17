from __future__ import annotations

from typing import List, Tuple, Optional
import itertools

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .composition import CANONICAL
from .fasta_io import ProcessedFastaRecord
from .Chaos__game import records_to_fcgr_matrix, kmer_to_fcgr_position


# ----------------------------------------------------------------------
# K-MER LABEL GENERATION (maps flattened FCGR index to k-mer string)
# ----------------------------------------------------------------------
def get_kmer_labels(k: int) -> List[str]:
    bases = ['A', 'C', 'G', 'T']
    grid_size = 2 ** k
    coord_to_kmer = {}
    for kmer_tuple in itertools.product(bases, repeat=k):
        kmer = ''.join(kmer_tuple)
        y, x = kmer_to_fcgr_position(kmer)
        coord_to_kmer[(y, x)] = kmer
    labels = []
    for y in range(grid_size):
        for x in range(grid_size):
            labels.append(coord_to_kmer.get((y, x), "?"))
    return labels


# ----------------------------------------------------------------------
# COVARIANCE COMPUTATION & VISUALISATION
# ----------------------------------------------------------------------
def compute_feature_covariance(X: np.ndarray) -> np.ndarray:
    """Compute feature covariance matrix (D x D)."""
    N = X.shape[0]
    X_centered = X - X.mean(axis=0, keepdims=True)
    S = (X_centered.T @ X_centered) / N
    return S


def plot_full_covariance(
    X: np.ndarray,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    return_fig: bool = False,
):
    """
    Plot the full feature covariance matrix as a heatmap.
    Returns S (the covariance matrix). If return_fig=True, returns
    (S, fig) instead, and leaves the figure open (caller is responsible
    for displaying/closing it, e.g. via st.pyplot(fig)).
    """
    N, D = X.shape
    S = compute_feature_covariance(X)

    fig = plt.figure(figsize=figsize)
    sns.heatmap(
        S, cmap='RdBu', center=0, square=True,
        cbar_kws={'label': 'Covariance'},
        xticklabels=False, yticklabels=False,
    )
    plt.title(f'Full Feature Covariance Matrix ({D}\u00d7{D})')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')

    if return_fig:
        return S, fig
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return S


def plot_covariance_subset(
    X: np.ndarray,
    n_features: int = 50,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> None:
    """Plot covariance for a subset of leading features."""
    if X.shape[1] < n_features:
        n_features = X.shape[1]
    X_sub = X[:, :n_features]
    S_sub = compute_feature_covariance(X_sub)

    plt.figure(figsize=figsize)
    sns.heatmap(
        S_sub, cmap='RdBu', center=0, square=True,
        cbar_kws={'label': 'Covariance'},
        xticklabels=False, yticklabels=False,
    )
    plt.title(f'Feature Covariance \u2013 First {n_features} Features')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_covariance_with_labels(
    X: np.ndarray,
    labels: Optional[List[str]] = None,
    step: int = 1,
    figsize: Tuple[int, int] = (16, 14),
    save_path: Optional[str] = None,
    return_fig: bool = False,
):
    """
    Plot feature covariance with k-mer labels on axes.
    Returns None by default. If return_fig=True, returns the fig
    instead (caller is responsible for displaying/closing it).
    """
    S = compute_feature_covariance(X)
    D = S.shape[0]

    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(
        S, cmap='RdBu', center=0, square=True,
        cbar_kws={'label': 'Covariance'},
        xticklabels=False, yticklabels=False,
    )

    if labels is not None and len(labels) == D:
        tick_indices = list(range(0, D, step))
        tick_labels = [labels[i] if i < D else '' for i in tick_indices]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax.set_yticks(tick_indices)
        ax.set_yticklabels(tick_labels, fontsize=6)

    plt.title(f'Feature Covariance Matrix ({D}\u00d7{D}) with k-mer labels (step={step})')
    plt.xlabel('k-mer')
    plt.ylabel('k-mer')
    plt.tight_layout()

    if return_fig:
        return fig
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return None


# ----------------------------------------------------------------------
# PCA + t-SNE PIPELINE
# ----------------------------------------------------------------------
def run_pca_tsne_on_fcgr(
    X: np.ndarray,
    pca_components: int = 50,
    perplexity: int = 30,
    random_state: int = 42,
) -> Tuple[np.ndarray, PCA, TSNE]:
    """
    Apply PCA for denoising, then t-SNE for 2D visualisation.
    Note: t-SNE requires perplexity < n_samples (roughly n_samples / 3
    as a safe rule of thumb), so callers with small datasets should
    lower perplexity accordingly.
    """
    n_components = min(pca_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    print(f"PCA reduced to {n_components} dimensions.")
    print(f"Explained variance by those components: {pca.explained_variance_ratio_.sum():.2%}")

    safe_perplexity = min(perplexity, max(1, X.shape[0] - 1))
    if safe_perplexity != perplexity:
        print(
            f"Reduced perplexity from {perplexity} to {safe_perplexity} "
            f"(perplexity must be less than the number of samples)."
        )

    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        random_state=random_state,
        max_iter=1000,
    )
    X_tsne = tsne.fit_transform(X_pca)
    print(f"t-SNE completed with perplexity={safe_perplexity}.")

    return X_tsne, pca, tsne


def plot_tsne_scatter(
    X_tsne: np.ndarray,
    labels: Optional[List[str]] = None,
    ids: Optional[List[str]] = None,
    title: str = "t-SNE of FCGR features",
    save_path: Optional[str] = None,
    return_fig: bool = False,
):
    """
    Scatter plot of t-SNE coordinates. If `labels` is provided (e.g. a
    species name, sample group, or category per sequence), points are
    colored by category with a legend. Otherwise points are plotted
    uniformly and annotated with `ids` if given.
    Returns None by default, or the fig if return_fig=True.
    """
    fig = plt.figure(figsize=(10, 8))

    if labels is not None:
        unique_labels = sorted(set(labels))
        cmap = plt.get_cmap("tab10")
        color_map = {lab: cmap(i % 10) for i, lab in enumerate(unique_labels)}
        colors = [color_map[lab] for lab in labels]
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.8, edgecolors='k')
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_map[lab], label=lab) for lab in unique_labels
        ]
        plt.legend(handles=legend_elements)
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.8, edgecolors='k')
        if ids is not None:
            for record_id, (x, y) in zip(ids, X_tsne):
                plt.annotate(record_id, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(title)
    plt.grid(alpha=0.3)

    if return_fig:
        return fig
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return None


def plot_explained_variance(
    pca: PCA,
    save_path: Optional[str] = None,
    return_fig: bool = False,
):
    fig = plt.figure(figsize=(10, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Cumulative Explained Variance')
    plt.legend()
    plt.grid(alpha=0.3)

    if return_fig:
        return fig
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return None


# ----------------------------------------------------------------------
# HIGH-LEVEL ENTRY POINT: run on YOUR real processed records
# ----------------------------------------------------------------------
def analyze_records(
    records: List[ProcessedFastaRecord],
    labels: Optional[List[str]] = None,
    k: int = 4,
    pca_components: int = 30,
    perplexity: int = 20,
    show_covariance: bool = True,
    save_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], PCA]:
    """
    Run the full covariance + PCA + t-SNE analysis on a list of your
    real ProcessedFastaRecord objects (e.g. from process_fasta_file).

    `labels`, if given, should be a list of category strings the same
    length as `records` (e.g. species name, group, or source file) used
    to color the t-SNE scatter plot. If omitted, points are plotted
    uniformly and annotated by record ID instead.

    Returns (X_tsne, X_raw_fcgr, ids, fitted_pca).
    """
    def out_path(name: str) -> Optional[str]:
        if save_dir is None:
            return None
        import os
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, name)

    print(f"Converting {len(records)} sequences to FCGR matrix with k={k}...")
    X, ids = records_to_fcgr_matrix(records, k=k, normalize=True)
    print(f"FCGR matrix shape: {X.shape}")

    if show_covariance:
        plot_full_covariance(X, save_path=out_path("covariance_full.png"))
        labels_kmer = get_kmer_labels(k)
        step = max(1, len(labels_kmer) // 32)
        plot_covariance_with_labels(
            X, labels=labels_kmer, step=step, save_path=out_path("covariance_labeled.png")
        )

    X_tsne, pca, _ = run_pca_tsne_on_fcgr(
        X, pca_components=pca_components, perplexity=perplexity
    )
    plot_explained_variance(pca, save_path=out_path("explained_variance.png"))
    plot_tsne_scatter(
        X_tsne, labels=labels, ids=ids,
        title=f"t-SNE of FCGR features (k={k})",
        save_path=out_path("tsne_scatter.png"),
    )

    return X_tsne, X, ids, pca


# ----------------------------------------------------------------------
# OPTIONAL DEMO PATH (synthetic data, for testing only)
# ----------------------------------------------------------------------
def generate_structured_dummy_records(
    n_gc: int = 30,
    n_at: int = 30,
    seq_length: int = 200,
) -> Tuple[List[ProcessedFastaRecord], List[str]]:
    """
    Generate two synthetic groups (GC-rich / AT-rich) for testing this
    module without needing real FASTA data. Not used in the real
    pipeline; this is here only as a sanity check / demo.
    """
    records = []
    labels = []
    bases = ['A', 'C', 'G', 'T']

    for i in range(n_gc):
        probs = [0.15, 0.35, 0.35, 0.15]
        seq = ''.join(np.random.choice(bases, size=seq_length, p=probs))
        records.append(_make_dummy_record(f"gc_{i+1}", seq))
        labels.append("GC-rich")

    for i in range(n_at):
        probs = [0.35, 0.15, 0.15, 0.35]
        seq = ''.join(np.random.choice(bases, size=seq_length, p=probs))
        records.append(_make_dummy_record(f"at_{i+1}", seq))
        labels.append("AT-rich")

    return records, labels


def _make_dummy_record(record_id: str, seq: str) -> ProcessedFastaRecord:
    """
    Build a real ProcessedFastaRecord for the demo path, filling
    required fields with placeholder values where they don't matter
    for FCGR computation (FCGR only reads .clean_sequence).
    """
    return ProcessedFastaRecord(
        record_id=record_id,
        description="synthetic demo record",
        raw_sequence=seq,
        clean_sequence=seq,
        raw_length=len(seq),
        clean_length=len(seq),
        gc_fraction=None,
        gc_percent=None,
        A=seq.count("A"),
        T=seq.count("T"),
        G=seq.count("G"),
        C=seq.count("C"),
    )


if __name__ == "__main__":
    print("Running tsne_analysis.py demo with synthetic GC-rich / AT-rich data...")
    records, labels = generate_structured_dummy_records(n_gc=40, n_at=40, seq_length=200)
    analyze_records(records, labels=labels, k=4, pca_components=30, perplexity=20)
