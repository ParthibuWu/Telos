from __future__ import annotations

import io
import re
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import streamlit as st
from Bio import SeqIO
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Direct submodule imports (avoids __init__.py re‑export issues)
from Telos_prime.composition import atgc_content, gc_fraction
from Telos_prime.fasta_io import FastaRecord, ProcessedFastaRecord, process_fasta_record
from Telos_prime.Chaos__game import records_to_fcgr_matrix
from Telos_prime.PCA import run_pca, pairwise_distances
from Telos_prime.tsne_analysis import (
    run_pca_tsne_on_fcgr,
    plot_tsne_scatter,
    plot_explained_variance,
    plot_full_covariance,
    plot_covariance_with_labels,
    get_kmer_labels,
)
from Telos_prime.clustering import (
    run_kmeans,
    run_dbscan,
    plot_dbscan_results,
    plot_k_distance,
    plot_genomic_track,
    plot_cluster_kmer_heatmap,
)

st.set_page_config(page_title="FASTA Comparator (FCGR + PCA + Clustering)", layout="wide")


def parse_uploaded_fasta(uploaded_file) -> List[FastaRecord]:
    """
    Parse an uploaded FASTA file (Streamlit UploadedFile) into FastaRecord
    objects without writing to disk, since SeqIO needs a text handle.
    """
    raw_bytes = uploaded_file.read()
    text = raw_bytes.decode("utf-8", errors="replace")
    handle = io.StringIO(text)

    records = []
    for record in SeqIO.parse(handle, "fasta"):
        records.append(
            FastaRecord(
                record_id=record.id,
                description=getattr(record, "description", ""),
                sequence=str(record.seq),
            )
        )
    return records


def main() -> None:
    st.title("FASTA Sequence Comparator")
    st.caption(
        "Upload FASTA files, inspect base composition, and compare sequences via FCGR + PCA + t-SNE + Clustering."
    )

    with st.sidebar:
        st.header("Settings")
        k = st.slider("K-mer size (k)", min_value=2, max_value=8, value=4, step=1)
        if k >= 7:
            st.caption(
                f"⚠️ k={k} means {4**k:,} features per sequence. Covariance "
                f"heatmaps will be disabled at this size to avoid crashing."
            )
        normalize = st.checkbox("Normalize FCGR (frequencies, not raw counts)", value=True)
        treat_u_as_t = st.checkbox("Treat U as T (for RNA sequences)", value=True)
        n_components = st.slider("PCA components", min_value=2, max_value=5, value=2, step=1)
        st.divider()

        # --- Clustering section (unified) ---
        st.subheader("Clustering")
        run_clustering = st.checkbox("Run clustering on PCA coordinates", value=False)

        if run_clustering:
            cluster_method = st.selectbox(
                "Clustering algorithm",
                options=["K-Means", "DBSCAN"],
                index=0,
            )

            scale_pca = st.checkbox("Scale PCA features before clustering", value=True)

            if cluster_method == "K-Means":
                kmeans_auto = st.checkbox("Automatically select K (silhouette)", value=True)
                if not kmeans_auto:
                    kmeans_k = st.slider(
                        "Number of clusters (K)", min_value=2, max_value=10, value=3, step=1
                    )
                else:
                    kmeans_k = None
            else:  # DBSCAN
                eps = st.slider(
                    "eps (neighborhood radius)",
                    min_value=0.1,
                    max_value=5.0,
                    value=0.5,
                    step=0.1,
                    help="Larger eps connects more points; smaller eps creates more clusters/noise.",
                )
                min_samples = st.slider(
                    "min_samples",
                    min_value=2,
                    max_value=20,
                    value=5,
                    step=1,
                    help="Minimum points to form a dense region. Higher values create more noise.",
                )
        else:
            cluster_method = None
            scale_pca = False
            kmeans_k = None
            eps = 0.5
            min_samples = 5

        st.divider()

        # --- t-SNE settings ---
        st.subheader("t-SNE settings")
        run_tsne = st.checkbox("Run t-SNE analysis", value=False)
        tsne_pca_dims = st.slider("PCA dims before t-SNE", min_value=2, max_value=50, value=20, step=1)
        tsne_perplexity = st.slider("t-SNE perplexity", min_value=2, max_value=50, value=15, step=1)
        show_covariance = st.checkbox("Show covariance heatmaps", value=False)
        st.divider()
        st.caption(
            "Tip: lower k if your sequences are short, since k=6+ needs longer "
            "sequences to avoid a sparse, mostly-empty FCGR matrix."
        )

    uploaded_files = st.file_uploader(
        "Upload one or more FASTA files",
        type=["fasta", "fa", "fna", "txt"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload at least one FASTA file to get started. A file may contain multiple sequences.")
        return

    all_records: List[FastaRecord] = []
    for uploaded_file in uploaded_files:
        try:
            parsed = parse_uploaded_fasta(uploaded_file)
        except Exception as exc:
            st.error(f"Failed to parse {uploaded_file.name}: {exc}")
            continue

        if not parsed:
            st.warning(f"No sequences found in {uploaded_file.name}.")
            continue

        all_records.extend(parsed)

    if not all_records:
        st.error("No valid sequences were parsed from the uploaded files.")
        return

    st.success(f"Parsed {len(all_records)} sequence(s) from {len(uploaded_files)} file(s).")

    processed: List[ProcessedFastaRecord] = [
        process_fasta_record(r, treat_u_as_t=treat_u_as_t) for r in all_records
    ]

    st.subheader("Base composition summary")
    summary_df = pd.DataFrame(
        [
            {
                "ID": r.record_id,
                "Description": r.description,
                "Raw length": r.raw_length,
                "Clean length": r.clean_length,
                "GC %": None if r.gc_percent is None else round(r.gc_percent, 2),
                "A": r.A,
                "T": r.T,
                "G": r.G,
                "C": r.C,
            }
            for r in processed
        ]
    )
    st.dataframe(summary_df, width='stretch')

    st.download_button(
        "Download summary as CSV",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="fasta_summary.csv",
        mime="text/csv",
    )

    if len(processed) < 2:
        st.warning("Upload at least 2 sequences total to run PCA comparison.")
        return

    st.divider()
    st.subheader("FCGR + PCA comparison")

    with st.spinner("Computing FCGR matrix and PCA..."):
        X, ids = records_to_fcgr_matrix(processed, k=k, normalize=normalize)

        max_possible = min(X.shape[0], X.shape[1])
        actual_components = min(n_components, max_possible)
        if actual_components < n_components:
            st.info(
                f"Reduced PCA components to {actual_components} "
                f"(limited by {X.shape[0]} sequences / {X.shape[1]} features)."
            )

        coords, pca = run_pca(X, n_components=actual_components)

    variance_df = pd.DataFrame(
        {
            "Component": [f"PC{i+1}" for i in range(actual_components)],
            "Explained variance ratio": pca.explained_variance_ratio_,
        }
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Explained variance**")
        st.dataframe(variance_df, use_container_width=True, hide_index=True)
        st.caption(f"Cumulative: {pca.explained_variance_ratio_.sum():.4f}")

    # ----- CLUSTERING EXECUTION (Unified) -----
    cluster_labels = None
    centroids = None
    chosen_k = None
    sil_scores = None
    cluster_method_name = "None"
    dbscan_model = None

    if run_clustering and cluster_method == "K-Means":
        with st.spinner("Running K-Means clustering..."):
            if kmeans_auto:
                cluster_labels, centroids, chosen_k, sil_scores = run_kmeans(
                    coords, n_clusters=None, scale=scale_pca, random_state=0, auto_k_max=10
                )
                st.info(f"Auto-selected K = {chosen_k} based on silhouette score.")
                if sil_scores is not None:
                    sil_df = pd.DataFrame({
                        "K": range(2, min(10, coords.shape[0]-1)+1),
                        "Silhouette": sil_scores
                    })
                    st.dataframe(sil_df, width='stretch')
            else:
                cluster_labels, centroids, chosen_k, _ = run_kmeans(
                    coords, n_clusters=kmeans_k, scale=scale_pca, random_state=0
                )
                st.info(f"Used K = {kmeans_k}")

            # Compute silhouette score for the chosen clustering
            if scale_pca:
                scaler = StandardScaler()
                scaled_coords = scaler.fit_transform(coords)
                sil_score = silhouette_score(scaled_coords, cluster_labels)
            else:
                sil_score = silhouette_score(coords, cluster_labels)
            st.metric("Silhouette score", value=f"{sil_score:.4f}")
            cluster_method_name = "K-Means"

    elif run_clustering and cluster_method == "DBSCAN":
        with st.spinner("Running DBSCAN clustering..."):
            cluster_labels, n_clusters_found, n_noise, dbscan_model = run_dbscan(
                coords, eps=eps, min_samples=min_samples, scale=scale_pca
            )
            st.info(
                f"DBSCAN found {n_clusters_found} cluster(s) and {n_noise} noise point(s) "
                f"(labeled -1)."
            )
            if n_noise == coords.shape[0]:
                st.warning(
                    "All points are labeled as noise. Try increasing `eps` or decreasing `min_samples`."
                )
            elif n_clusters_found == 1:
                st.info("Only one cluster found. The data might be too dense; try decreasing `eps`.")
            cluster_method_name = "DBSCAN"

    # Display cluster assignments if clustering was run
    if cluster_labels is not None:
        cluster_df = pd.DataFrame({
            "ID": ids,
            "Cluster": cluster_labels
        })
        st.write(f"**Cluster assignments ({cluster_method_name})**")
        st.dataframe(cluster_df, width='stretch')

        st.download_button(
            f"Download {cluster_method_name} cluster assignments as CSV",
            data=cluster_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{cluster_method_name.lower()}_clusters.csv",
            mime="text/csv",
        )

    # --- Plot PCA scatter, enhanced for DBSCAN ---
    with col2:
        if actual_components >= 2:
            if cluster_method == "DBSCAN" and cluster_labels is not None and dbscan_model is not None:
                # Enhanced DBSCAN plot
                fig_dbscan = plot_dbscan_results(
                    coords,
                    cluster_labels,
                    dbscan_model,
                    ids=ids,
                    title=f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})",
                )
                st.pyplot(fig_dbscan)

                # Show k-distance plot for eps selection
                st.subheader("K-distance plot (eps selection)")
                st.caption(
                    "The optimal `eps` is where the plot has an 'elbow' – a sharp change in slope. "
                    "Points to the right of the elbow are outliers."
                )
                fig_kdist = plot_k_distance(coords, min_samples=min_samples)
                st.pyplot(fig_kdist)

            else:
                # Standard scatter chart for no clustering or K-Means
                plot_df = pd.DataFrame(
                    {
                        "PC1": coords[:, 0],
                        "PC2": coords[:, 1],
                        "ID": ids,
                    }
                )
                if cluster_labels is not None:
                    plot_df["Cluster"] = cluster_labels.astype(str)
                    st.scatter_chart(plot_df, x="PC1", y="PC2", color="Cluster", size=80)
                else:
                    st.scatter_chart(plot_df, x="PC1", y="PC2", color=None, size=80)

                if cluster_method == "K-Means" and centroids is not None:
                    st.caption(
                        f"Cluster centroids (PC1, PC2):\n" +
                        "\n".join([f"Cluster {i}: ({c[0]:.3f}, {c[1]:.3f})" for i, c in enumerate(centroids)])
                    )
        else:
            st.write("Need at least 2 components to plot PC1 vs PC2.")

    # --- Biological Insight Visualizations ---
    if cluster_labels is not None:
        st.divider()
        st.subheader("Biological Insight Visualizations")
        
        # 1. Genomic Track Plot
        st.write("**Genome-Coordinate Mapping Plot**")
        st.caption("Each horizontal bar represents a sequence fragment, colored by its cluster label. Noise is shown in grey.")
        with st.spinner("Building genomic track..."):
            fig_track = plot_genomic_track(processed, cluster_labels, ids)
            st.pyplot(fig_track)
        
        # 2. K-mer Frequency Heatmap per Cluster
        st.write("**Cluster-specific k-mer Enrichment Heatmap**")
        st.caption("Heatmap of the most variable k-mers, averaged per cluster. Rows = clusters, columns = k-mers.")
        with st.spinner("Computing k-mer heatmap..."):
            fig_kmer = plot_cluster_kmer_heatmap(X, cluster_labels, k=k, top_n=30)
            st.pyplot(fig_kmer)

    # --- Display detailed coordinates ---
    st.write("**PCA coordinates**")
    coord_cols = {f"PC{i+1}": coords[:, i] for i in range(actual_components)}
    if cluster_labels is not None:
        coord_cols["Cluster"] = cluster_labels
    coord_df = pd.DataFrame({"ID": ids, **coord_cols})
    st.dataframe(coord_df, width='stretch')

    st.write("**Pairwise Euclidean distance (PCA space)**")
    dist = pairwise_distances(coords)
    dist_df = pd.DataFrame(dist, index=ids, columns=ids).round(4)
    st.dataframe(dist_df, width='stretch')

    st.download_button(
        "Download PCA coordinates as CSV",
        data=coord_df.to_csv(index=False).encode("utf-8"),
        file_name="pca_coordinates.csv",
        mime="text/csv",
    )

    # ------------------------------------------------------------------
    # t-SNE + covariance section (optional, since it's heavier to compute)
    # ------------------------------------------------------------------
    if not run_tsne:
        st.divider()
        st.caption(
            "Enable \"Run t-SNE analysis\" in the sidebar to see a t-SNE embedding and covariance heatmaps."
        )
        return

    st.divider()
    st.subheader("t-SNE embedding")

    n_samples = X.shape[0]
    if n_samples < 3:
        st.warning("t-SNE needs at least 3 sequences to produce a meaningful embedding. Upload more sequences.")
        return

    with st.spinner("Running PCA + t-SNE..."):
        X_tsne, pca_tsne, _ = run_pca_tsne_on_fcgr(
            X, pca_components=tsne_pca_dims, perplexity=tsne_perplexity
        )

    # Color t-SNE by cluster labels if available, else just scatter
    tsne_colors = cluster_labels if cluster_labels is not None else None
    fig = plot_tsne_scatter(
        X_tsne,
        labels=tsne_colors.astype(str) if tsne_colors is not None else None,
        ids=ids if tsne_colors is None else None,
        title=f"t-SNE of FCGR features (k={k})",
        save_path=None,
        return_fig=True,
    )
    st.pyplot(fig)

    fig_var = plot_explained_variance(pca_tsne, save_path=None, return_fig=True)
    st.pyplot(fig_var)

    if show_covariance:
        st.subheader("Feature covariance")

        n_features = X.shape[1]
        MAX_SAFE_FEATURES = 4096  # corresponds to k <= 6 (4^6 = 4096)

        if n_features > MAX_SAFE_FEATURES:
            st.warning(
                f"Skipping covariance heatmap: at k={k}, the feature matrix has "
                f"{n_features:,} columns, producing a {n_features:,}\u00d7{n_features:,} "
                f"covariance matrix. This is too large to render safely and will "
                f"crash the browser or run out of memory. Lower k to 6 or below "
                f"(4096 features or fewer) to see covariance heatmaps."
            )
        else:
            st.caption(
                "Covariance between FCGR feature positions (k-mer bins) across "
                "your uploaded sequences. Large matrices may take a moment to render."
            )
            _, fig_cov = plot_full_covariance(X, save_path=None, return_fig=True)
            st.pyplot(fig_cov)

            if k <= 5:
                labels_kmer = get_kmer_labels(k)
                step = max(1, len(labels_kmer) // 32)
                fig_cov_labeled = plot_covariance_with_labels(
                    X, labels=labels_kmer, step=step, save_path=None, return_fig=True
                )
                st.pyplot(fig_cov_labeled)
            else:
                st.caption("Labeled covariance heatmap skipped for k > 5 (too many k-mers to label readably).")


if __name__ == "__main__":
    main()
