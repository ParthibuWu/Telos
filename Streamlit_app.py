from __future__ import annotations

import io
from typing import List

import pandas as pd
import streamlit as st
from Bio import SeqIO

from Project__Telos.Telos_prime.composition import atgc_content, gc_fraction
from Project__Telos.Telos_prime.fasta_io import FastaRecord, ProcessedFastaRecord, process_fasta_record
from Project__Telos.Telos_prime.Chaos__game import records_to_fcgr_matrix
from Project__Telos.Telos_prime.PCA import run_pca, pairwise_distances

st.set_page_config(page_title="FASTA Comparator (FCGR + PCA)", layout="wide")


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
    st.caption("Upload FASTA files, inspect base composition, and compare sequences via FCGR + PCA.")

    with st.sidebar:
        st.header("Settings")
        k = st.slider("K-mer size (k)", min_value=2, max_value=8, value=4, step=1)
        normalize = st.checkbox("Normalize FCGR (frequencies, not raw counts)", value=True)
        treat_u_as_t = st.checkbox("Treat U as T (for RNA sequences)", value=True)
        n_components = st.slider("PCA components", min_value=2, max_value=5, value=2, step=1)
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
    st.dataframe(summary_df, use_container_width=True)

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

    with col2:
        if actual_components >= 2:
            plot_df = pd.DataFrame(
                {
                    "PC1": coords[:, 0],
                    "PC2": coords[:, 1],
                    "ID": ids,
                }
            )
            st.scatter_chart(plot_df, x="PC1", y="PC2", color=None, size=80)
        else:
            st.write("Need at least 2 components to plot PC1 vs PC2.")

    st.write("**PCA coordinates**")
    coord_cols = {f"PC{i+1}": coords[:, i] for i in range(actual_components)}
    coord_df = pd.DataFrame({"ID": ids, **coord_cols})
    st.dataframe(coord_df, use_container_width=True, hide_index=True)

    st.write("**Pairwise Euclidean distance (PCA space)**")
    dist = pairwise_distances(coords)
    dist_df = pd.DataFrame(dist, index=ids, columns=ids).round(4)
    st.dataframe(dist_df, use_container_width=True)

    st.download_button(
        "Download PCA coordinates as CSV",
        data=coord_df.to_csv(index=False).encode("utf-8"),
        file_name="pca_coordinates.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()