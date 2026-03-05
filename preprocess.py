import argparse
import os
import pandas as pd
import scanpy as sc
import anndata as ad


def preprocess_scrna(expr_path, label_path, output_h5, n_hvg=2500):
    """
    Preprocess scRNA-seq data and save into a single H5AD file.
    """

    print("=" * 60)
    print("Loading data...")
    print("=" * 60)

    # ----------------------------
    # Load expression matrix
    # ----------------------------
    df = pd.read_csv(expr_path, index_col=0)
    print(f"Expression shape: {df.shape}")

    # ----------------------------
    # Load labels
    # ----------------------------
    labels_df = pd.read_csv(label_path, index_col=0)

    # ----------------------------
    # Create AnnData
    # ----------------------------
    adata = sc.AnnData(df)
    adata.obs["cell_type"] = labels_df.iloc[:, 0].values

    print(f"Initial AnnData shape: {adata.shape}")

    # ============================================================
    # Step 1: Remove genes not expressed
    # ============================================================
    sc.pp.filter_genes(adata, min_cells=1)
    print(f"After gene filtering: {adata.shape}")

    # ============================================================
    # Step 2: Highly Variable Genes
    # ============================================================
    adata.layers["counts"] = adata.X.copy()

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_hvg,
        flavor="seurat_v3",
        layer="counts",
    )

    adata = adata[:, adata.var.highly_variable].copy()
    print(f"After HVG selection ({n_hvg}): {adata.shape}")

    # ============================================================
    # Step 3: Library normalization
    # ============================================================
    sc.pp.normalize_total(adata, target_sum=1e4)
    print("✓ Normalized to 10,000 counts per cell")

    # ============================================================
    # Step 4: Log transform
    # ============================================================
    sc.pp.log1p(adata)
    print("✓ Applied log1p transformation")

    # ============================================================
    # Save
    # ============================================================
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    adata.write_h5ad(output_h5)

    print("\n" + "=" * 60)
    print("Preprocessing completed!")
    print(f"Saved to: {output_h5}")
    print(f"Final shape: {adata.shape}")
    print("=" * 60)


# ================================================================
# Command Line Interface
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess scRNA-seq data and save as H5AD"
    )

    parser.add_argument(
        "--expr",
        type=str,
        required=True,
        help="Path to expression CSV (cells x genes)",
    )

    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to cell labels CSV",
    )

    parser.add_argument(
        "--out",
        type=str,
        default='./data/preprocessed_data.h5ad',
        help="Output .h5ad file path",
    )

    parser.add_argument(
        "--hvg",
        type=int,
        default=2500,
        help="Number of highly variable genes (default: 2500)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    preprocess_scrna(
        expr_path=args.expr,
        label_path=args.labels,
        output_h5=args.out,
        n_hvg=args.hvg,
    )