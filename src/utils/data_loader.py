"""
Data loading utilities for DEL screen data (WDR91.parquet).

Schema:
  COMPOUND_ID, LIBRARY_ID, BB1_ID, BB2_ID, BB3_ID — identifiers
  TARGET_VALUE  — raw DEL read count (0 for non-hits, 6–696 for hits)
  NTC_VALUE     — no-target control (null in current release)
  LABEL         — binary hit label (1 if TARGET_VALUE > 0)
  MW, ALOGP     — physicochemical properties
  ECFP4/6, FCFP4/6, MACCS, RDK, AVALON, ATOMPAIR, TOPTOR
                — fingerprints stored as sparse lists of ON-bit indices
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parents[2] / "data"

SCALAR_COLS = [
    "COMPOUND_ID", "LIBRARY_ID", "BB1_ID", "BB2_ID", "BB3_ID",
    "TARGET_ID", "TARGET_VALUE", "NTC_VALUE", "LABEL", "MW", "ALOGP",
]

FP_DIMS = {
    "ECFP4": 2048, "ECFP6": 2048,
    "FCFP4": 2048, "FCFP6": 2048,
    "MACCS": 167,
    "RDK": 2048,
    "AVALON": 512,
    "ATOMPAIR": 2048,
    "TOPTOR": 2048,
}


def load_scalar_data(path: str | Path) -> pd.DataFrame:
    """Load only scalar (non-fingerprint) columns. Fast and memory-efficient."""
    table = pq.read_table(str(path), columns=SCALAR_COLS)
    df = table.to_pandas()
    df["log1p_target"] = np.log1p(df["TARGET_VALUE"])
    df["lib_prefix"] = df["LIBRARY_ID"].str[:3]
    return df


def sparse_indices_to_matrix(index_lists: list[list[int]], n_bits: int) -> sp.csr_matrix:
    """
    Convert a list of ON-bit index lists to a sparse CSR matrix.
    Each row = one compound, each column = one fingerprint bit.
    """
    rows, cols = [], []
    for i, bits in enumerate(index_lists):
        for b in bits:
            if b < n_bits:
                rows.append(i)
                cols.append(b)
    data = np.ones(len(rows), dtype=np.uint8)
    return sp.csr_matrix((data, (rows, cols)), shape=(len(index_lists), n_bits))


def load_fingerprints(path: str | Path,
                      fp_name: str = "ECFP6",
                      batch_size: int = 50_000) -> tuple[sp.csr_matrix, pd.DataFrame]:
    """
    Load a single fingerprint column as a sparse CSR matrix, together with
    scalar metadata. Returns (X_sparse, df_scalars).

    Memory-efficient: reads in batches and stacks.
    """
    assert fp_name in FP_DIMS, f"Unknown fingerprint: {fp_name}. Choose from {list(FP_DIMS)}"
    n_bits = FP_DIMS[fp_name]
    cols_to_read = SCALAR_COLS + [fp_name]

    pf = pq.ParquetFile(str(path))
    scalar_frames, fp_matrices = [], []

    for batch in tqdm(pf.iter_batches(batch_size=batch_size, columns=cols_to_read),
                      desc=f"Loading {fp_name}", total=None):
        batch_df = batch.to_pandas()
        scalar_frames.append(batch_df[SCALAR_COLS])
        fp_matrices.append(sparse_indices_to_matrix(batch_df[fp_name].tolist(), n_bits))

    X = sp.vstack(fp_matrices, format="csr")
    df = pd.concat(scalar_frames, ignore_index=True)
    df["log1p_target"] = np.log1p(df["TARGET_VALUE"])
    df["lib_prefix"] = df["LIBRARY_ID"].str[:3]
    return X, df


def load_multi_fingerprints(path: str | Path,
                             fp_names: list[str],
                             batch_size: int = 50_000) -> tuple[sp.csr_matrix, pd.DataFrame]:
    """Load and horizontally concatenate multiple fingerprints."""
    matrices, df = None, None
    for fp in fp_names:
        X, df_fp = load_fingerprints(path, fp_name=fp, batch_size=batch_size)
        matrices = X if matrices is None else sp.hstack([matrices, X], format="csr")
        df = df_fp  # same rows every time
    return matrices, df
