"""
Molecular featurization: ECFP fingerprints and physicochemical descriptors.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm


def smiles_to_ecfp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    """Convert a SMILES string to an ECFP fingerprint (as numpy array)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)


def featurize_dataframe(df: pd.DataFrame, smiles_col: str = "smiles",
                         radius: int = 2, n_bits: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """
    Featurize all molecules in a dataframe.
    Returns (features, valid_mask).
    """
    features = []
    valid = []
    for smi in tqdm(df[smiles_col], desc="Featurizing"):
        fp = smiles_to_ecfp(smi, radius=radius, n_bits=n_bits)
        if fp is not None:
            features.append(fp)
            valid.append(True)
        else:
            features.append(np.zeros(n_bits))
            valid.append(False)
    return np.array(features), np.array(valid)


def compute_rdkit_descriptors(smiles: str) -> dict | None:
    """Compute a selection of RDKit physicochemical descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "rings": Descriptors.RingCount(mol),
    }
