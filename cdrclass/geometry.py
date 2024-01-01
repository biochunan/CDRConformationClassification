# basic
import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.spatial.transform import Rotation

# logger 
from loguru import logger
from rich.logging import RichHandler
logger.configure(
    handlers=[
        {"sink": RichHandler(rich_tracebacks=True), "format": "{message}"}
    ]
)


# ---------------------------------------------
# func required for 
# `merge_in_torsional` and `merge_in_cartesian`
# ---------------------------------------------
# find CB residue identifiers
def cb_ri(struct_df: pd.DataFrame, cdr: str) -> List[str]:
    """
    Args:
        df: (pd.DataFrame) CDR loop structure dataframe
        cdr: (str) CDR identifier either "L1", "L2", "L3", "H1", "H2", "H3"

    Returns:
        ri_list: (List[str]) List of residue identifiers with CB atom present e.g. ['L50']
    """
    assert "ri" in struct_df.columns
    ri_list = struct_df[(struct_df.cdr == cdr) & (struct_df.atom == "CB")]["ri"].drop_duplicates().to_list()

    return ri_list


# extract non-GLY position CB atom coordinates
def extract_cb_atoms(struct_df: pd.DataFrame, cdr: str, ri_list: List[str]) -> np.ndarray:
    """
    Extract CB atom coordinates from a structure dataframe, excluding glycine residues specified in `gly_ri_list`

    Args:
        df: (pd.DataFrame) CDR loop structure dataframe: (pd.DataFrame)
        gly_ri_list: (List[str]) A list of residue identifiers for Glycines found in the structure DataFrame

    Returns:
        cb_coord: (np.ndarray) CB atom coordinates
    """
    assert "ri" in struct_df.columns

    return struct_df[(struct_df.cdr == cdr) & (struct_df.atom == "CB") & (struct_df.ri.isin(ri_list))][["x", "y", "z"]].to_numpy()


# extract non-GLY position CB atom coordinates
def extract_ca_atoms(struct_df: pd.DataFrame, cdr: str) -> np.ndarray:
    """
    Extract CA atom coordinates from a structure dataframe

    Args:
        df: (pd.DataFrame) CDR loop structure dataframe: (pd.DataFrame)

    Returns:
        ca_coord: (np.ndarray) CA atom coordinates
    """
    assert "ri" in struct_df.columns

    return struct_df[(struct_df.cdr == cdr) & (struct_df.atom == "CA")][["x", "y", "z"]].to_numpy()


# superimpose 2 sets of atoms
def superimpose_atoms(ref: np.ndarray, mob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """

    Args:
        ref: (np.ndarray) shape [N, 3] xyz coordinates of a set of atoms
        mob: (np.ndarray) shape [N, 3] xyz coordinates of a set of atoms

    Returns:
        rot_b2a: (np.ndarray) shape [3, 3] best estimate of the rotation that transforms b to a.
        rmsd: (float) Root-Mean-Square Deviation after fitting
    """
    # assert both sets of points have the same shape
    assert ref.shape == mob.shape
    n_res = ref.shape[0]

    # translation: center both sets to the origin
    ref_trans = ref - ref.mean(axis=0)
    mob_trans = mob - mob.mean(axis=0)

    # calculate rotation matrix
    rot_b2a, rmsd = Rotation.align_vectors(a=ref_trans, b=mob_trans)
    rot_b2a = rot_b2a.as_matrix()  # to matrix 3 x 3
    rmsd /= np.sqrt(n_res)  # RMSD correction, loss function did not divided by num of atoms

    return rot_b2a, rmsd


def atom_wise_dist(xyz_a: np.ndarray, xyz_b: np.ndarray):
    """ Calculate atom-wise distance between 2 sets of atoms """
    return np.linalg.norm(xyz_a - xyz_b, axis=1, ord=2)

