import numpy as np
import pandas as pd
from typing import List, Tuple
# logging
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s {%(pathname)s:%(lineno)d} [%(levelname)s] %(name)s - %(message)s [%(threadName)s]',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def calc_dihedral(
    a_coords: np.ndarray,
    b_coords: np.ndarray,
    c_coords: np.ndarray,
    d_coords: np.ndarray,
    convert_to_degree: bool = True
) -> np.ndarray:
    # to unit vector
    def to_unit_vec(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    b1 = to_unit_vec(a_coords - b_coords).reshape((3,))
    b2 = to_unit_vec(b_coords - c_coords).reshape((3,))
    b3 = to_unit_vec(c_coords - d_coords).reshape((3,))

    n1 = to_unit_vec(np.cross(b1, b2))

    n2 = to_unit_vec(np.cross(b2, b3))

    # angle between n1 & b2 = 90 degree => ||m1|| = sin pi/2 = 1
    m1 = np.cross(n1, b2)

    dihedral = np.arctan2(np.dot(m1, n2), np.dot(n1, n2))  # np.dot(m1, n2) => y, np.dot(n1, n2) => x
    # dihedral = np.arctan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))

    if convert_to_degree:
        dihedral = dihedral * 180 / np.pi

    return dihedral


def calc_phi_psi_single_residue(struct_df: pd.DataFrame, node_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate phi and psi angles of the peptide between residue i (specified by node_id) and

    Args:
        struct_df (pd.DataFrame): _description_
        node_id (int): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    # get residues i-1, i, i+1
    ri = struct_df[struct_df.node_id == node_id]
    ra = struct_df[struct_df.node_id == node_id - 1]
    rb = struct_df[struct_df.node_id == node_id + 1]

    preceding_res_exist = True
    if ra.shape[0] == 0:
        logger.warning("Did not find preceding residue.")
        preceding_res_exist = False

    succeeding_res_exist = True
    if rb.shape[0] == 0:
        logger.warning("Did not find succeeding residue.")
        succeeding_res_exist = False

    # get coord
    ri_N = ri[ri.atom == "N"][["x", "y", "z"]].to_numpy()
    ri_CA = ri[ri.atom == "CA"][["x", "y", "z"]].to_numpy()
    ri_C = ri[ri.atom == "C"][["x", "y", "z"]].to_numpy()

    # phi and psi
    phi, psi = None, None
    if preceding_res_exist:
        # phi: i-1.C = i.N  - i.CA - i.C
        ra_C = ra[ra.atom == "C"][["x", "y", "z"]].to_numpy()
        phi = calc_dihedral(ra_C, ri_N, ri_CA, ri_C)

    if succeeding_res_exist:
        # psi: i.N   - i.CA - i.C  - i+1.N
        rb_N = rb[rb.atom == "N"][["x", "y", "z"]].to_numpy()
        psi = calc_dihedral(ri_N, ri_CA, ri_C, rb_N)

    return phi, psi


def calc_phi_psi_set_residues(
    struct_df: pd.DataFrame,
    node_ids: List[int]
) -> List[Tuple[str, str, float, float]]:
    """

    Args:
        struct_df: (pd.DataFrame) structure dataframe
        node_ids: (List[int]) a list of node ids for which to calculate phi and psi angles

    Returns:
        phi_psi: (List[Tuple[str, str, float, float]]) list of tuples, each tuple includes info for a single residue
            (residue_identifier (str), residue name (one-letter str), phi (float), psi (float))
    """
    # out vars
    phi_psi = []

    # add a column `residue_identifier`
    df = struct_df.copy()
    df["residue_identifier"] = [f"{c}{r}{a}" for (c, r, a) in struct_df[["chain", "resi", "alt"]].values]

    for ni in node_ids:
        _df = df[df.node_id == ni].drop_duplicates(["chain", "resi", "alt", "resn"])

        # residue identifier
        ri = _df["residue_identifier"].values[0]
        rn = _df["resn"].values[0]

        # compute phi and psi angles for specified residue identifiers
        phi, psi = calc_phi_psi_single_residue(struct_df=struct_df, node_id=ni)

        # append to result
        phi_psi.append((ri, rn, float(phi), float(psi)))

    return phi_psi


def calc_omega_single_residue(struct_df: pd.DataFrame, node_id: int) -> np.ndarray:
    """
    Calculate omega angle of the peptide between residue i (specified by node_id) and
    its preceding residue with node if `i - 1`

    Args:
        struct_df: (Structure DataFrame)
        node_id: (int) int index of node

    Returns:
        omega: (np.ndarray) 1-element np.ndarray
    """
    # get residues a, i
    ra = struct_df[struct_df.node_id == node_id - 1]
    ri = struct_df[struct_df.node_id == node_id]

    preceding_res_exist = True
    if ra.shape[0] == 0:
        logger.warning("Did not find preceding residue.")
        preceding_res_exist = False

    # omega: (i-1).CA - (i-1).C = i.N - i.CA
    omega = None
    if preceding_res_exist:
        # omega: i-1.CA  i.
        ra_CA = ra[ra.atom == "CA"][["x", "y", "z"]].to_numpy()
        ra_C = ra[ra.atom == "C"][["x", "y", "z"]].to_numpy()
        # get coord
        ri_N = ri[ri.atom == "N"][["x", "y", "z"]].to_numpy()
        ri_CA = ri[ri.atom == "CA"][["x", "y", "z"]].to_numpy()
        # omega
        omega = calc_dihedral(ra_CA, ra_C, ri_N, ri_CA)

    return omega


def calc_omega_set_residues(struct_df: pd.DataFrame,
                            node_ids: List[int]):
    """
    Calculate omega angles for a set of residues

    Args:
        struct_df: (pd.DataFrame) structure dataframe
        node_ids: (List[int]) a list of node ids for which to calculate omega angles

    Returns:
        omega: (List[Tuple[str, str, float]]) list of tuples, each tuple includes info for a single residue
            (residue_identifier (str), residue name (one-letter str), omega (float))
            e.g.
            - residue_identifier: "H100A"
            - residue name: "A"
            - omega: 179.45
    """
    # out vars
    omega_list = []

    # add a column `residue_identifier`
    df = struct_df.copy()
    df["residue_identifier"] = [f"{c}{r}{a}" for (c, r, a) in struct_df[["chain", "resi", "alt"]].values]

    for ni in node_ids:
        _df = df[df.node_id == ni].drop_duplicates(["chain", "resi", "alt", "resn"])

        # residue identifier
        ri = _df["residue_identifier"].values[0]
        rn = _df["resn"].values[0]

        # compute phi and psi angles for specified residue identifiers
        omega = calc_omega_single_residue(struct_df=struct_df, node_id=ni)

        # append to result
        omega_list.append((ri, rn, float(omega)))

    return omega_list