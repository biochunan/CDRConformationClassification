"""
Classify the CDR conformation of an input abdb entry into canonical classes 
using classifiers trained on unbound CDR conformations.
"""
# basic
import re
import json
import shutil 
import textwrap
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional, Any
from scipy.spatial.transform import Rotation

import joblib
import yaml
from cdrclass.examine_abdb_struct import (
    extract_atmseq_seqres, gen_struct_cdr_df,
    assert_HL_chains_exist, assert_cdr_no_missing_residues, assert_seqres_atmseq_length,
    assert_non_empty_file, assert_struct_file_exist, assert_cdr_no_big_b_factor,
    assert_cdr_no_non_proline_cis_peptide
)
import sys 
sys.path.append(Path(__file__).resolve().parent.as_posix())

# custom packages 
from cdrclass.utils import calc_omega_set_residues, calc_phi_psi_set_residues
from cdrclass.abdb import CDR_HASH_REV, extract_bb_atoms, parse_single_mar_file

# logging 
import sys
import logging
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s {%(pathname)s:%(lineno)d} [%(levelname)s] %(name)s - %(message)s [%(threadName)s]',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

CLUSTALO = shutil.which("clustalo")

# ==================== Function ====================
def process_single_mar_file(
    struct_fp: Path, 
    abdbid: str = None, 
    not_strict: bool = False, 
    resolution_thr: float = 2.8, 
    numbering_scheme: str = "ABM", 
    clustal_omega_exe_fp: Path = CLUSTALO, 
    b_factor_atom_set: List[str] = None, 
    b_factor_cdr_thr: float = 80.
):
    
    """
    Examine a single MAR file and return a dict of criteria

    Args:
        struct_fp (Path): path to MAR file
        abdbid (str, optional): the abdbid of the MAR file. 
            Defaults to None. this can be derived from the MAR file name
        not_strict (bool, optional):  
            If True, will continue to run even if any of the main stage checkpoints is not passed.
            If False, will break if any of the main stage checkpoints is not passed.
            Defaults to False.

    Returns:
        criteria: (Dict)a dict of criteria
        struct_df: (pd.DataFrame) a dataframe of the MAR file
    """
    if b_factor_atom_set is None:
        b_factor_atom_set = ["CA"]
    # criteria
    criteria = dict(
        mar_struct_exist=False,
        mar_struct_resolution=False,
        mar_struct_not_empty=False,
        struct_okay=False,
        chain_exist=False,
        chain_length_okay=False,
        cdr_no_missing_residue=False,
        cdr_no_big_b_factor=False,
        cdr_no_non_proline_cis_peptide=False,
    )
    if abdbid is None:
        abdbid = struct_fp.stem
    fc_type = re.search(r"[A-Za-z\d]{4}_\d+([A-Za-z]*)", abdbid)[1]
    struct_df = None

    # -------------------- assert mar file exists --------------------
    criteria["mar_struct_exist"] = assert_struct_file_exist(struct_fp=struct_fp)
    if not criteria["mar_struct_exist"]:
        logging.warning(f"{abdbid} mar file does not exist ...")

    # -------------------- check structure resolution --------------------
    if criteria["mar_struct_exist"]:
        # parse resolution
        with open(struct_fp, "r") as f:
            # get resolution from line 1 
            l = f.readline()
            if resolution := re.search(r", ([\d\.]+)A", l):
                resolution = float(resolution[1])
                criteria["mar_struct_resolution"] = resolution <= resolution_thr

        if not criteria["mar_struct_resolution"]:
            logging.warning(f"{abdbid} resolution greater than {resolution_thr} ...")

    # -------------------- assert not empty file --------------------
    if criteria["mar_struct_exist"]:
        criteria["mar_struct_not_empty"] = assert_non_empty_file(struct_fp=struct_fp)
        if not criteria["mar_struct_not_empty"]:
            logging.warning(f"{abdbid} mar file is empty ...")

    ckpt_file_pass = all(
        [
            criteria["mar_struct_exist"],
            criteria["mar_struct_resolution"],
            criteria["mar_struct_not_empty"],
        ]
    )
    # ----------------------------------------
    # II. checkpoint chain_pass
    # only check if ckpt_file_pass is True
    # ----------------------------------------
    ckpt_chain_pass = False
    # -------------------- assert mar structure is okay --------------------
    atmseq, seqres = {}, {}
    if ckpt_file_pass or not_strict:
        # get atmseq and seqres
        atmseq, seqres = extract_atmseq_seqres(struct_fp=struct_fp)

        # 1. check heavy and light chains
        criteria["chain_exist"] = assert_HL_chains_exist(struct_fp=struct_fp, atmseq=atmseq)
        if not criteria["chain_exist"]:
            logging.warning(f"{abdbid} chain ...")

        # 2. check seqres vs atmseq length
        if criteria["chain_exist"]:
            criteria["chain_length_okay"] = assert_seqres_atmseq_length(
                struct_fp=struct_fp,
                atmseq=atmseq,
                seqres=seqres)
            if not criteria["chain_length_okay"]:
                logging.warning(f"{abdbid} SEQRES vs ATMSEQ chain length ...")

        if all((criteria["chain_exist"], criteria["chain_length_okay"])):
            ckpt_chain_pass = True

    # ----------------------------------------
    # III. checkpoint cdr_pass
    # only check if ckpt_chain_pass is True
    # ----------------------------------------
    if ckpt_chain_pass or not_strict:
        # 3. check CDR no missing residues
        # parse abdb file
        df_H, df_L = gen_struct_cdr_df(
            struct_fp=struct_fp,
            cdr_dict=CDR_HASH_REV[numbering_scheme],
            concat=False,
            retain_b_factor=True,
            retain_hetatm=False,
            retain_water=False)
        criteria["cdr_no_missing_residue"] = assert_cdr_no_missing_residues(
            struct_fp=struct_fp,
            clustal_omega_executable=clustal_omega_exe_fp,
            numbering_scheme=numbering_scheme,
            atmseq=atmseq, seqres=seqres,
            df_H=df_H, df_L=df_L
        )
        if not criteria["cdr_no_missing_residue"]:
            logging.warning(f"{abdbid} CDR ...")

        # 4. check loop CA B-factor (filter out ≥ 80 & == 0.)
        if criteria["cdr_no_missing_residue"]:
            criteria["cdr_no_big_b_factor"] = assert_cdr_no_big_b_factor(
                struct_fp=struct_fp,
                b_factor_atoms=b_factor_atom_set,
                b_factor_thr=b_factor_cdr_thr,
                numbering_scheme=numbering_scheme,
                df_H=df_H, df_L=df_L
            )
            if not criteria["cdr_no_big_b_factor"]:
                logging.warning(f"{abdbid} Loop B factor ...")

        # concat Heavy and Light chain to single Structure DataFrame
        struct_df = pd.concat([df_H, df_L], axis=0, ignore_index=True)
        struct_df["node_id"][df_H.shape[0]:] += df_H.node_id.max() + 1  # correct node_id

        # 6. check non-Proline cis peptide i.e. -π/2 < ω < π/2
        # [x] [chunan]: only eliminate unbound abdb entries having non-proline cis-residues
        if criteria["cdr_no_missing_residue"]:
            # criteria["cdr_no_non_proline_cis_peptide"] = assert_cdr_no_non_proline_cis_peptide(
            cdr_no_non_proline_cis_peptide = assert_cdr_no_non_proline_cis_peptide(
                struct_fp=struct_fp,
                numbering_scheme=numbering_scheme,
                struct_df=struct_df
            )
            # pass or not depends on antibody type
            if fc_type == "":
                # Unbound antibody
                criteria["cdr_no_non_proline_cis_peptide"] = cdr_no_non_proline_cis_peptide
            else:
                # Bound antibodies
                # If found non-proline cis-residue issue warning but still pass
                criteria["cdr_no_non_proline_cis_peptide"] = True
                if not cdr_no_non_proline_cis_peptide:
                    logging.warning(f"{abdbid} (bound) CDR exists non-proline cis peptide.")
            # report
            if not criteria["cdr_no_non_proline_cis_peptide"]:
                logging.warning(f"{abdbid} (unbound) CDR exists non-proline cis peptide ...")

        # Finally, if all passed, set struct_okay=True
        if all((criteria["chain_exist"], criteria["mar_struct_resolution"],
                criteria["chain_length_okay"], criteria["cdr_no_missing_residue"],
                criteria["cdr_no_big_b_factor"], criteria["cdr_no_non_proline_cis_peptide"])):
            criteria["struct_okay"] = True

    return criteria, struct_df


# read cdr backbone atom coordinate csv
def read_cdr_bb_csv(fp: Path=None, df: pd.DataFrame=None, add_residue_identifier: bool = False) -> pd.DataFrame:
    """
    Extract CDR backbone atoms from a csv file or a dataframe
    At least provide either fp or df 

    Args:
        fp (Path, optional): file path to pre-processed backbone csv file. Defaults to None.
        df (pd.DataFrame, optional): parsed cdr backbones. Defaults to None.
        add_residue_identifier (bool, optional): add residue identifier format: [chain][resi][alt] e.g. "H100A". Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    # assert at least one of fp or df is provided 
    assert (fp is not None) or (df is not None)
    
    if fp is not None:
        df = pd.read_csv(fp)
    df.fillna("", inplace=True)

    # add residue identifier if True
    if add_residue_identifier:
        # create reisdue identifier
        ri_list = [f"{c}{r}{a}" for (c, r, a) in df[["chain", "resi", "alt"]].values]
        df["ri"] = ri_list

    return df


def extract_phi_psi_omega(struct_df: pd.DataFrame, add_residue_identifier: bool = True) -> pd.DataFrame:
    # CDR node_id list
    cdr_nodes: List[int] = struct_df[struct_df.cdr != ""].node_id.drop_duplicates().to_list()

    # calculate omega angles for cdr residues
    phi_psi_list: List[Tuple[str, str, float, float]] = calc_phi_psi_set_residues(struct_df=struct_df,
                                                                                  node_ids=cdr_nodes)
    omega_list: List[Tuple[str, str, float]] = calc_omega_set_residues(struct_df=struct_df,
                                                                       node_ids=cdr_nodes)
    # generate CDR Structure DataFrame
    cols = ["node_id", "chain", "resi", "alt", "resn", "atom", "b_factor", "cdr"]
    cdr_df = struct_df[(struct_df.cdr != "") & (struct_df.atom == "CA")][cols]
    dihedral_dict = {"node_id": [], "phi": [], "psi": [], "omega": []}
    for (i, phi_psi, omega) in zip(cdr_nodes, phi_psi_list, omega_list):
        dihedral_dict["node_id"].append(i)
        a, b = phi_psi[-2:]
        dihedral_dict["phi"].append(a)
        dihedral_dict["psi"].append(b)
        dihedral_dict["omega"].append(omega[-1])

    cdr_df = cdr_df.merge(pd.DataFrame(dihedral_dict), on="node_id").drop(columns=["node_id"])
    
    # add residue identifier if True
    if add_residue_identifier:
        # create reisdue identifier
        ri_list = [f"{c}{r}{a}" for (c, r, a) in cdr_df[["chain", "resi", "alt"]].values]
        cdr_df["ri"] = ri_list

    return cdr_df

# convert residue dihedral angles to trigonometric values for a single abdb file
def convert_one_loop_dihedral_to_trigonometric_array(
    dihedral_df: pd.DataFrame,
    loop_cdr_type: str,
    angle_type: Optional[List[str]] = None
) -> np.ndarray:
    """
    Args:
        ab_id: (str) abdb id
        angle_type: (List[str]) by default use phi psi omega angles
            the number of angle types determine the loop representation dimensionality
            e.g. if use all 3 angles, representation dim is sine cosine of 3 angles
                i.e. sin(phi) cos(phi) sin(psi) cos(psi) sin(omega) cos(omega)

    Returns:
        loop_repr: (np.ndarray) numpy array of loop representation.
            shape: [N, 2*m] where
                `N` is number of residues in a loop;
                `m` is number of angle types
    """
    # vars
    if angle_type is None:
        angle_type = ["phi", "psi", "omega"]
    n = len(angle_type)

    # extract dihedral angles for the cdr group
    arr_deg = dihedral_df[dihedral_df.cdr == loop_cdr_type][angle_type].to_numpy()
    arr_rad = arr_deg / 180. * np.pi  # to radian
    n_res = arr_rad.shape[0]
    loop_repr = np.empty((n_res, 2 * n))

    # populate values
    for i in range(n):
        loop_repr[:, 2 * i] = np.sin(arr_rad[:, i])
        loop_repr[:, 2 * i + 1] = np.cos(arr_rad[:, i])

    return loop_repr


# ---------------------------------------------
# func required for 
# `merge_in_torsional` and `merge_in_cartesian`
# ---------------------------------------------
# find CB residue identifiers
def cb_ri(df: pd.DataFrame, cdr: str) -> List[str]:
    """
    Args:
        df: (pd.DataFrame) CDR loop structure dataframe
        cdr: (str) CDR identifier either "L1", "L2", "L3", "H1", "H2", "H3"

    Returns:
        ri_list: (List[str]) List of residue identifiers with CB atom present e.g. ['L50']
    """
    assert "ri" in df.columns
    ri_list = df[(df.cdr == cdr) & (df.atom == "CB")]["ri"].drop_duplicates().to_list()

    return ri_list


# extract non-GLY position CB atom coordinates
def extract_cb_atoms(df: pd.DataFrame, cdr: str, ri_list: List[str]) -> np.ndarray:
    """
    Extract CB atom coordinates from a structure dataframe, excluding glycine residues specified in `gly_ri_list`

    Args:
        df: (pd.DataFrame) CDR loop structure dataframedf: (pd.DataFrame)
        gly_ri_list: (List[str]) A list of residue identifiers for Glycines found in the structure DataFrame

    Returns:
        cb_coord: (np.ndarray) CB atom coordinates
    """
    assert "ri" in df.columns

    return df[(df.cdr == cdr) & (df.atom == "CB") & (df.ri.isin(ri_list))][["x", "y", "z"]].to_numpy()


# extract non-GLY position CB atom coordinates
def extract_ca_atoms(df: pd.DataFrame, cdr: str) -> np.ndarray:
    """
    Extract CA atom coordinates from a structure dataframe

    Args:
        df: (pd.DataFrame) CDR loop structure dataframedf: (pd.DataFrame)

    Returns:
        ca_coord: (np.ndarray) CA atom coordinates
    """
    assert "ri" in df.columns

    return df[(df.cdr == cdr) & (df.atom == "CA")][["x", "y", "z"]].to_numpy()


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


# --------------------
# merge in torsion 
# and cartesian
# --------------------
# determine whether merge bound vs. unbound cluster using AP cluster radius
def merge_in_torsional(emb_a: np.ndarray, emb_b: np.ndarray, clu_radius: float, verbose: bool = True) -> bool:
    """
    Returns:
        bool: whether a pair of clusters should be merged
    """
    # flatten size
    emb_a = emb_a.flatten()
    emb_b = emb_b.flatten()

    # calculate distance (squared Euclidean distance)
    distance = float(np.linalg.norm(emb_a - emb_b) ** 2)

    # merge or not
    merge_clu: bool = distance < clu_radius

    # report str
    logging.info(f"\nSummary:\n"
                f"Distance to predicted cluster exemplar (D): {distance}\n"
                f"Predicted cluster radius (R): {clu_radius}\n"
                f"Merge in torsional space (True if D < R): {merge_clu}\n")

    return merge_clu


# determine whether merge bound vs. unbound cluster using Cartesian criteria from (Martin & Thornton, 1996)
def merge_in_cartesian(bb_df_a: pd.DataFrame, bb_df_b: pd.DataFrame, cdr: str, verbose: bool = True) -> bool:
    """
    Returns:
        bool: whether a pair of clusters should be merged
    """
    # extract CA atoms
    xyz_a_ca = extract_ca_atoms(df=bb_df_a, cdr=cdr)
    xyz_b_ca = extract_ca_atoms(df=bb_df_b, cdr=cdr)

    # extract CB atoms
    ri_list = list(set(cb_ri(df=bb_df_a, cdr=cdr)).intersection(set(cb_ri(df=bb_df_b, cdr=cdr))))
    xyz_a_cb = extract_cb_atoms(df=bb_df_a, cdr=cdr, ri_list=ri_list)
    xyz_b_cb = extract_cb_atoms(df=bb_df_b, cdr=cdr, ri_list=ri_list)

    # translation vector
    translation_a = xyz_a_ca.mean(axis=0)
    translation_b = xyz_b_ca.mean(axis=0)

    # fit on CA coordinates using Kabsch algorithm
    rot_b2a, ca_rmsd = superimpose_atoms(ref=xyz_a_ca, mob=xyz_b_ca)

    # apply translation and rotation
    xyz_a_ca = (xyz_a_ca - translation_a)
    xyz_b_ca = np.einsum("ic,bc->bi", rot_b2a, xyz_b_ca - translation_b)
    xyz_a_cb = (xyz_a_cb - translation_a)
    xyz_b_cb = np.einsum("ic,bc->bi", rot_b2a, xyz_b_cb - translation_b)

    # atom-wise distance
    ca_dists = atom_wise_dist(xyz_a=xyz_a_ca, xyz_b=xyz_b_ca)
    cb_dists = atom_wise_dist(xyz_a=xyz_a_cb, xyz_b=xyz_b_cb)

    if verbose:
        ca_dist_str = ", ".join([f"{i:.2f}" for i in ca_dists.flatten()])
        cb_dist_str = ", ".join([f"{i:.2f}" for i in cb_dists.flatten()])
        logging.info(
            # trans & rot matrix
            f"\n"
            f"Translation vector:\n"
            f"a: {translation_a}\n"
            f"b: {translation_b}\n"
            f"Rotation matrix:\n{rot_b2a}\n"
            # atom-wise distance
            f"Atom-wise distance after translation and rotation:\n"
            f"CA: {ca_dist_str}\n"
            f"CB: {cb_dist_str}\n"
        )

    _b1, _b2, _b3 = ca_rmsd < 1.0, ca_dists.max() < 1.5, cb_dists.max() < 1.9
    merge_clu = all((_b1, _b2, _b3))

    # report str
    logging.info(f"\nSummary:\n"
                f"CA RMSD (< 1.0 Å): {_b1} ({ca_rmsd:.2f} Å)\n"
                f"CA pairwise distance maximum (< 1.5 Å): {_b2} ({ca_dists.max():.2f} Å)\n"
                f"CB pairwise distance maximum (< 1.9 Å): {_b3} ({cb_dists.max():.2f} Å)\n"
                f"Merge in Cartesian space: {merge_clu}\n")

    return merge_clu


# fetch LRC, AP cluster, and Canonical cluster from the LRC_AP_cluster.json file
def fetch_lrc_ap_can(lrc_ap_cluster: Dict[str, Any], lrc_name: str, ap_clu_idx: int) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Fetch LRC, AP cluster, and Canonical cluster from the LRC_AP_cluster.json file

    Args:
        lrc_ap_cluster (Dict[str, Any]): parsed LRC_AP_cluster.json content 
        lrc_name (str): e.g. "H1-10-allT"
        ap_clu_idx (int): e.g. 4

    Returns:
        Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]: LRC, AP cluster, and Canonical cluster
            each is a dictionary object
    """
    # 1. find the LRC group 
    lrc = next(
        (i for i in lrc_ap_cluster if i["LRC"] == lrc_name),
        None 
    )
    # 2. find the AP cluster with the specified ap_clu_idx
    ap_clu = next(
        (
            ap
            for ap in lrc['AP_clusters']
            if ap['ap_clu_label'] == ap_clu_idx
        ),
        None 
    )
    # ap cluster exemplar name e.g. "1a4k_0"
    ap_name = ap_clu['ap_clu_cen_abdbid']
    
    # find canonical cluster that has the AP cluster specified by `ap_name`
    can_clu = next(
        (
            i
            for i in lrc['Canonicals']['multi_ap_clu_canonicals'] + lrc['Canonicals']['single_ap_clu_canonicals']
            if any(j.startswith(ap_name) for j in i['ap_clu_centers'])
        ),
        None,
    )
    return lrc, ap_clu, can_clu


# --------------------
# wrappers
# --------------------
def process_cdr(
    cdr: str, 
    dihedral_df: pd.DataFrame, 
    bb_df: pd.DataFrame,
    abdbid: str,
    classifier_dir: Path,
    lrc_ap_cluster: Dict[str, Any],
) -> Dict[str, Any]:
    # ----------------------------------------
    # prepare processing for each CDR type 
    # ----------------------------------------
    # extract column cdr == `cdr`
    dihedral_df = dihedral_df[dihedral_df.cdr == cdr].copy()
    bb_df = bb_df[bb_df.cdr == cdr].copy()

    # load classifiers of the same CDR type and CDR length e.g. H1-12
    cdr_len = dihedral_df.shape[0]

    # load classifiers of the same CDR type and CDR length e.g. H1-12 
    clf_fps = list(classifier_dir.glob(f"{cdr}-{cdr_len}-*-FreeAb.joblib"))
    clfs = {
        fp.name.split("-FreeAb")[0]: joblib.load(filename=fp)
        for fp in clf_fps 
    }

    # ===== 1. create a warning if non-proline cis-residues are found =====
    non_pro_cis_res = dihedral_df[
        (dihedral_df.omega > -90.0)
        & (dihedral_df.omega < 90.0)
        & (dihedral_df.resn != "P")
    ]

    if non_pro_cis_res.shape[0] > 0:
        report_str = "cis-peptide thr: -90 < ω < 90\n" \
                        "residue    cdr    aa     omega(degree)\n"
        for (ri, aa, omega) in non_pro_cis_res[["ri", "resn", "omega"]].values:
            report_str += f"{ri:>7}  {cdr:>5}  {aa:>4}  {omega:>17.2f}\n"
        logging.warning(f"Found following non-proline cis-residues in Bound structure:\n"
                        f"{report_str}")

    # ===== 2. Found closest AP cluster =====
    # generate loop representation
    loop_repr = convert_one_loop_dihedral_to_trigonometric_array(
        dihedral_df=dihedral_df,
        loop_cdr_type=cdr,
        angle_type=["phi", "psi"]
    )  # => shape (L, 4)
    X = loop_repr.reshape(1, -1)

    # ===== 3. find the closest LRC/CAN/AP to the query loop =====
    clf_name, dists, cluster_idx_list = [], [], []
    for k, clf in clfs.items():
        # find the closest exemplar to the query loop
        cluster_idx: int = clf.predict(X)[0]
        closest_exemplar_arr = clf.cluster_centers_[cluster_idx]
        # append to list
        clf_name.append(k)
        dists.append(np.linalg.norm(X - closest_exemplar_arr))
        cluster_idx_list.append(cluster_idx)

    # find the closest cluster
    closest_clf = clf_name[np.argmin(dists)]
    closest_ap_clu_idx = cluster_idx_list[np.argmin(dists)]
    closest_exemplar_arr = clfs[closest_clf].cluster_centers_[closest_ap_clu_idx]

    # get AP and CAN cluster
    lrc, closest_ap, closest_can = fetch_lrc_ap_can(
        lrc_ap_cluster=lrc_ap_cluster, 
        lrc_name=closest_clf, 
        ap_clu_idx=closest_ap_clu_idx
    )

    closest_ap_lab, closest_ap_size, closest_exemplar_id, closest_can_clu_idx =\
        closest_ap['ap_clu_label'], closest_ap['ap_clu_size'], closest_ap['ap_clu_cen_abdbid'], \
            closest_can['canonical_idx_global']
    clu_radius = closest_ap['ap_clu_radius']

    # ===== 4. Compare query and closest cluster in Torsional/Cartesian space =====
    """
    4.1 Compare in torsional space
    - if merged => return
    - otherwise => compare in Cartesian space with all exemplars in the same LRC group
    4.2 (if not merged) Compare in Cartesian space
    - if merged => return 
    """

    merged: bool = False
    merged_exemplar_id: Optional[str] = None
    merge_with_closest_exemplar_in_torsional: Optional[bool] = None
    merge_with_any_exemplar_in_cartesian: Optional[bool] = None

    # 4.1 compare in torsional space against the closest exemplar in the same LRC group
    # clu_radius = lrc_exemplar_df[lrc_exemplar_df.clu_center_ab_id == closest_exemplar_id]["clu_radius"].values[0]
    merge_with_closest_exemplar_in_torsional = merge_in_torsional(
        emb_a=closest_exemplar_arr,
        emb_b=X,
        clu_radius=clu_radius
    )
    if merge_with_closest_exemplar_in_torsional:
        merged = True
        merged_exemplar_id = closest_exemplar_id
        logging.info(
            f"Merge bound cdr loop `{abdbid}` with exemplar `{merged_exemplar_id}` in torsional space"
        )
    # 4.2 if not merged in torsional space compare in Cartesian space against all exemplars
    merged_ap_clu_label: Optional[int] = None
    # if not merge_with_closest_exemplar_in_torsional:
    if not merge_with_closest_exemplar_in_torsional:
        # 4.2.1 retrieve all exemplars in the same LRC group and compare
        for ap in lrc['AP_clusters']:
            exemplar_id = ap['ap_clu_cen_abdbid']
            # exemplar bb_df
            exemplar_bb_df = read_cdr_bb_csv(
                df=extract_bb_atoms(struct_df=parse_single_mar_file(ABDB.joinpath(f"pdb{exemplar_id}.mar"))),
                add_residue_identifier=True
            )
            if merge_with_any_exemplar_in_cartesian := merge_in_cartesian(
                bb_df_a=exemplar_bb_df, 
                bb_df_b=bb_df, 
                cdr=cdr
            ):
                merged = True
                merged_ap_clu_label = ap['ap_clu_label']
                break

    # ========== 5. Report ==========
    #  metadata: exemplar AP cluster merged into
    merged_ap_lab, merged_exemplar_id, merged_ap_size, merged_can_clu_idx = \
        None, None, None, None
    if merged_ap_clu_label is not None:
        # find the ap cluster whose `ap_clu_cen_abdbid` === `merged_exemplar_id` 
        _, merged_ap, merged_can = fetch_lrc_ap_can(
            lrc_ap_cluster=lrc_ap_cluster, 
            lrc_name=closest_clf, 
            ap_clu_idx=merged_ap_clu_label
        )
        merged_ap_lab, merged_exemplar_id, merged_ap_size, merged_can_clu_idx = \
            merged_ap['ap_clu_label'], merged_ap['ap_clu_cen_abdbid'], merged_ap['ap_clu_size'], \
                merged_can['canonical_idx_global']

    # report
    report_str = textwrap.dedent(f"""
    ----------   query cdr info ----------
    Abdb entry: {abdbid}
    Closest LRC group, CDR cluster and family info:
    CDR            can_clu    cluster_label    cluster_exemplar_id    cluster_size    canonical_cluster_index 
    {cdr:>3}    {closest_clf:>15}    {closest_ap_lab:>13}    {closest_exemplar_id:>19}    {closest_ap_size:>12}    {closest_can_clu_idx:>5}
    ----------   merge info     ----------
    Could merge with closest exemplar in torsional space?: {merge_with_closest_exemplar_in_torsional}
    """)

    if not merge_with_closest_exemplar_in_torsional:
        report_str += textwrap.dedent(
            f"""
        Could merge with any exemplar within the same LRC group in Cartesian space?: {merge_with_any_exemplar_in_cartesian}
        Merged exemplar abdbid (None if not merged): {merged_exemplar_id}
        Merged canonical cluster index (None if not merged): {merged_can_clu_idx}
        """)

    logging.info(report_str)



    return {
        # closest AP/CAN cluster summary (CAN: canonical, AP: Affinity Propagation)
        "closest_lrc": closest_clf,                                # LRC group         (str) e.g. "H1-10-allT"
        "closest_AP_cluster_label": closest_ap_lab,               # AP  cluster       (int) e.g. 1
        "closest_AP_cluster_exemplar_id": closest_exemplar_id,     # AP  cluster       (str) e.g. "6azk_0"
        "closest_AP_cluster_size": closest_ap_size,               # AP  cluster       (int) e.g. 93 
        "closest_can_cluster_index": closest_can_clu_idx,          # CAN cluster index (int) e.g. 4
        # merged AP/CAN cluster summary
        "merged_AP_cluster_label": merged_ap_lab,                 # AP  cluster
        "merged_AP_cluster_exemplar_id": merged_exemplar_id,       # AP  cluster
        "merged_AP_cluster_size": merged_ap_size,                 # AP  cluster
        "merged_can_cluster_index": merged_can_clu_idx,            # CAN cluster index 
        # merging summary 
        "merge_with_closest_exemplar_torsional": bool(merge_with_closest_exemplar_in_torsional),
        "merge_with_any_exemplar_cartesian": merge_with_any_exemplar_in_cartesian,
        "merged": merged,
    }


def worker(
    abdbid: str, 
    cdr: str = None, 
    abdb_dir: Path = None, 
    classifier_dir: Path = None, 
    lrc_ap_cluster: Dict[str, Any] = None):
    if cdr is None:
        cdr = "all"
    # ----------------------------------------
    # parse and validate the struct
    # ----------------------------------------
    criteria, struct_df = process_single_mar_file(
        struct_fp=abdb_dir.joinpath(f"pdb{abdbid}.mar"),
        abdbid=abdbid,
        not_strict=True,
    )
    if not criteria["cdr_no_missing_residue"]:
        logging.warning(f"CDR {cdr} of {abdbid} has missing residue(s).")

    # ----------------------------------------
    # extract dihedrals and backbone atoms
    # ----------------------------------------
    # extract dihedrals and backbone atoms 
    dihedral_df = extract_phi_psi_omega(struct_df=struct_df)
    bb_df = extract_bb_atoms(struct_df=struct_df, add_residue_identifier=True)

    if cdr != "all":
        logging.info(textwrap.dedent(f"""
            ------------------------------
            Processing {abdbid}, CDR-{cdr}
            ------------------------------
        """))
        return process_cdr(
            cdr=cdr,
            dihedral_df=dihedral_df,
            bb_df=bb_df,
            abdbid=abdbid,
            classifier_dir=classifier_dir,
            lrc_ap_cluster=lrc_ap_cluster
        )
    results = []
    for cdr in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        logging.info(textwrap.dedent(f"""
            ------------------------------
            Processing {abdbid}, CDR-{cdr}
            ------------------------------
        """))
        results.append( {
            cdr: process_cdr(
                cdr=cdr,
                dihedral_df=dihedral_df,
                bb_df=bb_df,
                abdbid=abdbid,
                classifier_dir=classifier_dir,
                lrc_ap_cluster=lrc_ap_cluster
            )
        })
        logging.info(f"Processing {abdbid}, CDR-{cdr} Done.")
        print("\n\n\n")
    return results


# --------------------
# Main
# --------------------
def cli():  # sourcery skip: inline-immediately-returned-variable
    parser = argparse.ArgumentParser()
    parser.add_argument("abdbid", type=str, help="input file path")
    # cdr
    parser.add_argument("--cdr", type=str, help="CDR type", default="all",
                        choices=["H1", "H2", "H3", "L1", "L2", "L3", "all"])
    # output 
    parser.add_argument("--outdir", help="results directory", default=None)
    # folders - abdb, classifier, exemplar, bb read from config file
    parser.add_argument("--config", type=str, help="config file path", required=True)
    
    # parse args
    args = parser.parse_args()
    
    # parse config file yaml 
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    return args, config 

def main(args):
    args, cfg = cli()
    ABDB           = Path(cfg["ABDB"]).expanduser()
    classifier_dir = Path(cfg["classifier"])
    LRC_AP_fp      = Path(cfg["LRC_AP_cluster"])
    outdir = Path.cwd().joinpath("results") if args.outdir is None else Path(args.outdir)
        
    abdbid = args.abdbid
    cdr = args.cdr
    # mkdir
    outdir.mkdir(parents=True, exist_ok=True)
    
    # parse the LRC_AP_cluster.json
    with open(LRC_AP_fp, "r") as f:
        LRC_AP_CLUSTER = json.load(f)
    
    # run worker
    results = worker(
        abdbid=abdbid, 
        cdr=cdr, 
        abdb_dir=ABDB, 
        classifier_dir=classifier_dir, 
        lrc_ap_cluster=LRC_AP_CLUSTER
    )
    
    # save results
    with open(outdir/f"{abdbid}.json", "w") as f:
        json.dump(results, f, indent=4)

def app(): 
    args = cli() 
    main(args)

# ==================== Main ====================
if __name__ == "__main__":
    app()