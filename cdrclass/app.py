"""
Classify the CDR conformation of an input abdb entry into canonical classes 
using classifiers trained on unbound CDR conformations.
"""
# basic
import re
import json
import gdown
import shutil 
import joblib
import textwrap
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

# custom packages 
from cdrclass.examine_abdb_struct import (
    extract_atmseq_seqres, 
    gen_struct_cdr_df,
    assert_chain_type_exist,
    assert_non_empty_file, 
    assert_struct_file_exist, 
    assert_seqres_atmseq_length,
    assert_no_cdr_missing_residues, 
    assert_cdr_no_big_b_factor,
    assert_cdr_no_non_proline_cis_peptide
)
from cdrclass.geometry import (
    extract_ca_atoms, extract_cb_atoms, cb_ri, 
    superimpose_atoms, atom_wise_dist
)
from cdrclass.utils import calc_omega_set_residues, calc_phi_psi_set_residues
from cdrclass.exceptions import *
from cdrclass.abdb import CDR_HASH_REV, extract_bb_atoms, get_resolution_from_abdb_file, get_chain_map_from_remark_950

# logger 
from loguru import logger
from rich.logging import RichHandler
logger.configure(
    handlers=[
        {"sink": RichHandler(rich_tracebacks=True), "format": "{message}"}
    ]
)


# ==================== Constants ====================
CLUSTALO = shutil.which("clustalo")
BASE = Path(__file__).resolve().parent  # cdrclass/
ASSETS_URL = 'https://drive.google.com/uc?id=1kERH5wYVMhCvmAlDPL835Ms4ZhxL0pQQ'
ABDB=None

# ==================== Installation ====================
def is_first_run() -> bool:
    return not BASE.joinpath("assets", "classifier").exists()

def download_and_extract_classifier() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # download classifier
        o = Path(tmpdir).joinpath('assets.tar.gz').as_posix()
        gdown.download(ASSETS_URL, output=o, quiet=False)
        # extract classifier
        (BASE/'assets').mkdir(exist_ok=True, parents=True)
        shutil.unpack_archive(filename=o, extract_dir=BASE/'assets')


# ==================== Function ====================
def concat_chain_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    # iterate over dfs and add max node_id from previous df to the current df node_id
    dfo = None 
    for dfi in dfs: 
        df = dfi.copy()
        if dfo is None: 
            dfo = dfi.copy()
        else: 
            df["node_id"] += dfo.node_id.max() + 1
            dfo = pd.concat([dfo, df], axis=0, ignore_index=True)
    return dfo

def process_single_mar_file(
    struct_fp: Path, 
    abdbid: str = None, 
    strict: bool = True, 
    resolution_thr: float = 2.8, 
    numbering_scheme: str = "ABM", 
    clustal_omega_exe_fp: Path = CLUSTALO, 
    b_factor_atom_set: List[str] = None, 
    b_factor_cdr_thr: float = 80.,
    chain_types: List[str] = None,
):

    """
    Examine a single MAR file and return a dict of criteria

    Args:
        struct_fp (Path): path to MAR file
        abdbid (str, optional): the abdbid of the MAR file. 
            Defaults to None. this can be derived from the MAR file name
        strict (bool, optional):  
            If True, will break if any of the main stage checkpoints is not passed.
            If False, will continue to run even if any of the main stage checkpoints is not passed.
            Defaults to True.

    Returns:
        criteria: (Dict)a dict of criteria
        struct_df: (pd.DataFrame) a dataframe of the MAR file
    """
    b_factor_atom_set = b_factor_atom_set or ["CA"]
    abdbid = abdbid or struct_fp.stem
    fc_type = re.search(r"[A-Za-z\d]{4}_\d+([A-Za-z]*)", abdbid)[1]
    chain_types = chain_types or ["H", "L"]
    # checkpoints:
    # NOTE: if strict, all checkpoints must be passed 
    criteria = dict(mar_struct_exist=False,
                    mar_struct_resolution=False,
                    mar_struct_not_empty=False,
                    struct_okay=False,
                    chain_exist=False,
                    chain_length_okay=False,
                    cdr_no_missing_residue=False,
                    cdr_no_big_b_factor=False,
                    cdr_no_non_proline_cis_peptide=False)
    metadata = dict(cdr_missing_residue={},
                    cdr_no_big_b_factor={},
                    cdr_no_non_proline_cis_peptide={})

    # -------------------- assert mar file exists --------------------
    criteria["mar_struct_exist"] = assert_struct_file_exist(struct_fp=struct_fp)
    if not criteria["mar_struct_exist"]:
        logger.critical(f"{abdbid} mar file does not exist ...")
        exit(1)

    # -------------------- assert not empty file --------------------
    criteria["mar_struct_not_empty"] = assert_non_empty_file(struct_fp=struct_fp)
    if not criteria["mar_struct_not_empty"]:
        logger.critical(f"{abdbid} mar file is empty ...")
        exit(2)

    # -------------------- check structure resolution --------------------
    r = get_resolution_from_abdb_file(abdb_struct_fp=struct_fp)
    metadata['resolution'] = r
    criteria["mar_struct_resolution"] = r <= resolution_thr
    if not criteria["mar_struct_resolution"]:
        logger.warning(f"{abdbid} resolution greater than {resolution_thr} ...")

    ckpt_file_pass = all(
        [
            criteria["mar_struct_exist"],
            criteria["mar_struct_not_empty"],
            criteria["mar_struct_resolution"],
        ]
    )
    # ----------------------------------------
    # II. checkpoint chain_pass
    # only check if ckpt_file_pass is True
    # ----------------------------------------
    ckpt_chain_pass = False
    # -------------------- assert mar structure is okay --------------------
    atmseq, seqres = {}, {}
    if ckpt_file_pass or not strict:
        # get atmseq and seqres
        atmseq, seqres = extract_atmseq_seqres(struct_fp=struct_fp)

        # get chain map from abdb file 
        chain_map = get_chain_map_from_remark_950(abdb_fp=struct_fp)
        ab_chain_labels = chain_map.query('chain_type in ["H", "L"]').chain_label.to_list()
        atmseq = {k: v for k, v in atmseq.items() if k in ab_chain_labels}
        seqres = {k: v for k, v in seqres.items() if k in ab_chain_labels}

        # 1. check chain_types exist 
        metadata['chain_type_exists'] = {}
        for chain_type in chain_types:
            b = assert_chain_type_exist(struct_fp=struct_fp, chain_type=chain_type)
            if not b: 
                logger.warning(f"{abdbid} chain {chain_type=} not exist ...")
            criteria["chain_exist"] = criteria["chain_exist"] and b
            metadata['chain_type_exists'][chain_type] = b

        # 2. check seqres vs atmseq length
        criteria["chain_length_okay"] = assert_seqres_atmseq_length(struct_fp=struct_fp,
                                                                    atmseq=atmseq,
                                                                    seqres=seqres)
        if not criteria["chain_length_okay"]:
            logger.warning(f"{abdbid} SEQRES vs ATMSEQ chain length ...")

        if all((criteria["chain_exist"], criteria["chain_length_okay"])):
            ckpt_chain_pass = True

    # ----------------------------------------
    # III. checkpoint cdr_pass
    # only check if ckpt_chain_pass is True
    # ----------------------------------------
    if ckpt_chain_pass or not strict:
        # 3. check CDR no missing residues
        # parse abdb file
        df_H, df_L = gen_struct_cdr_df(
            struct_fp=struct_fp,
            cdr_dict=CDR_HASH_REV[numbering_scheme],
            concat=False,
            retain_b_factor=True,
            retain_hetatm=False,
            retain_water=False)
        criteria["cdr_no_missing_residue"] = all(
            assert_no_cdr_missing_residues(struct_fp=struct_fp,
                                           struct_df=d['df'],
                                           chain_type=d['chain_type'],
                                           chain_label=d['chain_label'],
                                           atmseq=atmseq[d['chain_label']],
                                           seqres=seqres[d['chain_label']],
                                           clustal_omega_executable=clustal_omega_exe_fp,
                                           numbering_scheme=numbering_scheme)
            for d in df_H + df_L
        )
        if not criteria["cdr_no_missing_residue"]:
            logger.warning(f"{abdbid} CDR ...")

        # 4. check loop CA B-factor (filter out ≥ 80 & == 0.)
        if criteria["cdr_no_missing_residue"]:
            criteria["cdr_no_big_b_factor"] = all(
                assert_cdr_no_big_b_factor(struct_fp=struct_fp,
                                           struct_df=d['df'],
                                           b_factor_atoms=b_factor_atom_set,
                                           b_factor_thr=b_factor_cdr_thr,
                                           numbering_scheme=numbering_scheme)
                for d in df_H + df_L
            )
            if not criteria["cdr_no_big_b_factor"]:
                logger.warning(f"{abdbid} Loop B factor ...")

        # concat chains to a single Structure DataFrame
        struct_df = concat_chain_dfs(dfs=[d['df'] for d in df_H + df_L])

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
                    logger.warning(f"{abdbid} (bound) CDR exists non-proline cis peptide, but still pass ...")
                    criteria['metadata'] = {'cdr non proline cis peptide': 'found but still pass'}
            # report
            if not criteria["cdr_no_non_proline_cis_peptide"]:
                logger.warning(f"{abdbid} (unbound) CDR exists non-proline cis peptide ...")

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
        # create residue identifier
        ri_list = [f"{c}{r}{a}" for (c, r, a) in df[["chain", "resi", "alt"]].values]
        df["ri"] = ri_list

    return df

def extract_phi_psi_omega(struct_df: pd.DataFrame, add_residue_identifier: bool = True) -> pd.DataFrame:
    """
    Extract phi, psi, omega angles from a structure dataframe

    Args:
        struct_df (pd.DataFrame): structure dataframe
        add_residue_identifier (bool, optional): add residue identifier using chain_label, residue_number, insertion. Defaults to True.
        --------------------------
        name           | col name 
        --------------------------
        chain_label    | chain
        residue_number | resi 
        insertion      | alt
        --------------------------

    Returns:
        pd.DataFrame: _description_
    """
    assert 'cdr' in struct_df.columns
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
        # create residue identifier
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
    logger.info(f"\nSummary:\n"
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
    xyz_a_ca = extract_ca_atoms(struct_df=bb_df_a, cdr=cdr)
    xyz_b_ca = extract_ca_atoms(struct_df=bb_df_b, cdr=cdr)

    def _find_equivalent_cdr_atom(dfi1: pd.DataFrame, dfi2: pd.DataFrame, cdr:str, atom: str) -> Tuple[List[str], List[str]]:
        """
        Find equivalent CDR atoms between two dataframes

        Args:
            dfi1 (pd.DataFrame): input structure dataframe 1 
            dfi2 (pd.DataFrame): input structure dataframe 2
            cdr (str): CDR type e.g. 'L1', 'L2', 'L3', 'H1', 'H2', 'H3' 
            atom (List[str]): name of atom e.g. 'CA', 'CB'

        Returns:
            Tuple[List[str], List[str]]: a tuple of lists of equivalent CDR atoms
            e.g. 
            (['L50', 'L51', 'L52', 'L53', 'L54', 'L55', 'L56'], 
             ['l50', 'l51', 'l52', 'l53', 'l54', 'l55', 'l56'])
             Note the order of the lists are the same as df1, df2 
        """
        assert 'ri' in dfi1.columns and 'ri' in dfi2.columns
        df1, df2 = dfi1.copy(), dfi2.copy()
        df1['_ri'], df2['_ri'] = df1.ri.apply(str.upper), df2.ri.apply(str.upper)
        df1 = df1.query('cdr==@cdr and atom==@atom')
        df2 = df2.query('cdr==@cdr and atom==@atom')
        # find equivalent _ri 
        cols=['ri', '_ri']
        dfm = pd.merge(df1[cols], df2[cols], on='_ri', how='inner', suffixes=('_a', '_b'))
        return dfm.ri_a.to_list(), dfm.ri_b.to_list()
    
    # extract CB atoms
    ri_list_a, ri_list_b = _find_equivalent_cdr_atom(dfi1=bb_df_a, dfi2=bb_df_b, cdr=cdr, atom='CB')
    xyz_a_cb = extract_cb_atoms(struct_df=bb_df_a, cdr=cdr, ri_list=ri_list_a)
    xyz_b_cb = extract_cb_atoms(struct_df=bb_df_b, cdr=cdr, ri_list=ri_list_b)

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
        logger.info(
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
    logger.info(f"\nSummary:\n"
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


# --------------------
# Catch error decorators  
# --------------------
def catch_error_from_process_cdr(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ClassifierNotExistError as e:
            logger.error(f"{func.__name__} raised an exception: {e}")
            return {'ClassifierNotExistError': str(e)}
        except NaNDihedralError as e:
            logger.error(f"{func.__name__} raised an exception: {e}")
            return {'NaNDihedralError': str(e)}
    return wrapper

# --------------------
# wrappers
# --------------------
@catch_error_from_process_cdr
def process_cdr(
    cdr: str, 
    chain_label: str,
    abdb_dir: Path,
    dihedral_df: pd.DataFrame, 
    bb_df: pd.DataFrame,
    abdbid: str,
    classifier_dir: Path,
    lrc_ap_cluster: Dict[str, Any],
) -> Dict[str, Any]:
    # ----------------------------------------
    # prepare processing for each CDR type 
    # ----------------------------------------
    # # extract column cdr == `cdr`
    # dihedral_df = dihedral_df[dihedral_df.cdr == cdr].copy()
    # bb_df = bb_df[bb_df.cdr == cdr].copy()
    dihedral_df = dihedral_df.query('cdr==@cdr and chain==@chain_label').reset_index(drop=True)
    bb_df = bb_df.query('cdr==@cdr and chain==@chain_label').reset_index(drop=True)

    # load classifiers of the same CDR type and CDR length e.g. H1-12
    cdr_len = dihedral_df.shape[0]

    # load classifiers of the same CDR type and CDR length e.g. H1-12 
    # ===== 0. retrieve pre-calculated classifiers =====
    clf_fps = list(classifier_dir.glob(f"{cdr}-{cdr_len}-*-FreeAb.joblib"))
    try:
        assert clf_fps
    except AssertionError as e:
        raise ClassifierNotExistError(f"Cannot find classifier for {cdr=} and {cdr_len=}") from e

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
        logger.warning(f"Found following non-proline cis-residues in the structure:\n"
                        f"{report_str}")

    # ===== 2. Found closest AP cluster =====
    # generate loop representation
    loop_repr = convert_one_loop_dihedral_to_trigonometric_array(
        dihedral_df=dihedral_df,
        loop_cdr_type=cdr,
        angle_type=["phi", "psi"]
    )  # => shape (L, 4)
    X = loop_repr.reshape(1, -1)
    # if np.nan in X, raise NaNDihedralError error 
    if np.isnan(X).any():
        logger.error(f"Found NaN dihedral angles in {abdbid=}, {cdr=}")
        logger.error(f'Dihedral DataFrame:\n{dihedral_df}')
        raise NaNDihedralError(f"{abdbid=}, {cdr=}, Found NaN dihedral angles")

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
        logger.info(
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
                # df=extract_bb_atoms(struct_df=parse_single_mar_file(struct_fp=abdb_dir.joinpath(f"pdb{exemplar_id}.mar"))),
                df=extract_bb_atoms(struct_df=gen_struct_cdr_df(struct_fp=abdb_dir/f"pdb{exemplar_id}.mar",
                                                                cdr_dict=CDR_HASH_REV['ABM'],
                                                                concat=True,
                                                                retain_b_factor=True)),
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

    logger.info(report_str)

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
    lrc_ap_cluster: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    The core function that compare the input AbDb antibody with pre-calculated 
    Length and Residue Configuration (LRC) Affinity Propagation (AP) clusters
    and return the closest cluster information for specified CDR type.
    
    Args:
        abdbid (str): AbDb entry id e.g. "1ikf_0P"
        cdr (str, optional): CDR type e.g. "H1", "H2", "H3", "L1", "L2", "L3". Defaults to None.
            if None, will use all CDRs i.e. "all"
        abdb_dir (Path, optional): AbDb directory. Defaults to None.
            if None will use the default directory: BASE/'dirs'/'ABDB'
        classifier_dir (Path, optional): LRC AP classifier directory. Defaults to None.
        lrc_ap_cluster (Dict[str, Any], optional): LRC AP cluster information. Defaults to None.
    Returns:
        List[Dict[str, Any]]: Either 
            - A list of dictionaries containing the closest cluster information for each CDR type
            - A dictionary containing the closest cluster information for the specified CDR type
    """
    cdr = cdr or "all"
    # ----------------------------------------
    # parse and validate the struct
    # ----------------------------------------
    criteria, struct_df = process_single_mar_file(
        struct_fp=abdb_dir.joinpath(f"pdb{abdbid}.mar"),
        abdbid=abdbid,
        strict=False,
    )
    if not criteria["cdr_no_missing_residue"]:
        logger.warning(f"{abdbid} CDR has missing residue(s).")

    # ----------------------------------------
    # extract dihedrals and backbone atoms
    # ----------------------------------------
    dihedral_df = extract_phi_psi_omega(struct_df=struct_df)
    bb_df = extract_bb_atoms(struct_df=struct_df, add_residue_identifier=True)
    

    # create a dictionary mapping CDR type to chain labels
    mapping = dihedral_df[['chain', 'cdr']].drop_duplicates().groupby('cdr')['chain'].apply(list).to_dict()
    # ----------------------------------------
    # Specified CDRs 
    # ----------------------------------------
    if cdr != "all":
        # modify mapping to only include the specified CDR type
        try:
            assert cdr in mapping.keys()
            mapping = {cdr: mapping[cdr]}
        except AssertionError:
            logger.error(f"CDR {cdr} not found in {abdbid}.")
            logger.warning("Continue with other CDRs ...")
    
    # ----------------------------------------
    # Iterate over all CDR types
    # ----------------------------------------
    results = []
    for cdr, chain_labels in mapping.items():
        for chain_label in chain_labels:
            logger.info(textwrap.dedent(f"""
                --------------------------------------------------------------
                Processing {abdbid}, CDR: {cdr}, chain_label: {chain_label}
                --------------------------------------------------------------
            """))
            results.append({'chain_type': cdr[0],
                            'chain_label': chain_label,
                            'cdr': cdr,
                            'result': process_cdr(cdr=cdr,
                                                  chain_label=chain_label,
                                                  abdb_dir=abdb_dir,
                                                  dihedral_df=dihedral_df,
                                                  bb_df=bb_df,
                                                  abdbid=abdbid,
                                                  classifier_dir=classifier_dir,
                                                  lrc_ap_cluster=lrc_ap_cluster)
                    })
            logger.info(f"Processing {abdbid}, CDR-{cdr} Done.")
            print("\n\n\n")
    return results

# --------------------
# Main
# --------------------
def cli() -> argparse.Namespace:  # sourcery skip: inline-immediately-returned-variable
    parser = argparse.ArgumentParser(description="Process a single abdb entry",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog=textwrap.dedent("""
        Example: 
            python classify_general_abdb_entry.py \
                -o ./results \
                -c all \
                -db ./dirs/ABDB \
                -clu ./dirs/classifier \
                -f ./dirs/LRC_AP_cluster.json \
                1ikf_0P
            python classify_general_abdb_entry.py \
                --cdr all \
                --outdir ./results \
                --config ./config/classify_general_abdb_entry.yaml \
                1ikf_0P
        """))
    parser.add_argument("abdbid", type=str, help="input file path")
    # cdr
    parser.add_argument('-c', '--cdr', type=str, help="CDR type", default="all",
                        choices=["H1", "H2", "H3", "L1", "L2", "L3", "all"])
    # output 
    parser.add_argument('-o', '--outdir', type=Path, default=Path.cwd()/'output', 
                        help="results directory")
    # abdb 
    parser.add_argument('-db', "--abdb", type=Path, default=BASE/'assets'/'ABDB', 
                        help="AbDb directory (version: 20220926)")
    # precalculated Length and Residue Configuration (LRC) Affinity Propagation (AP) clusters  
    parser.add_argument('-clu', '--lrc_ap_clusters', type=Path, default=BASE/'assets'/'classifier', 
                        help="Length and Residue Configuration (LRC) Affinity Propagation (AP) classifier directory")
    # LRC AP cluster associated info file  
    parser.add_argument('-f', '--lrc_ap_info', type=Path, default=BASE/'assets'/'LRC_AP_cluster.json', 
                        help="LRC_AP_cluster.json file path that holds information about each precalculated LRC AP cluster")
    # add a log file handle 
    parser.add_argument('-l', '--log', type=Path, default=None, 
                        help="log file path to save log info. If None, only print to stdout and stderr.")
    
    # parse args
    args = parser.parse_args()
    
    return args


def main(args) -> None:
    if args.log is not None:
        logger.add(sink=args.log, level="DEBUG")
        
    if is_first_run():
        # first run
        logger.info("First run, downloading assets ...")
        download_and_extract_classifier()
    
    # validate abdb, classifier, LRC_AP_fp
    assert args.abdb.exists(), f"{args.abdb} does not exist."
    assert args.lrc_ap_clusters.exists(), f"{args.lrc_ap_clusters} does not exist."
    assert args.lrc_ap_info.exists(), f"{args.lrc_ap_info} does not exist."
    
    abdb_dir: Path       = args.abdb
    classifier_dir: Path = args.lrc_ap_clusters
    lrc_ap_fp: Path      = args.lrc_ap_info
    outdir: Path         = args.outdir
    abdbid: str          = args.abdbid
    cdr: str             = args.cdr
    
    # ensure output directory exists
    outdir.mkdir(parents=True, exist_ok=True)
    
    # parse the LRC_AP_cluster.json
    with open(file=lrc_ap_fp, mode="r") as f:
        LRC_AP_CLUSTER = json.load(f)
    
    # block: run the worker 
    # ------------------------------------------------------------------------------
    results: List[Dict[str, Any]] = worker(
        abdbid=abdbid, 
        cdr=cdr, 
        abdb_dir=abdb_dir, 
        classifier_dir=classifier_dir, 
        lrc_ap_cluster=LRC_AP_CLUSTER
    )
    # ------------------------------------------------------------------------------
    
    # save results
    o = (outdir / f"{abdbid}.json").resolve()
    results.append({'job': {'abdbid': abdbid, 
                            'cdr': cdr,
                            'abdb_dir': str(abdb_dir),
                            'classifier_dir': str(classifier_dir),
                            'lrc_ap_fp': str(lrc_ap_fp)}})
    with open(o, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {o}")


def app() -> None: 
    args = cli() 
    main(args=args)


# ==================== Main ====================
if __name__ == "__main__":
    app()