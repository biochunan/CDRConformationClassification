# basic
import os
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
from typing import List, Tuple, Dict, Union, Optional, Any, Callable, Iterable, Mapping, Set

# Biopython 
from Bio.PDB import PDBParser
from Bio.PDB.Selection import unfold_entities
from Bio.SeqIO import PdbIO

# logging
from loguru import logger

# custom 
from cdrclass.pdb import chain2df


# ==================== Constants ====================
CDRs = ("L1", "L2", "L3", "H1", "H2", "H3")

CDR = {  # http://www.bioinf.org.uk/abs/info.html#cdrdef
    "ABM": {
        "H1": [26, 35],
        "H2": [50, 58],
        "H3": [95, 102],
        "L1": [24, 34],
        "L2": [50, 56],
        "L3": [89, 97]
    },
}

_to_list = lambda a, b, C: [f"{C}{i}" for i in range(a, b + 1)]

CDR_HASH = {  # http://www.bioinf.org.uk/abs/info.html#cdrdef
    "ABM": {
        "H1": _to_list(26, 35, "H"),
        "H2": _to_list(50, 58, "H"),
        "H3": _to_list(95, 102, "H"),
        "L1": _to_list(24, 34, "L"),
        "L2": _to_list(50, 56, "L"),
        "L3": _to_list(89, 97, "L"),
    }
}

CDR_HASH_REV = dict(
    ABM={
        v: i for i, j in CDR_HASH["ABM"].items() for v in j
    },
)


# ==================== Function ====================
def summarize_ab_type_from_chain_map(chain_map: pd.DataFrame) -> List[str]:
    """
    Summarize antibody type from chain_map
    E.g. chain_map 
        chain_type  chain_label  chain_original
    0            H            H               A

    Args:
        chain_map (pd.DataFrame): _description_

    Returns:
        List[str]: e.g.
        ['single-domain', 'single-heavy'] or 
        ['single-domain', 'single-light'] or 
        ['double-domain', 'double-heavy'] or
        ['double-domain', 'double-light'] or
        ['double-domain', 'heavy-light']
    """
    ab_type = []
    # count "H" and "L" in chain_type
    num_ab_chains = 0
    counts = chain_map["chain_type"].value_counts()
    num_H, num_L = counts.get("H", 0), counts.get("L", 0)
    num_ab_chains = sum([num_H, num_L])
    if num_ab_chains == 1:
        ab_type.append("single-domain")
        if num_H == 1:
            ab_type.append("single-heavy")
        elif num_L == 1:
            ab_type.append("single-light")
    elif num_ab_chains == 2:
        ab_type.append("double-domain")
        if num_H == 2:
            ab_type.append("double-heavy")
        if num_L == 2:
            ab_type.append("double-light")
        if num_H == 1 and num_L == 1:
            ab_type.append("heavy-light")
    elif num_ab_chains > 2:
        ab_type.append("multi-domain")
    return ab_type


def get_chain_map_from_remark_950(
        abdb_fp: Path,
        return_ab_type: bool = False
) -> Tuple[float, List[str], pd.DataFrame]:
    # parse REMARK 950
    with open(abdb_fp, "r") as f:
        remark950 = []
        l = f.readline()
        while not l.startswith("REMARK 950"):
            l = f.readline()
        while l.startswith("REMARK 950"):
            remark950.append(l.strip())
            l = f.readline()

    # chain_map
    cols = ["chain_type", "chain_label", "chain_original"]
    chain_map = pd.DataFrame(
        [l.split()[-3:] for l in remark950[2:]],
        columns=cols
    )
    if return_ab_type:
        return chain_map, summarize_ab_type_from_chain_map(chain_map)
    return chain_map


def get_resolution_from_abdb_file(abdb_struct_fp: Path) -> float:
    """ get resolution from .mar file """
    with open(abdb_struct_fp, "r") as f:
        # get resolution from line 1 
        l = f.readline()
        if resolution := re.search(r", ([\d\.]+)A", l):
            resolution = float(resolution[1])
    return resolution


def assign_cdr_class(df: pd.DataFrame, cdr_dict: Dict[str, str]):
    assert "chain" in df.columns and "resi" in df.columns
    df["cdr"] = [
        cdr_dict.get(k, "") for k in map(lambda cr: f"{cr[0]}{cr[1]}", df[["chain", "resi"]].values)
    ]
    return df


def gen_struct_cdr_df(
    struct_fp: Path,
    cdr_dict: Dict[str, str],
    concat: bool = False,
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Args:
        struct_fp: (Path) path to structure file
        cdr_dict: (Dict) mapping residue identifier to CDR label e.g. "L24" => "L1"
        concat: (bool) default False, if True, return single DataFrame by concatenating df_H and df_L
        **kwargs: kwargs for :func: `unpack_chain`, including
            retain_hetatm (bool)
            retain_water (bool)
            retain_b_factor (bool)
    """
    # use default kwargs if not specified
    retain_hetatm = kwargs.get("retain_hetatm", False)
    retain_water = kwargs.get("retain_water", False)
    retain_b_factor = kwargs.get("retain_b_factor", False)

    # vars
    pdbid, pdbfp = struct_fp.stem, struct_fp.as_posix()
    parser = PDBParser()
    structure = parser.get_structure(id=pdbid, file=pdbfp)
    # chain objs
    chain_objs = {c.id: c for c in unfold_entities(structure, "C")}

    # H and L chain Structure DataFrame
    df_H = assign_cdr_class(
        df=chain2df(chain_objs["H"],
        retain_hetatm=retain_hetatm,
        retain_water=retain_water,
        retain_b_factor=retain_b_factor),
        cdr_dict=cdr_dict
    )
    df_L = assign_cdr_class(
        df=chain2df(chain_objs["L"],
        retain_hetatm=retain_hetatm,
        retain_water=retain_water,
        retain_b_factor=retain_b_factor),
        cdr_dict=cdr_dict
    )
    if concat:
        df_L["node_id"] += df_H.node_id.max() + 1
        return pd.concat([df_H, df_L], axis=0)

    return df_H, df_L


def parse_single_mar_file(
    struct_fp: Path,
    abdbid: str = None,
    cdr_definition: str=None
) -> pd.DataFrame:
    """ parse single .mar file """
    abdbid = abdbid or struct_fp.stem
    cdr_definition = cdr_definition or "ABM"
    
    struct_df = None
    struct_df = gen_struct_cdr_df(
        struct_fp=struct_fp,
        cdr_dict=CDR_HASH_REV[cdr_definition],
        retain_b_factor=True,
        retain_hetatm=False,
        retain_water=False,
        concat=True,
    )
    return struct_df


def extract_bb_atoms(struct_df: pd.DataFrame, include_CB: bool = True, add_residue_identifier: bool = True) -> pd.DataFrame:
    assert 'cdr' in struct_df.columns
    # atom set
    ATOMS = ["N", "CA", "C", "O"]
    if include_CB:
        ATOMS += ["CB"]

    # generate CDR Structure DataFrame
    cols = ["node_id", "chain", "resi", "alt", "resn", "atom", "element",
            "cdr", "x", "y", "z", "b_factor"]
    cdr_bb_df = struct_df[(struct_df.cdr != "") & (struct_df.atom.isin(ATOMS))][cols]

    # check if atoms exist
    for node_id in cdr_bb_df.node_id.drop_duplicates().values:
        # check resn
        _df = cdr_bb_df[cdr_bb_df.node_id == node_id]
        resn = _df.resn.drop_duplicates().values
        atms = _df.atom.values
        (c, r, alt, cdr) = _df[["chain", "resi", "alt", "cdr"]].drop_duplicates().values[0]
        cdr = "NA" if cdr == "" else cdr

        assert_atms = ATOMS
        # remove "CB" if the residue is a GLY
        if resn == "G" and "CB" in assert_atms:
            assert_atms.remove("CB")
        # examine atoms
        for a in assert_atms:
            try:
                assert a in atms
            except AssertionError:
                logger.warning(f"CAUTION: Atom type {a} not found in {c}{r}{alt}, resn: {resn}, cdr: {cdr}")

    # add residue identifier if True
    if add_residue_identifier:
        # create reisdue identifier
        ri_list = [f"{c}{r}{a}" for (c, r, a) in cdr_bb_df[["chain", "resi", "alt"]].values]
        cdr_bb_df["ri"] = ri_list

    return cdr_bb_df