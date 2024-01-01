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
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s {%(pathname)s:%(lineno)d} [%(levelname)s] %(name)s - %(message)s [%(threadName)s]',
                    datefmt='%H:%M:%S')

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
    abdbid: str = None
) -> pd.DataFrame:
    """ parse single .mar file """
    
    if abdbid is None:
        abdbid = struct_fp.stem
    
    struct_df = None
    df_H, df_L = gen_struct_cdr_df(
        struct_fp=struct_fp,
        cdr_dict=CDR_HASH_REV["ABM"],
        concat=False,
        retain_b_factor=True,
        retain_hetatm=False,
        retain_water=False
    )

    # concat Heavy and Light chain to single Structure DataFrame
    struct_df = pd.concat([df_H, df_L], axis=0, ignore_index=True)
    
    # correct node_id
    struct_df["node_id"][df_H.shape[0]:] += df_H.node_id.max() + 1

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
                logging.warning(f"CAUTION: Atom type {a} not found in {c}{r}{alt}, resn: {resn}, cdr: {cdr}")

    # add residue identifier if True
    if add_residue_identifier:
        # create reisdue identifier
        ri_list = [f"{c}{r}{a}" for (c, r, a) in cdr_bb_df[["chain", "resi", "alt"]].values]
        cdr_bb_df["ri"] = ri_list

    return cdr_bb_df