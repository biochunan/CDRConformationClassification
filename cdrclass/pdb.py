# basic 
import pandas as pd
from typing import Dict, List, Union, Tuple, Any

# biopython 
import Bio.PDB.Chain
from Bio.PDB import PDBParser, Structure, Atom

# cdrclass
from cdrclass.alphabet import AA_3to1


def chain2df(chain_obj: Bio.PDB.Chain.Chain, **kwargs) -> pd.DataFrame:
    """
    Turn structure chain into DataFrame, iterate over residues
    Wrapper of function unpack_chain, to accept a struct

    Args:
        chain_obj: (chain: Bio.PDB.Chain.Chain)
        kwargs: retain_hetatm, retain_water, retain_b_factor

    Returns:
        df: (pd.DataFrame) DataFrame
    """
    # d = dict(node_id=[], chain=[], resi=[], alt=[], resn=[], atom=[], element=[], x=[], y=[], z=[])
    retain_hetatm = kwargs.get("retain_hetatm", False)
    retain_water = kwargs.get("retain_water", False)
    retain_b_factor = kwargs.get("retain_b_factor", False)
    d = unpack_chain(
        chain_obj=chain_obj,
        retain_hetatm=retain_hetatm,
        retain_water=retain_water,
        retain_b_factor=retain_b_factor
    )

    df = pd.DataFrame(d)
    # curate column data type
    df["node_id"] = df.node_id.astype("Int64")

    return df


def init_struct_dict():
    return dict(
        node_id=[], chain=[],  # chain
        resi=[], alt=[], resn=[],  # residue
        atom=[], element=[],  # atom
        x=[], y=[], z=[]  # coordinate
    )


def unpack_chain(
    chain_obj: Bio.PDB.Chain.Chain,
    retain_hetatm: bool = False,
    retain_water: bool = False,
    retain_b_factor: bool = False
)-> Dict[str, List[Union[str, int, float]]]:
    """
    Unpack a Biopython Chian object into a dictionary containing

    node_id: (List[int]) zero-based residue index
    chain: (List[str]) chain id
    resi:  (List[int]) residue index
    alt: (List[str]) insertion code
    resn: (List[str]) residue one-letter code
    atom: (List[str]) atom name
    element: (List[str]) atom element type, remove Hydrogen atoms from df
    x, y, z: (List[float]) coordinates

    Args:
        chain_obj: (Bio.PDB.Chain.Chain)
        retain_water: (bool) if True, add WATER ("W") atoms to DataFrame
        retain_hetatm: (bool) if True, add HETATM ("H_*") atoms to DataFrame
        retain_b_factor: (bool) if True, add `b_factor` column to DataFrame

    Returns:
        d: Dict[str, List[Union[str, int, float]]]
    """
    n = -1  # in case of HETATM residue
    chain_id = chain_obj.id
    d = init_struct_dict()
    if retain_b_factor:
        d["b_factor"] = []

    def _add_atom(
        d: Dict, atm: Bio.PDB.Atom.Atom,
        chain_id: str, resi: str, alt: str, resn: str
    ):
        d["chain"].append(chain_id)
        d["resi"].append(int(resi))
        d["alt"].append(alt)
        d["resn"].append(resn)
        d["atom"].append(atm.id)
        d["element"].append(atm.element)
        # coord
        x, y, z = atm.coord
        d["x"].append(float(x))
        d["y"].append(float(y))
        d["z"].append(float(z))
        # b_factor
        if retain_b_factor:
            d["b_factor"].append(atm.get_bfactor())

    for _, res in enumerate(chain_obj):
        # residue info
        het, resi, alt = res.id
        alt = "" if alt == " " else alt
        if het == " ":  # amino acid
            n += 1
            resn = AA_3to1[res.resname]
            atms = res.child_list  # atms = [a.id for a in res.child_list]
            for a in atms:
                d["node_id"].append(n)
                _add_atom(d=d, atm=a, chain_id=chain_id, resi=resi, alt=alt, resn=resn)
        elif het == "W" and retain_water:  # water solvent
            resn = res.resname
            atms = res.child_list  # atms = [a.id for a in res.child_list]
            for a in atms:
                d["node_id"].append(None)
                _add_atom(d=d, atm=a, chain_id=chain_id, resi=resi, alt=alt, resn=resn)
        elif het.startswith("H_") and retain_hetatm:  # HETATM
            resn = f"H_{res.resname}"
            atms = res.child_list  # atms = [a.id for a in res.child_list]
            for a in atms:
                d["node_id"].append(None)
                _add_atom(d=d, atm=a, chain_id=chain_id, resi=resi, alt=alt, resn=resn)
    return d


def chain2df(chain_obj: Bio.PDB.Chain.Chain, **kwargs) -> pd.DataFrame:
    """
    Turn structure chain into DataFrame, iterate over residues
    Wrapper of function unpack_chain, to accept a struct

    Args:
        chain_obj: (chain: Bio.PDB.Chain.Chain)
        kwargs: retain_hetatm, retain_water, retain_b_factor

    Returns:
        df: (pd.DataFrame) DataFrame
    """
    # d = dict(node_id=[], chain=[], resi=[], alt=[], resn=[], atom=[], element=[], x=[], y=[], z=[])
    retain_hetatm = kwargs.get("retain_hetatm", False)
    retain_water = kwargs.get("retain_water", False)
    retain_b_factor = kwargs.get("retain_b_factor", False)
    d = unpack_chain(
        chain_obj=chain_obj,
        retain_hetatm=retain_hetatm,
        retain_water=retain_water,
        retain_b_factor=retain_b_factor
    )

    df = pd.DataFrame(d)
    # curate column data type
    df["node_id"] = df.node_id.astype("Int64")

    return df

