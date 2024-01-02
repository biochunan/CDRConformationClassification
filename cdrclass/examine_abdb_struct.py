"""
Unit functions for examining AbDb files, to filter out problematic Chothia numbered file identifiers:
1. (file-level) do not have a Chothia numbered file
2. (file-level) have empty Chothia numbered file
3. (struct-level) missing H or L or both HL chains
4. (struct-level) chain length SEQRES < ATMSEQ
5. (struct-level) any of the six CDRs have missing residues

Example cases:
- `3mme_0.vs.3lrs_3P.pse`: 3lrs_3P has missing residues in H3 loop
- `3mme_0.vs.3mug_5PH.pse`: 3mug_5PH does not contain an antibody
- `pdb6s2i_2.cho` has missing residues in its H3 loop, all comparison involve this Free antibody failed.
- `pdb6e8v_0.cho` is a single light chain antibody
- cases like 6s2i_2 and 6s2i_3 have the same set of missing residues in their H3 loop !!!
"""
# basics 
import re
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Sequence, List, Optional, Union

# biopython 
from Bio.PDB import PDBParser
from Bio.PDB.Selection import unfold_entities
from Bio.SeqIO import PdbIO
# logging 
from loguru import logger 

# cdrclass 
from cdrclass.pdb import chain2df
from cdrclass.abdb import CDR_HASH_REV, CDR_HASH
from cdrclass.run_align import pairwise_align_clustalomega, print_format_alignment
from cdrclass.utils import calc_omega_set_residues


def generate_anchor_residue_identifier(cdr_dict: Dict[str, Dict[str, List[str]]], numbering_scheme: str) -> Dict[str, List[Tuple[str, int]]]:
    """
    e.g.
    >>> generate_anchor_residue_identifier(CDR_HASH, "ABM")
    {'L1': [('L', 23), ('L', 35)],
     'L2': [('L', 49), ('L', 57)],
     'L3': [('L', 88), ('L', 98)],
     'H1': [('H', 25), ('H', 36)],
     'H2': [('H', 49), ('H', 59)],
     'H3': [('H', 94), ('H', 103)]}

    Args:
        cdr_dict: (Dict) CDR definition
        numbering_scheme: (str) .upper() must be either "CHOTHIA" or "ABM"

    Returns:
        anchor_res_identifier: (Dict)
    """
    assert numbering_scheme.upper() in ("CHOTHIA", "ABM")

    # out var
    anchor_res_identifier = {
        "L1": None,
        "L2": None,
        "L3": None,
        "H1": None,
        "H2": None,
        "H3": None,
    }

    cdr_scheme = cdr_dict[numbering_scheme]
    for chain in ("H", "L"):
        for loop_id in (1, 2, 3):
            # loop name either H1 H2 H3 L1 L2 L3
            loop = f"{chain}{loop_id}"

            # get resi & convert to int & sort in ascending order
            resi_list = sorted([int(re.search(r"(\d+)", i)[1]) for i in cdr_scheme[loop]])
            anchor_res_identifier[loop] = ([(chain, resi_list[0] - 1),
                                            (chain, resi_list[-1] + 1)])

    return anchor_res_identifier


def get_handle(fp: Path, extension: str):
    """
    Args:
        fp: (Path) file path
        extension: (str) ".gz" or other extensions

    Returns:
        handler: file handler
    """
    handle = None
    if extension == ".gz":
        import gzip
        handle = gzip.open(fp, "rt")
    else:
        handle = open(fp, "r")

    return handle


def extract_atmseq_seqres(struct_fp: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    struct_id = struct_fp.stem
    # check file type
    extension = struct_fp.suffix

    # atmseq
    atmseq = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        handle = get_handle(fp=struct_fp, extension=extension)
        atmseq = {i.id[-1]: str(i.seq).replace("X", "") for i in
                  PdbIO.PdbAtomIterator(handle)}  # Dict[chain_id: str, SeqRecord]
        handle.close()

    # seqres
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        handle = get_handle(fp=struct_fp, extension=extension)
        seqres = {i.id[-1]: str(i.seq) for i in PdbIO.PdbSeqresIterator(handle)}
        handle.close()

    return atmseq, seqres


def assign_cdr_class(df: pd.DataFrame, cdr_dict: Dict[str, str]):
    assert "chain_type" in df.columns and "residue_number" in df.columns
    df["cdr"] = [
        cdr_dict.get(k, "") for k in map(lambda cr: f"{cr[0]}{cr[1]}", df[["chain_type", "residue_number"]].values)
    ]
    return df


def gen_struct_cdr_df(struct_fp: Path,
                      cdr_dict: Dict[str, str],
                      concat: bool = False,
                      **kwargs) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
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
    retain_b_factor = kwargs.get("retain_b_factor", True)

    # vars
    pdbid, pdbfp = struct_fp.stem, struct_fp.as_posix()
    parser = PDBParser()
    structure = parser.get_structure(id=pdbid, file=pdbfp)
    # chain objs
    chain_objs = {c.id: c for c in unfold_entities(structure, "C")}

    # H and L chain Structure DataFrame
    df_H, df_L = [], []
    for chain_label, chain_obj in chain_objs.items():
        if chain_label in ('H', 'h', 'L', 'l'):
            df = chain2df(chain_obj=chain_obj,
                          retain_hetatm=retain_hetatm,
                          retain_water=retain_water,
                          retain_b_factor=retain_b_factor)
            chain_type = "H" if chain_label in ('H', 'h') else "L"
            df['chain_type'] = chain_type
            df['residue_number'] = df['resi']
            if chain_label in ('H', 'h'):
                df_H.append({'chain_type': 'H', 
                             'chain_label': chain_label,
                             'df': assign_cdr_class(df=df, cdr_dict=cdr_dict)})
            else:
                df_L.append({'chain_type': 'L', 
                             'chain_label': chain_label,
                             'df': assign_cdr_class(df=df, cdr_dict=cdr_dict)})
    
    if concat: 
        # iterate over df_H and df_L, and add the max node_id of previous df to the current df
        dfs = []
        for d in df_H + df_L:
            df = d['df']
            df["node_id"] += dfs[-1]["node_id"].max() + 1 if len(dfs) > 0 else 0
            dfs.append(df.copy())
        return pd.concat(dfs, axis=0, ignore_index=True)

    return df_H, df_L


def assert_struct_file_exist(struct_fp: Path):
    struct_id = struct_fp.stem
    try:
        assert struct_fp.exists()
    except AssertionError:
        return False
    return True


def assert_struct_resolution(struct_fp: Path, resolution_thr: float, config: Dict):
    """
    Assert structure resolution is
    Args:
        struct_fp: (Path)
        resolution_thr: (float) resolution threshold, struct with resolution exceeding this value will be filtered out
        config: (Dict)
    """
    struct_id = struct_fp.stem
    struct_4_letter_code = re.search(r"pdb([\d\w]{4})_", struct_fp.stem)[1]
    # retrieve resolution json file
    fn = f"{struct_4_letter_code}.resolution.json"
    json_fp = config["dataset"]["ABDB_resolution"].joinpath("okay", fn)

    if not json_fp.exists():
        json_fp_list = list(
            config["dataset"]["ABDB_resolution"].glob(r"{not_applicable,null_resolution}/" + fn))
        if len(json_fp_list) == 0:
            logger.error(f"{struct_id} Did not find resolution file ... FAILED")
            return False
        else:
            json_fp = json_fp_list[0]

    # parse json
    with open(json_fp, "r") as f:
        data = json.load(f)
    resolution = data["resolution"]

    # check
    if isinstance(resolution, float) and resolution > resolution_thr:
        logger.error(f"{struct_id} resolution value ({resolution}) exceeds threshold ({resolution_thr} Å) ... FAILED")
        return False
    elif isinstance(resolution, str):  # `NOT APPLICABLE` or `NULL`
        logger.error(f"{struct_id} resolution is {resolution} ... FAILED")
        return False

    return True


def assert_non_empty_file(struct_fp: Path) -> bool:
    """ In the past, we encountered some empty AbDb files """
    struct_id = struct_fp.stem
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("test", struct_fp)
        assert len(struct.child_list) > 0
    except AssertionError:
        logger.error(f"{struct_id} Empty file ... FAILED")
        return False
    return True


def assert_H_L_chains(seq_dict: Dict[str, str], seq_type: str):
    chain_ids = list(seq_dict.keys())
    missing_chains = []
    chains_exist = True
    try:
        assert "H" in chain_ids
    except AssertionError:
        logger.error(f"Heavy chain not in {seq_type} ... failed")
        missing_chains.append("H")
        chains_exist = False
    try:
        assert "L" in chain_ids
    except AssertionError:
        logger.error(f"Light chain not in {seq_type} ... failed")
        missing_chains.append("L")
        chains_exist = False

    return chains_exist, missing_chains


def assert_chain_type_exist(struct_fp: Path, chain_type: str) -> bool:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(struct_fp.stem, struct_fp)
    chain_labels = [c.id for c in unfold_entities(struct, "C")]
    try:
        assert chain_type in map(str.upper, chain_labels)
    except AssertionError:
        logger.error(f"{struct_fp.stem} {chain_type=} does not exist ... FAILED")
        return False


def assert_HL_chains_exist(struct_fp: Path,
                           atmseq: Optional[Dict[str, str]] = None) -> bool:
    """
    Assert SEQRES length is not shorter than ATMSEQ.
    If `atmseq` not provided, extract both by parsing file at `struct_fp`.

    Args:
        struct_fp: (Path) path to structure file
        atmseq: (Dict) key: chain ids; val: string sequence, default None.
            see example below

    e.g. input `atmseq`
    atmseq = {
        'H': 'AVKLVQAGGGVVQPGRSLRLSCIASGFTFSNYGMHWVRQAPGKGLEWVAVIWYNGSRTYYGDSVKGRFTISRDNSKRTLYMQMNSLRTEDTAVYYCARDPDILTAFSFDYWGQGVLVTVS',
        'L': 'ELTQPPSVSVSPGQTARITCSANALPNQYAYWYQQKPGRAPVMVIYKDTQRPSGIPQRFSSSTSGTTVTLTISGVQAEDEADYYCQAWDNSASIFGGGTKLTVLGQ'
    }

    Returns:
        True if SEQRES is not shorter than ATMSEQ, otherwise False
    """
    struct_id = struct_fp.stem
    if atmseq is None:
        # get atmseq
        atmseq, _ = extract_atmseq_seqres(struct_fp)

    # assert heavy and light chain exist
    chains_exist, missing_chains = assert_H_L_chains(atmseq, seq_type="ATMSEQ")

    try:
        assert chains_exist
        return True
    except AssertionError:
        logger.error(f"{struct_id}, assert Heavy and Light chains, missing chains: {missing_chains}")
        return False


def assert_seqres_atmseq_length(struct_fp: Path,
                                atmseq: Optional[Dict[str, str]] = None,
                                seqres: Optional[Dict[str, str]] = None) -> bool:
    """
    Assert SEQRES length is not shorter than ATMSEQ.
    If `atmseq` and `seqres` not provided, extract both by parsing file at `struct_fp`.

    Args:
        struct_fp: (Path) path to structure file
        atmseq: (Dict) key: chain ids; val: string sequence, default None.
        seqres: (Dict) key: chain ids; val: string sequence, default None.
            see example below

    e.g. input `atmseq` and `seqres`
    atmseq = {
        'H': 'AVKLVQAGGGVVQPGRSLRLSCIASGFTFSNYGMHWVRQAPGKGLEWVAVIWYNGSRTYYGDSVKGRFTISRDNSKRTLYMQMNSLRTEDTAVYYCARDPDILTAFSFDYWGQGVLVTVS',
        'L': 'ELTQPPSVSVSPGQTARITCSANALPNQYAYWYQQKPGRAPVMVIYKDTQRPSGIPQRFSSSTSGTTVTLTISGVQAEDEADYYCQAWDNSASIFGGGTKLTVLGQ'
    }
    seqres = {
        'H': 'AVKLVQAGGGVVQPGRSLRLSCIASGFTFSNYGMHWVRQAPGKGLEWVAVIWYNGSRTYYGDSVKGRFTISRDNSKRTLYMQMNSLRTEDTAVYYCARDPDILTAFSFDYWGQGVLVTVS',
        'L': 'ELTQPPSVSVSPGQTARITCSANALPNQYAYWYQQKPGRAPVMVIYKDTQRPSGIPQRFSSSTSGTTVTLTISGVQAEDEADYYCQAWDNSASIFGGGTKLTVLGQ'
    }

    Returns:
        True if SEQRES is not shorter than ATMSEQ, otherwise False
    """
    struct_id = struct_fp.stem
    if atmseq is None and seqres is None:
        atmseq, seqres = extract_atmseq_seqres(struct_fp)

    # check if seqres is >= atmseq
    chain_length_okay = True
    for cid, seq in seqres.items():
        try:
            assert len(seq) >= len(atmseq[cid])
        except AssertionError:
            logger.error(f"{struct_id} {cid=} seqres length < atmseq length ... FAILED")
            logger.error(f"seqres: {seq}")
            logger.error(f"atmseq: {atmseq[cid]}")
            chain_length_okay = False
    
    return chain_length_okay


def _gen_seq_id(seq_iterable: Sequence[str]):
        n = -1
        for char in seq_iterable:
            if char != "-":
                n += 1
                yield n
            else:
                yield char


def _cdr_has_missing_residues(*, 
                              merged_df: pd.DataFrame, 
                              chain_type: str,
                              anchor_res: Dict[str, List[Tuple[str, int]]],
                              struct_id: str)-> Dict[str, bool]:
    """
    Find CDR missing residues (if any)
    Updates: use anchor residue as boundary residue node_id

    e.g. merged_df of pdb3lrs_1.mar
    node_id, chain, resi, alt, cdr, seqres_id, atmseq_id, seqres, atmseq
        96    ,     H,   93,    ,    ,        96,        96,      A,      A
        97    ,     H,   94,    ,    ,        97,        97,      R,      R     # =====> anchor residue "H94" <=====
    --------------------------------------------------------------------------------------------------------------
        98    ,     H,   95,    ,  H3,        98,        98,      E,      E     # CDR-H3 loop
        99    ,     H,   96,    ,  H3,        99,        99,      A,      A     # CDR-H3 loop
       100    ,     H,   97,    ,  H3,       100,       100,      G,      G     # CDR-H3 loop
       101    ,     H,   98,    ,  H3,       101,       101,      G,      G     # CDR-H3 loop
       102    ,     H,   99,    ,  H3,       102,       102,      P,      P     # CDR-H3 loop
       nan    ,   nan,  nan, nan, nan,       103,         -,      I,      -     # CDR-H3 loop
       nan    ,   nan,  nan, nan, nan,       104,         -,      W,      -     # CDR-H3 loop
       nan    ,   nan,  nan, nan, nan,       105,         -,      H,      -     # CDR-H3 loop
       nan    ,   nan,  nan, nan, nan,       106,         -,      D,      -     # CDR-H3 loop
       nan    ,   nan,  nan, nan, nan,       107,         -,      D,      -     # CDR-H3 loop
       nan    ,   nan,  nan, nan, nan,       108,         -,      V,      -     # CDR-H3 loop
       nan    ,   nan,  nan, nan, nan,       109,         -,      K,      -     # CDR-H3 loop
       nan    ,   nan,  nan, nan, nan,       110,         -,      Y,      -     # CDR-H3 loop
       nan    ,   nan,  nan, nan, nan,       111,         -,      Y,      -     # CDR-H3 loop
       nan    ,   nan,  nan, nan, nan,       112,         -,      D,      -     # CDR-H3 loop
       nan    ,   nan,  nan, nan, nan,       113,         -,      F,      -     # CDR-H3 loop
       nan    ,   nan,  nan, nan, nan,       114,         -,      N,      -     # CDR-H3 loop
       nan    ,   nan,  nan, nan, nan,       115,         -,      D,      -     # CDR-H3 loop
       nan    ,   nan,  nan, nan, nan,       116,         -,      G,      -     # CDR-H3 loop
       103    ,     H,  100,   N,  H3,       117,       103,      Y,      Y     # CDR-H3 loop
       104    ,     H,  100,   O,  H3,       118,       104,      Y,      Y     # CDR-H3 loop
       105    ,     H,  100,   P,  H3,       119,       105,      N,      N     # CDR-H3 loop
       106    ,     H,  100,   Q,  H3,       120,       106,      Y,      Y     # CDR-H3 loop
       107    ,     H,  100,   R,  H3,       121,       107,      H,      H     # CDR-H3 loop
       108    ,     H,  100,   S,  H3,       122,       108,      Y,      Y     # CDR-H3 loop
       109    ,     H,  100,   T,  H3,       123,       109,      M,      M     # CDR-H3 loop
       110    ,     H,  101,    ,  H3,       124,       110,      D,      D     # CDR-H3 loop
       111    ,     H,  102,    ,  H3,       125,       111,      V,      V     # CDR-H3 loop
    --------------------------------------------------------------------------------------------------------------
       112    ,     H,  103,    ,    ,       126,       112,      W,      W     # =====> anchor residue "H103" <=====
       113    ,     H,  104,    ,    ,       127,       113,      G,      G

    - `nan`: denotes empty cells, 
    - empty cells => empty str i.e. ''

    Args:
        merged_df: (DataFrame) struct dataframe
        chain_type: (str) either "H" or "L"
        anchor_res: (Dict) anchor residue identifier
        struct_id: (str) structure id
    Returns:
        cdr_containing_missing_residues: (Dict) key: cdr loop name; val: bool
        e.g. {'H1': False, 'H2': False, 'H3': True} 
        means H1 and H2 have no missing residues, H3 has missing residues
    """
    # if chain_type not in the dataframe, add column chain_type 
    assert chain_type in ('H', 'L')
    if "chain_type" not in merged_df.columns:
        merged_df["chain_type"] = chain_type
    # out vars
    cdr_containing_missing_residues = {}

    # iterate over H1 H2 H3 OR L1 L2 L3 depending on `chain`
    for i in (1, 2, 3):
        # cdr loop name
        cdr = f"{chain_type}{i}"

        # assert there are CDR residues in the merged DataFrame
        cdr_loop_exist = True
        r = merged_df[merged_df["cdr"] == cdr]["seqres_id"].values
        if r.shape[0] == 0:
            # no CDR loop was found in the merged DataFrame
            cdr_loop_exist = False
            cdr_containing_missing_residues[cdr] = True
            logger.error(f"{struct_id}: CDR {cdr} is missing")

        if cdr_loop_exist:
            # Then examine missing residues using anchor residues
            # get anchor residue seqres_id
            a, b = anchor_res[cdr][0][1], anchor_res[cdr][1][1]
            anchor_seqres_id_begin = merged_df[(merged_df.chain_type == chain_type) & (merged_df.resi == a)].seqres_id.values
            anchor_seqres_id_end = merged_df[(merged_df.chain_type == chain_type) & (merged_df.resi == b)].seqres_id.values

            # assert anchor residues exist
            anchor_seqres_exist = True
            try:
                assert anchor_seqres_id_begin.shape[0] > 0
            except AssertionError:
                anchor_seqres_exist = False
                cdr_containing_missing_residues[cdr] = True
                logger.warning(f"{struct_id} {cdr} missing anchor residue {chain_type}{a}")
            try:
                assert anchor_seqres_id_end.shape[0] > 0
            except AssertionError:
                anchor_seqres_exist = False
                cdr_containing_missing_residues[cdr] = True
                logger.warning(f"{struct_id} {cdr} missing anchor residue {chain_type}{b}")

            if anchor_seqres_exist:
                # slice Merged Structure&Alignment DataFrame using anchor residue atmseq_id
                cdr_loop = merged_df[(merged_df.seqres_id > anchor_seqres_id_begin[0]) &
                                        (merged_df.seqres_id < anchor_seqres_id_end[0])]["atmseq"].values
                # if "-" in `atmseq`, then set `cdr_containing_missing_residues[cdr]` to True
                if "-" in cdr_loop:
                    cdr_containing_missing_residues[cdr] = True
                    logger.error(f"{struct_id}: CDR {cdr} has missing residues.")
                else:
                    cdr_containing_missing_residues[cdr] = False

    return cdr_containing_missing_residues


def assert_no_cdr_missing_residues(struct_fp: Path,
                                        chain_label: str, 
                                        chain_type: str,
                                        struct_df: pd.DataFrame,
                                        clustal_omega_executable: str, 
                                        numbering_scheme: str,
                                        atmseq: str,
                                        seqres: str,
                                        ) -> Tuple[bool, Dict[str, bool]]:
    """
    Assert no CDR missing residues in the CDR loops of the specified chain.
    Args:
        struct_fp: (Path) path to structure file
        chain_label: (str) chain label
        chain_type: (str) chain type
        struct_df: (pd.DataFrame) Structure DataFrame for a chain 
        clustal_omega_executable: (str) path to clustal omega executable
        numbering_scheme: (str) cdr numbering scheme either `CHOTHIA` or `ABM`
        atmseq: (str) ATMSEQ sequence
        seqres: (str) SEQRES sequence
    Returns:
        bool: True if no missing residues detected in any of the CDR loops in the specified chain, 
            otherwise False meaning there are missing residues in at least one of the CDR loops in the specified chain
        result: (Dict) key: cdr loop name; val: bool
            e.g. {'H1': False, 'H2': False, 'H3': True} for chain_type="H"
    """
    df = struct_df.copy()
    # vars
    numbering_scheme = numbering_scheme.upper()
    struct_id = struct_fp.stem
    ANCHOR_RES = generate_anchor_residue_identifier(cdr_dict=CDR_HASH,
                                                    numbering_scheme="ABM")

    # Extract SEQRES & ATMSEQ sequences if not provided
    if atmseq is None or seqres is None:
        atmseq, seqres = extract_atmseq_seqres(struct_fp)

    try:
        assert numbering_scheme in ("CHOTHIA", "ABM")
    except AssertionError:
        raise ValueError(f"Numbering scheme must be either `Chothia` or `AbM`. Provided: {numbering_scheme}")

    # Align Heavy chain, seq1: SEQRES, seq2: ATMSEQ
    logger.debug(f"{struct_id} aligning {chain_type} chain {chain_label} ...")
    if 'X' in atmseq:
        logger.warning(f"{struct_id} {chain_type=} {chain_label=} atmseq contains 'X' residues, removing ...")
        atmseq = atmseq.replace('X', '')
    aln = pairwise_align_clustalomega(clustal_omega_executable=clustal_omega_executable,
                                      seq1=seqres,
                                      seq2=atmseq.replace("X", ""))

    # Create two DataFrames for each chain
    # DF 1: Structure DataFrame containing info about (1) CDR labels; (2) node id.
    df = df.drop_duplicates("node_id", ignore_index=True)

    # DF 2: convert alignment string (SEQRES vs. ATMSEQ) to DataFrame => Alignment DataFrame
    df_aln = pd.DataFrame(dict(seqres_id=_gen_seq_id(aln[0].seq),
                                    atmseq_id=_gen_seq_id(aln[1].seq),
                                    seqres=list(aln[0].seq),
                                    atmseq=list(aln[1].seq)))
    """
    e.g. if missing residues, 3lrs_1
    seqres_id, atmseq_id, seqres, atmseq 
    101,       101      ,   G   ,     G 
    102,       102      ,   P   ,     P 
    103,       -        ,   I   ,     -  
    104,       -        ,   W   ,     - 
    ...        ...      ,  ...  ,    ...
    116,       -        ,   G   ,     -  
    117,       103      ,   Y   ,     Y 
    118,       104      ,   Y   ,     Y 
    """
    # Merge Alignment DataFrame with Structure DataFrame on atmseq_id (node_id)
    df_aln_m = pd.merge(df[["node_id", "chain", "resi", "alt", "cdr"]], 
                        df_aln,
                        left_on="node_id", 
                        right_on="atmseq_id",
                        how="outer")
    df_aln_m.sort_values(by=["seqres_id"], ascending=True, inplace=True)
    
    # check if any cdr loop has missing residues
    result = _cdr_has_missing_residues(merged_df=df_aln_m, 
                                       chain_type=chain_type,
                                       anchor_res=ANCHOR_RES,
                                       struct_id=struct_id)

    # print alignment if found missing residues
    if True in result.values():
        aln_str = print_format_alignment(alns=[str(i.seq) for i in aln],
                                         ids=[f"{struct_id}_{chain_label}_SEQRES", f"{struct_id}_{chain_label}_ATMSEQ"],
                                         return_fmt_aln_str=True)
        logger.info(f"{struct_id}: {chain_type=} {chain_label=} alignment SEQRES vs. ATMSEQ:\n{aln_str}")

    # check missing residues
    if True in result.values():
        return False, result

    return True, result


def assert_cdr_no_big_b_factor(struct_fp: Path,
                               struct_df: pd.DataFrame,
                               b_factor_atoms: List[str],
                               b_factor_thr: float,
                               numbering_scheme: str,
                               ) -> bool:
    """
    Assert CDR loop specified atom set B factor is smaller than the thr by default 80.
    Args:
        struct_fp: (Path) path to abdb file
        struct_df: (pd.DataFrame) Structure DataFrame for a chain 
        b_factor_thr: (float) default 80.0, specified atom B factor should < thr
        numbering_scheme: (str) cdr numbering scheme either `CHOTHIA` or `ABM`
        b_factor_atoms: (List[str]) List of atoms e.g. ["CA"]

    Returns:
        bool
    """
    df = struct_df.copy()
    struct_id = struct_fp.stem

    # assert numbering scheme
    numbering_scheme = numbering_scheme.upper()
    try:
        assert numbering_scheme in ("CHOTHIA", "ABM")
    except AssertionError:
        raise ValueError(f"Numbering scheme must be either `Chothia` or `AbM`. Provided: {numbering_scheme}")

    # get cdr atoms
    # cdr_df = pd.concat([
    #     # df_H[(df_H.cdr != "") & list(map(lambda x: x in b_factor_atoms, df_H.atom))],
    #     # df_L[(df_L.cdr != "") & list(map(lambda x: x in b_factor_atoms, df_L.atom))]
    #     df_H[(df_H.cdr != "") & (df_H.atom.isin(b_factor_atoms))],
    #     df_L[(df_L.cdr != "") & (df_L.atom.isin(b_factor_atoms))]
    # ])
    cdr_df = df.query('cdr != "" and atom in @b_factor_atoms')

    bad_atoms = cdr_df[(cdr_df.b_factor >= b_factor_thr) | (cdr_df.b_factor == 0.000)]
    if bad_atoms.shape[0] > 0:
        logger.error(f"{struct_id} questionable B factor atoms:\n"
                     f"{bad_atoms.to_string()}\n")
        return False

    return True


def assert_cdr_no_non_proline_cis_peptide(struct_fp: Path,
                                          numbering_scheme: str,
                                          struct_df: pd.DataFrame = None,
                                          ) -> bool:
    # 1. Structure DataFrame
    if struct_df is None:
        struct_df = gen_struct_cdr_df(struct_fp=struct_fp,
                                      cdr_dict=CDR_HASH_REV[numbering_scheme],
                                      concat=True)

    cdr_nodes = struct_df[struct_df.cdr != ""].node_id.drop_duplicates().to_list()

    # calculate omega angles for cdr residues
    omega_list: List[Tuple[str, str, float]] = calc_omega_set_residues(struct_df=struct_df, node_ids=cdr_nodes)

    # aa-wise check stereo-isomers cis or trans
    cdr_no_non_proline_cis_peptide = True
    cis_proline = []
    non_proline_cis_residues = []
    for (ri, aa, omega) in omega_list:
        # issue warning if non-proline cis residue
        if -90. < omega < 90:
            if aa != "P":
                logger.warning(f"{ri}.{aa}: non-proline cis residue (measured with its preceding residue)")
                non_proline_cis_residues.append((ri, aa, omega))
            else:
                cis_proline.append((ri, aa, omega))

    # report
    if len(non_proline_cis_residues) > 0:
        cdr_no_non_proline_cis_peptide = False
        report_str = "cis-peptide thr: -90 < ω < 90\n" \
                     "residue    cdr    aa     omega(degree)\n"
        for (ri, aa, omega) in non_proline_cis_residues:
            chain_resi = re.match(r"([HL]\d+)[A-Z]*", ri)[1]  # remove insertion code
            report_str += f"{ri:>7}  {CDR_HASH_REV[numbering_scheme].get(chain_resi, ''):>5}  " \
                          f"{aa:>4}  {omega:>17.2f}\n"
        logger.warning(f"Non-proline peptide detected for the following residues:\n"
                       f"{report_str}")
    # else:
    #     logger.info("No non-proline cis residue detected, looking good.")

    # report cis-proline
    if len(cis_proline) > 0:
        report_str = "cis-peptide thr: -90 < ω < 90\n" \
                     "residue    cdr    aa     omega(degree)\n"
        for (ri, aa, omega) in cis_proline:
            chain_resi = re.match(r"([HL]\d+)[A-Z]*", ri)[1]  # remove insertion code
            report_str += f"{ri:>7}  {CDR_HASH_REV[numbering_scheme].get(chain_resi, ''):>5}  " \
                          f"{aa:>4} {omega:>17.2f}\n"
        logger.warning(f"Found cis-proline:\n"
                       f"{report_str}")

    return cdr_no_non_proline_cis_peptide

