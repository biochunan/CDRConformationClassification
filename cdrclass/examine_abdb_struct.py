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

# cdrclass 
from cdrclass.pdb import chain2df
from cdrclass.abdb import CDR_HASH_REV, CDR_HASH
from cdrclass.run_align import pairwise_align_clustalomega, print_format_alignment
from cdrclass.utils import calc_omega_set_residues

# logging 
import logging
logger = logging.getLogger("examine_abdb_struct")


def generate_anchor_residue_identifier(cdr_dict: Dict[str, Dict[str, List[str]]], numbering_scheme: str):
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
    assert "chain" in df.columns and "resi" in df.columns
    df["cdr"] = [
        cdr_dict.get(k, "") for k in map(lambda cr: f"{cr[0]}{cr[1]}", df[["chain", "resi"]].values)
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
    retain_b_factor = kwargs.get("retain_b_factor", False)

    # vars
    pdbid, pdbfp = struct_fp.stem, struct_fp.as_posix()
    parser = PDBParser()
    structure = parser.get_structure(id=pdbid, file=pdbfp)
    # chain objs
    chain_objs = {c.id: c for c in unfold_entities(structure, "C")}

    # H and L chain Structure DataFrame
    df_H = assign_cdr_class(df=chain2df(chain_objs["H"],
                                        retain_hetatm=retain_hetatm,
                                        retain_water=retain_water,
                                        retain_b_factor=retain_b_factor),
                            cdr_dict=cdr_dict)
    df_L = assign_cdr_class(df=chain2df(chain_objs["L"],
                                        retain_hetatm=retain_hetatm,
                                        retain_water=retain_water,
                                        retain_b_factor=retain_b_factor),
                            cdr_dict=cdr_dict)
    if concat:
        df_L["node_id"] += df_H.node_id.max() + 1
        return pd.concat([df_H, df_L], axis=0)

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


def assert_HL_chains_exist(struct_fp: Path,
                           atmseq: Optional[Dict[str, str]] = None) -> bool:
    """
    Assert SEQRES length is not shorter than ATMSEQ.
    If `atmseq` not provided, extract both by parsing file at `struct_fp`.

    Args:
        struct_fp: (Path) path to structure file
        atmseq: (Dict) key: cahin ids; val: string sequence, default None.
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
        atmseq: (Dict) key: cahin ids; val: string sequence, default None.
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
    if len(seqres["H"]) < len(atmseq["H"]):
        logger.error(f"{struct_id} Heavy chain length SEQRES < ATMSEQ ... FAILED\n"
                     f"seqres: {seqres['H']}\n"
                     f"atmseq: {atmseq['H']}")
        chain_length_okay = False
    if len(seqres["L"]) < len(atmseq["L"]):
        logger.error(f"{struct_id} Light chain length SEQRES < ATMSEQ ... FAILED\n"
                     f"seqres: {seqres['L']}\n"
                     f"atmseq: {atmseq['L']}")
        chain_length_okay = False

    return chain_length_okay


def assert_cdr_no_missing_residues(struct_fp: Path,
                                   clustal_omega_executable: str,
                                   numbering_scheme: str,
                                   atmseq: Optional[Dict[str, str]] = None,
                                   seqres: Optional[Dict[str, str]] = None,
                                   df_H: pd.DataFrame = None,
                                   df_L: pd.DataFrame = None,
                                   ) -> bool:
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
    H_aln = pairwise_align_clustalomega(clustal_omega_executable=clustal_omega_executable,
                                        seq1=seqres["H"], seq2=atmseq["H"].replace("X", ""))
    L_aln = pairwise_align_clustalomega(clustal_omega_executable=clustal_omega_executable,
                                        seq1=seqres["L"], seq2=atmseq["L"].replace("X", ""))

    # Create two DataFrames for each chain
    # DF 1: Structure DataFrame containing info about (1) CDR labels; (2) node id.
    if df_H is None or df_L is None:
        df_H, df_L = gen_struct_cdr_df(struct_fp=struct_fp, cdr_dict=CDR_HASH_REV[numbering_scheme])
    df_H = df_H.drop_duplicates("node_id", ignore_index=True)
    df_L = df_L.drop_duplicates("node_id", ignore_index=True)

    # DF 2: convert alignment string (SEQERS vs. ATMSEQ) to DataFrame => Alignment DataFrame
    def gen_seq_id(seq_iterable: Sequence[str]):
        n = -1
        for char in seq_iterable:
            if char != "-":
                n += 1
                yield n
            else:
                yield char

    df_aln_H = pd.DataFrame(dict(seqres_id=gen_seq_id(H_aln[0].seq),
                                 atmseq_id=gen_seq_id(H_aln[1].seq),
                                 seqres=list(H_aln[0].seq),
                                 atmseq=list(H_aln[1].seq)))
    df_aln_L = pd.DataFrame(dict(seqres_id=gen_seq_id(L_aln[0].seq),
                                 atmseq_id=gen_seq_id(L_aln[1].seq),
                                 seqres=list(L_aln[0].seq),
                                 atmseq=list(L_aln[1].seq)))
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
    df_aln_H = pd.merge(df_H[["node_id", "chain", "resi", "alt", "cdr"]], df_aln_H,
                        left_on="node_id", right_on="atmseq_id",
                        how="outer")
    df_aln_H = df_aln_H.sort_values(by=["seqres_id"], ascending=True)
    df_aln_L = pd.merge(df_L[["node_id", "chain", "resi", "alt", "cdr"]], df_aln_L,
                        left_on="node_id", right_on="atmseq_id",
                        how="outer")
    df_aln_L = df_aln_L.sort_values(by=["seqres_id"], ascending=True)

    # func to find CDR missing residues if any
    def cdr_has_missing_residues(merged_df, chain: str):
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

        where nan denotes empty cells, cells without a value are empty str i.e. ""

        Args:
            merged_df: (DataFrame) struct dataframe
            chain: (str) either "H" or "L"
        """
        # out vars
        cdr_containing_missing_residues = {}

        # iterate over H1 H2 H3 OR L1 L2 L3 depending on `chain`
        for i in (1, 2, 3):
            # cdr loop name
            cdr = f"{chain}{i}"

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
                a, b = ANCHOR_RES[cdr][0][1], ANCHOR_RES[cdr][1][1]
                anchor_seqres_id_begin = merged_df[(merged_df.chain == chain) & (merged_df.resi == a)].seqres_id.values
                anchor_seqres_id_end = merged_df[(merged_df.chain == chain) & (merged_df.resi == b)].seqres_id.values

                # assert anchor residues exist
                anchor_seqres_exist = True
                try:
                    assert anchor_seqres_id_begin.shape[0] > 0
                except AssertionError:
                    anchor_seqres_exist = False
                    cdr_containing_missing_residues[cdr] = True
                    logger.warning(f"{struct_id} {cdr} missing anchor residue {chain}{a}")
                try:
                    assert anchor_seqres_id_end.shape[0] > 0
                except AssertionError:
                    anchor_seqres_exist = False
                    cdr_containing_missing_residues[cdr] = True
                    logger.warning(f"{struct_id} {cdr} missing anchor residue {chain}{b}")

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

    # check if any cdr loop has missing residues
    result_H = cdr_has_missing_residues(merged_df=df_aln_H, chain="H")
    result_L = cdr_has_missing_residues(merged_df=df_aln_L, chain="L")

    # print alignment if found missing residues
    if True in result_H.values():
        aln_str = print_format_alignment(alns=[str(i.seq) for i in H_aln],
                                         ids=[f"{struct_id}_H_SEQERS", f"{struct_id}_H_ATMSEQ"],
                                         return_fmt_aln_str=True)
        logger.info(f"{struct_id}: Heavy chain alignment SEQRES vs. ATMSEQ:\n{aln_str}")

    if True in result_L.values():
        aln_str = print_format_alignment(alns=[str(i.seq) for i in L_aln],
                                         ids=[f"{struct_id}_L_SEQERS", f"{struct_id}_L_ATMSEQ"],
                                         return_fmt_aln_str=True)
        logger.info(f"{struct_id}: Light chain alignment SEQRES vs. ATMSEQ:\n{aln_str}")

    # check missing residues
    if True in result_H.values() or True in result_L.values():
        return False

    return True


def assert_cdr_no_big_b_factor(struct_fp: Path,
                               b_factor_atoms: List[str],
                               b_factor_thr: float,
                               numbering_scheme: str,
                               df_H: pd.DataFrame = None,
                               df_L: pd.DataFrame = None,
                               ) -> bool:
    """
    Assert CDR loop specified atom set B factor is smaller than the thr by default 80.
    Args:
        struct_fp: (Path) path to abdb file
        b_factor_thr: (float) default 80.0, specified atom B factor should < thr
        numbering_scheme: (str) cdr numbering scheme either `CHOTHIA` or `ABM`
        b_factor_atoms: (List[str]) List of atoms e.g. ["CA"]
        df_H: (pd.DataFrame) heavy chain Structure DataFrame
        df_L: (pd.DataFrame) light chain Structure DataFrame

    Returns:
        bool
    """
    struct_id = struct_fp.stem

    # assert numbering scheme
    numbering_scheme = numbering_scheme.upper()
    try:
        assert numbering_scheme in ("CHOTHIA", "ABM")
    except AssertionError:
        raise ValueError(f"Numbering scheme must be either `Chothia` or `AbM`. Provided: {numbering_scheme}")

    # create H/L chain dataframe
    if df_H is None or df_L is None:
        df_H, df_L = gen_struct_cdr_df(struct_fp=struct_fp,
                                       cdr_dict=CDR_HASH_REV[numbering_scheme],
                                       retain_b_factor=True)

    # get cdr atoms
    cdr_df = pd.concat([
        # df_H[(df_H.cdr != "") & list(map(lambda x: x in b_factor_atoms, df_H.atom))],
        # df_L[(df_L.cdr != "") & list(map(lambda x: x in b_factor_atoms, df_L.atom))]
        df_H[(df_H.cdr != "") & (df_H.atom.isin(b_factor_atoms))],
        df_L[(df_L.cdr != "") & (df_L.atom.isin(b_factor_atoms))]
    ])

    questionable_atoms = cdr_df[(cdr_df.b_factor >= b_factor_thr) | (cdr_df.b_factor == 0.000)]
    if questionable_atoms.shape[0] > 0:
        logger.error(f"{struct_id} questionable B factor atoms:\n"
                     f"{questionable_atoms.to_string()}\n")
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


if __name__ == "__main__":
    config = {
        "dataset": {
            "ABDB": Path("/Users/chunan/Dataset/AbDb/abdb_newdata"),
            "ABDB_resolution": Path("/Users/chunan/Dataset/AbDb/abdb_new_data.resolution"),
        },
        "input": {
            "free_vs_complex_ab_list": Path(
                "/Users/chunan/UCL/scripts/abag_interface_analysis/abyint/data/processed/free_vs_complex_ab.08July2022.csv"),
        },
        "executable": {
            "clustal-omega": "/usr/local/bin/clustal-omega",
        },
        "numbering_scheme": "ABM",
        "b_factor_atom_set": ["CA"],
        "b_factor_cdr_thr": 80.,
        "resolution_thr": 2.8
    }

    # ab_id = struct_id = "6jfi_1"  # resolution failed
    ab_id = struct_id = "8fab_1"  # high torsion energy at H97 CDR-H3
    logger.info(f"{ab_id}")
    struct_fp = config["dataset"]["ABDB"].joinpath(f"pdb{struct_id}.mar")

    # criteria
    criteria = dict(
        mar_struct_exist=False,
        mar_struct_resolution=False,
        mar_struct_not_empty=False,
        struct_okay=False,
        chain_length_okay=False,
        cdr_no_missing_residue=False,
        cdr_no_big_b_factor=False,
        cdr_no_high_torsion_energy=False,
        cdr_no_non_proline_cis_peptide=False,
    )
    if ab_id is None:
        ab_id = struct_fp.stem

    # assert mar file exists
    criteria["mar_struct_exist"] = assert_struct_file_exist(struct_fp=struct_fp)
    if not criteria["mar_struct_exist"]:
        logger.error(f"{ab_id} mar file does not exist ... FAILED")

    # check structure resolution
    if criteria["mar_struct_exist"]:
        criteria["mar_struct_resolution"] = assert_struct_resolution(struct_fp=struct_fp,
                                                                     resolution_thr=config["resolution_thr"],
                                                                     config=config)
    if not criteria["mar_struct_resolution"]:
        logger.error(f"{ab_id} resolution greater than {config['resolution_thr']} ... FAILED")

    # assert not empty file
    if criteria["mar_struct_exist"]:
        criteria["mar_struct_not_empty"] = assert_non_empty_file(struct_fp=struct_fp)
        if not criteria["mar_struct_not_empty"]:
            logger.error(f"{ab_id} mar file is empty ... FAILED")

    # assert mar structure is okay
    if criteria["mar_struct_not_empty"] and criteria["mar_struct_resolution"]:
        # get atmseq and seqres
        atmseq, seqres = extract_atmseq_seqres(struct_fp=struct_fp)

        # 1. check heavy and light chains
        criteria["chain_okay"] = assert_HL_chains_exist(struct_fp=struct_fp, atmseq=atmseq)
        if not criteria["chain_okay"]:
            logger.error(f"{ab_id} chain ... FAILED")

        # 2. check seqres vs atmseq length
        if criteria["chain_okay"]:
            criteria["chain_length_okay"] = assert_seqres_atmseq_length(struct_fp=struct_fp,
                                                                        atmseq=atmseq,
                                                                        seqres=seqres)
            if not criteria["chain_length_okay"]:
                logger.error(f"{ab_id} SEQRES vs ATMSEQ chain length ... FAILED")

        # 3. check CDRs
        # parse abdb file
        df_H, df_L = gen_struct_cdr_df(struct_fp=struct_fp,
                                       cdr_dict=CDR_HASH_REV[config["numbering_scheme"]],
                                       concat=False,
                                       retain_b_factor=True)
        if criteria["chain_length_okay"]:
            criteria["cdr_no_missing_residue"] = assert_cdr_no_missing_residues(
                struct_fp=struct_fp,
                clustal_omega_executable=config["executable"]["clustal-omega"],
                numbering_scheme=config["numbering_scheme"],
                atmseq=atmseq, seqres=seqres,
                df_H=df_H, df_L=df_L
            )
            if not criteria["cdr_no_missing_residue"]:
                logger.error(f"{ab_id} CDR ... FAILED")

        # 4. check loop CA B-factor (filter out ≥ 80 & == 0.)
        if criteria["cdr_no_missing_residue"]:
            criteria["cdr_no_big_b_factor"] = assert_cdr_no_big_b_factor(
                struct_fp=struct_fp,
                b_factor_atoms=config["b_factor_atom_set"],
                b_factor_thr=config["b_factor_cdr_thr"],
                numbering_scheme=config["numbering_scheme"],
                df_H=df_H, df_L=df_L
            )
            if not criteria["cdr_no_big_b_factor"]:
                logger.error(f"{ab_id} Loop B factor ... FAILED")

        # concat Heavy and Light chain to single Structure DataFrame
        struct_df = pd.concat([df_H, df_L], axis=0, ignore_index=True)
        struct_df["node_id"][df_H.shape[0]:] += df_H.node_id.max() + 1  # correct node_id

        # 5. check torsion energy
        if criteria["cdr_no_missing_residue"]:
            criteria["cdr_no_high_torsion_energy"] = assert_cdr_no_high_torsion_energy(
                struct_fp=struct_fp,
                numbering_scheme=config["numbering_scheme"],
                struct_df=struct_df
            )
            if not criteria["cdr_no_high_torsion_energy"]:
                logger.warning(f"{ab_id} CDR torsion energy ... WARNING")

        # 6. check non-Proline cis peptide i.e. -π/2 < ω < π/2
        if criteria["cdr_no_missing_residue"]:
            criteria["cdr_no_non_proline_cis_peptide"] = assert_cdr_no_non_proline_cis_peptide(
                struct_fp=struct_fp,
                numbering_scheme=config["numbering_scheme"],
                struct_df=struct_df
            )
            if not criteria["cdr_no_non_proline_cis_peptide"]:
                logger.warning(f"{ab_id} CDR exists non-proline cis peptide ... WARNING")

        # Finally, if all passed, set struct_okay=True
        if all((criteria["chain_okay"], criteria["mar_struct_resolution"],
                criteria["chain_length_okay"], criteria["cdr_no_missing_residue"],
                criteria["cdr_no_big_b_factor"])):
            criteria["struct_okay"] = True
