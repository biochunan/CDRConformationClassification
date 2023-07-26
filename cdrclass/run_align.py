import os
import shutil
import tempfile
import contextlib
from typing import Optional, List

from Bio import SeqIO
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqIO import SeqRecord
from Bio.Align.Applications import ClustalOmegaCommandline

@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str] = None):
    """Context manager that deletes a temporary directory on exit."""
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# align seq using ClustalOmega
def pairwise_align_clustalomega(
    clustal_omega_executable: str,
    seq1: str = None, seq2: str = None,
    seqs: List[str] = None
) -> List[SeqRecord]:
    # sourcery skip: extract-method, for-append-to-extend, remove-redundant-fstring, simplify-generator
    """

    Args:
        seqs:
        seq1:
        seq2:
        clustal_omega_executable: (str) path to clustal omega executable
            e.g. "/usr/local/bin/clustal-omega"
    Returns:

    """
    # assert input
    if seqs is None and (seq1 is None or seq2 is None):
        raise NotImplementedError(
            "Provide either List of seqs as `seqs` OR a pair of seqs as `seq1` and `seq2`."
        )

    # generate seq_recs
    seq_rec = [None]
    if seqs:
        seq_rec = [
            SeqRecord(id=f"seq{i + 1}", seq=Seq(seqs[i]), description="")
            for i in range(len(seqs))
        ]
    elif seq1 is not None and seq2 is not None:
        seq_rec = [
            SeqRecord(id='seq1', seq=Seq(seq1), description=""),
            SeqRecord(id='seq2', seq=Seq(seq2), description=""),
        ]

    with tmpdir_manager() as tmpdir:
        # executable
        cmd = clustal_omega_executable

        # create input seq fasta file and output file for clustal-omega
        in_file = os.path.join(tmpdir, "seq.fasta")
        out_file = os.path.join(tmpdir, f"aln.fasta")
        with open(in_file, "w") as f:
            SeqIO.write(seq_rec, f, "fasta")
        # create Clustal-Omega commands
        clustalomega_cline = ClustalOmegaCommandline(cmd=cmd, infile=in_file, outfile=out_file, verbose=True, auto=True)

        # run Clustal-Omega
        stdout, stderr = clustalomega_cline()

        # read aln
        aln_seq_records = []
        with open(out_file, "r") as f:
            for record in AlignIO.read(f, "fasta"):
                aln_seq_records.append(record)

        return aln_seq_records


# print formatted alignment
def print_format_alignment(
    aln1: str = None, aln2: str = None,
    alns: List[str] = None,
    ids: List[str] = None,
    return_fmt_aln_str: bool = False,
    pdbid: str = None, **kwargs
) -> Optional[str]:
    # assert inputs
    if (aln1 and aln2) and (alns is None):
        raise NotImplementedError(
            "Either provide both `aln1` and `aln2` or a list of seqs to `alns`"
        )
            # only works for equal aln seq length
    if aln1 and aln2:
        assert len(aln1) == len(aln2)
    if alns:
        seq_lens = {len(a) for a in alns}
        assert len(seq_lens) == 1

    # generate seq ids
    pdbid = "" if pdbid is None else f"({pdbid})"
    ids = ids or [f"seq{i + 1}" for i in range(3)]
    # match ids length
    max_id_len = max(len(i) for i in ids)
    ids = [f"{''.join([' '] * (max_id_len - len(i)))}{i}" for i in ids]

    # print alignment
    n = len(alns[0])
    step = kwargs["step"] if "step" in kwargs else 50
    fmt_aln = "\n"

    def gen_aln_block(i):
        seq_str = []
        seqs = []
        end = i + step if i + step < n else n
        for j in range(len(alns)):
            seq = alns[j][i:i + step]
            # insert space to every 10 aa
            seq = " ".join([seq[k:k + 10] for k in range(0, len(seq), 10)])
            seq_str.append(f"{ids[j]}{pdbid}  {i + 1:<4}  {seq}  {end:<4} \n")
            seqs.append(seq)
        # columns
        _id = ''.join([' ']*max_id_len)
        # generate alignment column
        seq_col = ""
        _len = len(seqs[0])
        for k in range(_len):
            aa = list({seqs[x][k] for x in range(len(alns))})
            if aa != [" "] and len(aa) == 1:
                seq_col += "*"
            elif aa == [" "]:
                seq_col += " "
            else:
                seq_col += " "
        seq_str.append(f"{_id}{pdbid}  {i + 1:<4}  {seq_col}  {end:<4} \n")

        # output seq str
        seq = "".join(seq_str) + "\n"
        return seq

    for i in range(0, n, step):
        aln_block = gen_aln_block(i)
        fmt_aln += aln_block

    if return_fmt_aln_str:
        return fmt_aln

    print(fmt_aln)
