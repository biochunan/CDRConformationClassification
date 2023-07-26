from Bio.SeqUtils import IUPACData
from typing import Dict

AA_3to1: Dict[str, str] = {k.upper(): v for k, v in IUPACData.protein_letters_3to1.items()}