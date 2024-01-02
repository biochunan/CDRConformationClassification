"""
Use pytest to run this test.
Test parsing single-heavy and single-light domains.
They differ from common mAb domains in that they only have one heavy chain and one light chain.
In AbDb, the Ab chains are numbered: H and h for heavy chains, L and l for light chains.
"""
import pytest
from pathlib import Path
from loguru import logger
from argparse import Namespace
import cdrclass 
from cdrclass.app import main

# ==================== Configuration ====================
common_kwargs = {
    'outdir': Path.cwd().joinpath('test_output'),
    'abdb': Path('/AbDb'),
    'lrc_ap_clusters': Path(cdrclass.__path__[0]).joinpath('assets', 'classifier'),
    'lrc_ap_info': Path(cdrclass.__path__[0]).joinpath('assets', 'LRC_AP_cluster.json'),
    'log': None
}
cases = [
    Namespace(abdbid='1u0q_1', cdr='all', **common_kwargs),  # 1u0q_1: single-heavy
    Namespace(abdbid='1mvf_1', cdr='all', **common_kwargs),  # 1mvf_1: single-heavy
]

# ==================== Tests ====================
def test_main():
    for args in cases:
        # Call the function and check if it runs without any errors
        try:
            print(f"Testing {args.abdbid}")
            main(args)
        except Exception as e:
            pytest.fail(f"Function call failed with error: {e}")
