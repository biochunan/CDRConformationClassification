"""
Use pytest to run this test.
Test 2h32_0P, which has a nan dihedral angle error.
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
    'lrc_ap_clusters': Path('/workspaces/CDRConformationClassification/dirs/classifier'),
    'lrc_ap_info': Path('/workspaces/CDRConformationClassification/dirs/LRC_AP_cluster.json'),
    'log': None
}

args = Namespace(abdbid='2h32_0P', cdr='all', **common_kwargs)  # 1dcl_0: double-light

# ==================== Tests ====================
def test_main():
    # Call the function and check if it runs without any errors
    try:
        print(f"Testing {args.abdbid}")
        main(args)
    except Exception as e:
        pytest.fail(f"{args.abdbid}: function call failed with error: {e}")
