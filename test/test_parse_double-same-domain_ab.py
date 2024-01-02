"""
Use pytest to run this test.
Test parsing double-heavy and double-light domains.
They differ from common mAb domains in that they have two heavy chains and two light chains.
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
    'lrc_ap_clusters': Path('/workspaces/CDRConformationClassification/dirs/classifier'),
    'lrc_ap_info': Path('/workspaces/CDRConformationClassification/dirs/LRC_AP_cluster.json'),
    'log': None
}

# args = Namespace(abdbid='3b5g_0', cdr='all', **common_kwargs)  # 3b5g_0: double-light
args = Namespace(abdbid='1dcl_0', cdr='all', **common_kwargs)  # 1dcl_0: double-light
# args = Namespace(abdbid='5vm4_0', cdr='all', **common_kwargs)  # 5vm4_0: double-heavy
# args = Namespace(abdbid='4m3j_0', cdr='all', **common_kwargs)  # 4m3j_0: double-heavy
# args = Namespace(abdbid='6sc5_0P', cdr='all', **common_kwargs)  # 6sc5_0P: double-heavy, complex

# ==================== Tests ====================
def test_main():
    # Call the function and check if it runs without any errors
    try:
        print(f"Testing {args.abdbid}")
        main(args)
    except Exception as e:
        pytest.fail(f"{args.abdbid}: function call failed with error: {e}")
