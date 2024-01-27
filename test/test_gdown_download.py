# basic
import shutil
from pathlib import Path
import gdown
import subprocess
import pytest

BASE = Path(__file__).resolve().parent
ASSETS_URL = 'https://drive.google.com/uc?id=1kERH5wYVMhCvmAlDPL835Ms4ZhxL0pQQ'
ASSETS_ID = '1kERH5wYVMhCvmAlDPL835Ms4ZhxL0pQQ'

# pytest function to test download of assets.tar.gz from google drive using gdown
# and ensure that it can be tar decompressed
def test_download():
    o = Path(__file__).resolve().parent.joinpath('test_output', 'assets.tar.gz')
    d = Path(__file__).resolve().parent.joinpath('test_output', 'assets')
    d.mkdir(parents=True, exist_ok=True)
    gdown.download(id=ASSETS_ID, output=o.as_posix(), quiet=False,
                   fuzzy=True, use_cookies=False, verify=False)
    # assert that assets.tar.gz is downloaded
    assert o.is_file()
    # assert that assets.tar.gz can be decompressed
    subprocess.run(['tar', '-xzf', o.as_posix(), '-C', d.as_posix()])
    # assert d/'classifier' is a directory, d/'LRC_AP_cluster.json' is a file
    assert d.joinpath('classifier').is_dir()
    assert d.joinpath('LRC_AP_cluster.json').is_file()
    # remove assets.tar.gz and assets
    o.unlink()
    shutil.rmtree(d.as_posix())
