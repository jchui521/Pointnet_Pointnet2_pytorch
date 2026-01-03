"""
Template entry to run the Twinner01 quickstart using template config.
"""

import os
import sys
import runpy
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEMPLATE_DIR.parents[1]

sys.path.insert(0, str(TEMPLATE_DIR))
sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)
runpy.run_path(str(REPO_ROOT / 'twinner01_quickstart.py'), run_name='__main__')
