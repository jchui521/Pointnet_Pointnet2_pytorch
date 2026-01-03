"""
Template entry to view Twinner01 semantic segmentation results.
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
runpy.run_path(str(REPO_ROOT / 'view_semseg_results.py'), run_name='__main__')
