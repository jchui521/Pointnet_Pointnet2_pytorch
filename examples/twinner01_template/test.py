"""
Template entry to test Twinner01 using the template-local class config.
"""

import os
import sys
import runpy
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEMPLATE_DIR.parents[1]

sys.path.insert(0, str(TEMPLATE_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'models'))

os.chdir(REPO_ROOT)
runpy.run_path(str(REPO_ROOT / 'twinner01_test.py'), run_name='__main__')
