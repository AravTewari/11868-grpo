#!/usr/bin/env python3
"""
Compatibility wrapper for the PPO harness.
The PPO implementation now lives in scripts/run_ppo.py.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from run_ppo import main as run_ppo_main
from run_ppo import run_ppo_training as run_rlhf_training

main = run_ppo_main


if __name__ == "__main__":
    main()
