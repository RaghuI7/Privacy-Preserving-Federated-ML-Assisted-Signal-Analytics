# -*- coding: utf-8 -*-
"""PPML revision experiments - IEEE Access-2026-40021 - NOTEBOOK @@NB_ID@@ (@@NB_DATASETS@@)

One file, one Kaggle notebook. Paste the whole file into ONE code cell and run it.
Two notebooks split the work so they can run at the same time on two Kaggle accounts:
    Notebook A : HARTH + PAMAP2            (first session)
    Notebook B : UCI HAR + WISDM v2        (first session)
    Notebook C : UCI HAR + HARTH Byzantine (rerun)
    Notebook D : WISDM v2 + PAMAP2   - DP-SGD at eps 0.85 / 8.3 / 0.3 + controls
    Notebook E : HARTH + UCI HAR     - DP-SGD at eps 0.85 / 8.3 / 0.3 + controls
Both carry the identical `ppml` package (section 2); they differ only in SETTINGS.

    1. SETTINGS   - what to run, how long a session may take
    2. PACKAGE    - every ppml/*.py (do not edit here)
    3. INSTALLER  - writes the package to /kaggle/working/ppml and imports it
    4. DRIVER     - preflight, stage loop, resume, outputs

Kaggle checklist (per notebook)
-------------------------------
* Settings -> Accelerator: GPU T4.  Settings -> Internet: ON (only needed if the
  datasets are not attached and for `pip install dp-accounting`, which is optional).
* Add Input: attach the dataset(s) this notebook uses (see DATASETS below). The loader
  finds them by their FILES, whatever the Kaggle folder is called.
* Click Save Version -> Save & Run All (Commit). PROFILE is already "full".
* Expected run time about 7-9 h. Kaggle allows 12 h; this file stops starting new runs at
  10 h (SESSION_BUDGET_H), then writes the report, so it always ends inside the limit.
* The preflight block at the top of the log shows subjects (UCI 30, PAMAP2 9, WISDM 51,
  HARTH 22), classes and channels - check it once the run has started.
* If it ever ends with NOT COMPLETE: Add Input -> attach that version's Output, then
  Save & Run All again; finished runs are read back and skipped.
* At the end download /kaggle/working/results and /kaggle/working/figures.
"""

import gc
import glob
import inspect
import json
import os
import shutil
import sys
import time
import traceback

# ══════════════════════════ 1. SETTINGS ══════════════════════════════════════════

NOTEBOOK_ID = "@@NB_ID@@"
RUN_DATASETS = @@RUN_DATASETS@@

# Stage order = priority. Each stage runs for BOTH datasets before the next stage starts,
# so if the session budget runs out, it is the low-priority stages that are missing,
# never a whole dataset.
RUN_STAGES = ["main", "privacy", "dplearn", "dpsgd", "splits", "mia", "noniid",
              "clip", "byzantine", "resampling", "features", "cost",
              # revision round 2 (notebooks D and E): one stage per DP-SGD budget
              "dpsgd_e085", "dpsgd_cnn83", "dpsgd_e03", "dpsgd_clip0",
              "dpsgd_cnn3", "dpsgd_cnn085", "dpsgd_cnn0"]
# Which stages run on which dataset (revision scope; stated in the paper).
STAGES_BY_DATASET = @@STAGES_BY_DATASET@@

# "smoke" = 2 seeds / 4 rounds: a plumbing check, logged reportable=False.
# "full"  = the revision scope; the only numbers that may go in the paper.
PROFILE = "full"

# Stop starting new runs after this many hours. Kaggle ends a session at 12 h; the run in
# progress, the report and the output save fit in the remaining margin.
SESSION_BUDGET_H = 10.0

# Per-dataset window cap that KEEPS EVERY SUBJECT (subsamples inside each subject).
# HARTH (~2 h of 50 Hz data per person) and WISDM are subsampled to 12,000 windows;
# every one of the 22 / 51 people is kept.
# The cap is printed in preflight and stamped on every logged run.
MAX_WINDOWS_BY_DATASET = {"HARTH": 12000, "WISDM v2": 12000}

MAX_SUBJECTS = None     # None = every subject (required for reportable runs)
VERBOSE = True
MAKE_FIGURES = True
INSTALL_DIR = ("/kaggle/working" if os.path.isdir("/kaggle/working")
               else os.path.join(os.getcwd(), "ppml_bundle"))

# ══════════════════════════ 2. THE PACKAGE ════════════════════════════════════════
# Byte-identical copies of ppml/*.py and ppml/experiments/*.py.
# Each file starts at its "#@@FILE" line.

_PPML_BLOB = r'''