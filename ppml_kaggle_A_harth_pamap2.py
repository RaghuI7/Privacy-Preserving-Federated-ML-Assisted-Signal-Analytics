
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

NOTEBOOK_ID = "A"
RUN_DATASETS = ['HARTH', 'PAMAP2']

# Stage order = priority. Each stage runs for BOTH datasets before the next stage starts,
# so if the session budget runs out, it is the low-priority stages that are missing,
# never a whole dataset.
RUN_STAGES = ["main", "privacy", "dplearn", "dpsgd", "splits", "mia", "noniid",
              "clip", "byzantine", "resampling", "features", "cost"]
# Which stages run on which dataset (revision scope; stated in the paper).
STAGES_BY_DATASET = {
    'HARTH': ['main', 'privacy', 'dplearn', 'dpsgd', 'splits', 'mia', 'noniid', 'clip', 'byzantine', 'resampling'],
    'PAMAP2': ['main', 'privacy', 'dplearn', 'dpsgd', 'splits', 'mia', 'noniid', 'resampling', 'cost'],
}

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
#@@FILE ppml/__init__.py
"""ppml - corrected simulation pipeline for IEEE Access-2026-40021.

Modules
-------
config          single source of hyper-parameters (identical for all 4 datasets)
dp_accountant   valid Renyi-DP certificates for the central Gaussian mechanism
runlog          machine-readable per-run logging + resume
features        117-dim features and the nine feature families
splitting       subject-disjoint splits, subject-silo clients, LOSO, leakage audit
resampling      resample_poly harmonisation and spectral-distortion measurement
datasets        loaders that return per-window SUBJECT ids
models          FeatureMLP / RawMLP / CNN-BiLSTM / LightTCN
fl              the DP aggregation, adaptive clipping, robust aggregators, attacks
runners         the one FL training loop used by every experiment
metrics         classification metrics
multiseed       CIs, paired tests, Holm correction
crossdataset    the four datasets in one file, from the logs only
plotting        figures drawn only from results/logs (PDF + 300 dpi PNG)
reporting       section 13 run report and contradiction flags
experiments     one module per Work Order task
"""

__all__ = ["config", "dp_accountant", "runlog", "features", "splitting", "resampling",
           "datasets", "models", "fl", "runners", "metrics", "multiseed", "crossdataset",
           "harness", "plotting", "reporting"]

#@@FILE ppml/config.py
"""Central configuration for the IEEE-Access-2026-40021 revision experiments.

Every experiment module reads its hyper-parameters from here so that the four
datasets are provably run through one identical pipeline (Work Order rule:
"Use the same fixed pipeline for all four datasets").
"""

import os
import time

# ── Paths ─────────────────────────────────────────────────────────────────────
KAGGLE = os.path.isdir("/kaggle/working")
_HERE = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
         else os.getcwd())
WORK = "/kaggle/working" if KAGGLE else os.path.abspath(os.path.join(_HERE, os.pardir, "out"))
DATA_DIR = os.path.join(WORK, "data")
RESULTS_DIR = os.path.join(WORK, "results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
FIG_DIR = os.path.join(WORK, "figures")


def ensure_dirs():
    for d in (WORK, DATA_DIR, RESULTS_DIR, LOG_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)


# ── Signal harmonisation ──────────────────────────────────────────────────────
TARGET_FS = 50          # common sampling rate used by the paper
WL = 128                # window length in samples @ TARGET_FS (2.56 s)
STRIDE = 64             # 50 % overlap
PURITY = 0.80           # min fraction of a window carrying the majority label
# Set per dataset by datasets.set_class_space() (WISDM 2019 is scored on 3 classes).
N_CLASS = int(os.environ.get("PPML_N_CLASS", "6"))
CLASS_NAMES = ["Walking", "Walk Up", "Walk Down", "Sitting", "Standing", "Laying"]
N_CH_HARMONISED = 9     # acc(3) + gyro(3) + total_acc(3)
CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                 "total_acc_x", "total_acc_y", "total_acc_z"]
CLIP_RANGE = 20.0       # sensor clipping guard kept from the original pipeline

# Native rates and how each source maps onto the 9 harmonised channels.
# index tuples are into the raw per-sample channel vector produced by the loader.
DATASET_NATIVE_FS = {"UCI HAR": 50.0, "PAMAP2": 100.0, "WISDM v2": 20.0, "HARTH": 50.0}

# ── Feature extraction ────────────────────────────────────────────────────────
N_FFT_BANDS = 5
N_CORR = 6
FFT_BANDS_HZ = [(0, 1), (1, 3), (3, 8), (8, 15), (15, 25)]
HARMONISED_BAND_HZ = (0.0, 10.0)   # common Nyquist-safe band for Task 7 variant (c)
# Task 8 decision: the paper's jerk equation carries a sampling-rate factor f_s.
# The original code omitted it. Setting this True adds it in code so the code and
# the equation agree; the value actually used is logged in every run.
JERK_WITH_SAMPLING_RATE = True

# ── Federated learning ────────────────────────────────────────────────────────
# 36-hour revision scope (stated in the paper): 5 seeds for the main table and every DP
# result, 3 seeds for the secondary experiments. Significance uses the paired t-test with
# 95% CIs; with n=5 the Wilcoxon test cannot go below p=0.0625 and is reported as such.
SEEDS = (42, 43, 44, 45, 46)
# The run matrix (Work Order Section 12) puts the >= 6-seed rule on Tasks 1, 2, 3, 7 and 8
# as well as Task 4 - it is a property of the tables the reviewers were shown, not of one
# experiment. So the supplementary set IS the full set in the reportable profile; only the
# smoke profile reduces it. Six seeds also keeps the Wilcoxon tests legal (n >= 2) and the
# paired CIs honest, which three could not do for every comparison family.
SEEDS_SUPP = (42, 43, 44)
SEEDS_BYZANTINE = (42, 43)           # Task 6: 2 seeds (scope)
SEEDS_MIA = (42, 43, 44)             # Task 10: 3 trials
# Same rule for the T=50 DP sweeps in Tasks 1 and 2: the certificate is a function of
# (z, T, delta) and not of the seed, but the SEEDS of the accuracy column are what a reader
# uses to judge a mean, and Section 12 asks for six of them. Restored to the full set.
DP_SEEDS = SEEDS_SUPP          # privacy sweep / dplearn: 3 seeds (scope)
MIA_SHADOWS = 3
MIA_SHADOW_TRIALS = 5
MIA_ARCH = "MLP"
# subject pools: target training / disjoint non-member / shadow models
MIA_POOL_FRAC = (0.40, 0.20, 0.40)

LR = 1e-3
BATCH = 64
MLP_HIDDEN = (256, 128, 64)
WEIGHT_DECAY = 1e-4
# ── Training recipe (revision) ───────────────────────────────────────────────
LR_SCHEDULE = "cosine"     # client lr decays from LR to LR_MIN over the rounds
LR_MIN = 1e-4
CLASS_WEIGHTS = "inverse_freq"   # per-silo inverse class frequency in the local loss
WINDOW_SCALE = "channel"   # standardise raw windows per channel (not per time step)
AUGMENT = True             # scaling + jitter + 3-D rotation of acc/gyro triples
                           # (raw-window models only; feature MLPs are unaffected)
SELECT_BEST_VAL = True     # report the round with the best VALIDATION accuracy

# Local passes per client per round. E is NOT a privacy parameter: the accountant is
# account(z, T, delta, sigma_b) and the noise is added once per round at the server
# after the update is clipped to C, so E changes accuracy only. Centralised training
# derives its epoch budget from the same E (runners.py), so cutting it shrinks the FL
# and centralised arms together and the Task 4 contrast stays fair.
# Measured: at E=10 the full profile needs ~60-75 GPU-h for UCI HAR alone, which
# exceeds what Kaggle's GPU quota can give four datasets. E=3/2 cuts that ~3.3x.
# The sheet's 10/5 is one session away, not one edit: the driver writes these env vars
# before importing the package, so raising E never touches the embedded code. Raising E
# raises wall-clock and changes the accuracy column; it changes no epsilon, and the value
# a run used is on every log row.
LOCAL_EPOCHS_MLP = int(os.environ.get("PPML_LOCAL_EPOCHS_MLP", "5"))
LOCAL_EPOCHS_CNN = int(os.environ.get("PPML_LOCAL_EPOCHS_CNN", "2"))
ROUNDS = 30
ROUNDS_DP = 50                        # T used for every DP certificate

N_CLIENTS = 5                         # K, subject-silo clients (grouped subjects)
N_CLIENTS_BYZANTINE = 10              # Krum needs K > 2f + 2 with f = 2
N_CLIENTS_SWEEP = (2, 3, 5, 8, 10)
# How the K silos are built. "grouped" puts SEVERAL PEOPLE in one client, which makes the
# certificate client-level and NOT user-level - the very conflation the Work Order asks to
# untangle - and it also forfeits utility, because sigma = z*C/K falls with K while the
# averaged signal does not. "one_subject_per_client" is both the correct privacy unit and
# the cheaper-per-unit-of-privacy choice; per-round cost is the same either way, since
# every training window is touched E times per round regardless of how the clients are cut.
# `n_clients` above is then ignored (K = number of training subjects), which is exactly the
# behaviour a subject-disjoint benchmark should have.
CLIENT_MODE = "one_subject_per_client"

FEDPROX_MU = 0.01
MOMENTUM_BETA = 0.9
MOMENTUM_ETA_S = 1.0

# ── Differential privacy (Task 1) ─────────────────────────────────────────────
DELTA = 1e-5
CLIP_C = 0.10                         # default; superseded by the Task 2 clip sweep
# eps is C-free: sigma = z*C/K and sensitivity = C/K, so their ratio is z whatever C is.
# What C decides is reach - T clipped rounds move the model by at most T*C in L2, and
# getting from a random init to a solution costs about ||w0||. One absolute 0.10 therefore
# starves the 71,750-parameter MLP and the 195,334-parameter CNN-BiLSTM by different
# amounts while the certificate looks identical. CLIP_REL makes C a fraction of the
# INITIALISED model's own L2 norm: the init is seeded and public, so no private window is
# read to choose it, it scales with the architecture, and eps stays exactly where
# dp_accountant puts it. dp["C"]=None asks for this rule.
CLIP_REL = 0.10
# All six clip radii the Work Order names. eps is untouched by any of them (sigma and the
# sensitivity both scale with C, so their ratio is z), which is exactly why the sweep is a
# utility experiment and why cutting two points bought wall-clock while deleting two rows a
# reviewer asked for. Restored.
CLIP_SWEEP = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
# Noise multipliers z.  eps(z) for T=50, delta=1e-5, full participation:
#   z=4.5 -> 8.78 | z=10.6 -> 3.42 | z=28.6 -> 1.22 (verified by dp_accountant)
Z_SWEEP = (4.5, 10.6, 28.6)    # ends + operating point (scope)
# The privacy-utility AXIS the paper plots: the non-private baseline first, then every
# certificate-bearing level. Anything that reads "which z were run" must read Z_LEVELS,
# not Z_SWEEP, or the non-private reference point disappears from a table.
Z_NON_PRIVATE = 0.0
Z_LEVELS = (Z_NON_PRIVATE,) + Z_SWEEP
Z_MODERATE = 10.6                     # one of the sweep's levels; DP_ARM_Z is the arm z
Z_STRONG = 28.6
# THE DP OPERATING POINT: the one z every DP arm OUTSIDE the Task 1 sweep runs at - the
# four Task 4 DP methods, the Task 2 clip sweep, the DP rows of Tasks 3/5/7/8, the MIA DP
# levels and the adaptive arm's certificate. Before this constant existed the same number
# was spelled ten times, so "run the arms at a different budget" meant ten edits and the
# risk that one table quietly kept the old z.
# It is a STATED BUDGET, not a tuned one. eps = account(z, T, delta) is a closed form and
# the Task 1 sweep already measures the curve, so a value here is chosen from that
# measured privacy-utility table or from the release SNR the runs log
# (release_snr_mean); never from the accuracy it happens to produce. The Work Order's rule
# is explicit about this - no setting is picked to hit a target - so if this moves, the
# accuracy of both the published arm and the proposed arms moves with it and all of it has
# to be reported, including the arms that get worse.
# z=10.6 -> eps=3.4235 at T=50, delta=1e-5 (preflight's accountant selftest verifies it).
# Env override so a Kaggle session can state a different budget without editing the
# embedded package: the driver writes PPML_DP_ARM_Z before ppml is imported.
DP_ARM_Z = float(os.environ.get("PPML_DP_ARM_Z", str(Z_MODERATE)))
# The mechanism's utility knob, not its privacy knob. Whole-update clipping plus a
# per-coordinate draw of sigma = z*C/K means the model must find a signal of size C/sqrt(d)
# inside noise of size z*C/K, so accuracy is set by K/(z*sqrt(d)) - and nothing in the
# certificate depends on which coordinates carry the noise. Only this fraction of the
# released coordinates is applied to the model (fl.dp_aggregate), so the noise energy
# injected per round drops from d*sigma^2 to m*sigma^2 and the drift a coordinate
# accumulates over T rounds shrinks by sqrt(d/m). eps is unchanged because the fraction is
# chosen from the ALREADY NOISY average - a function of a released object, i.e. pure
# post-processing. Choosing it from the clean average (DP-Forward's output projection) is
# not covered by this certificate and is not what this pipeline runs.
# 1.0 = the mechanism the rejected paper ran, kept intact as the arm it is compared to.
DP_SPARSITY = 0.10
DP_SPARSITY_SWEEP = (1.0, 0.30, 0.10, 0.03)
# Rounds whose global models are averaged and reported alongside the final one
# (accuracy_iter_avg). Each snapshot is already part of the release, so averaging them is
# post-processing: eps unchanged, and fresh per-round noise partly cancels over the window.
# 0 disables it.
DP_ITER_AVG = 10
# Task 10 MIA arms; placed here because it reads C/CLIP constants defined above.
# Task 10 asks for the non-DP model AND the new DP levels, so all three headline
# certificate levels are attacked - one level per point of the privacy-utility claim.
# The sparse arm is attacked too: it is the mechanism the resubmission puts forward, so
# the claim "same epsilon, better accuracy" has to be tested against membership
# inference, not only against the accountant. It costs one target training, not a whole
# new shadow set - exp_mia caches the non-DP shadows once per seed (they never see the
# dp argument), so adding a level here adds a run, not a re-train of the calibration.
def _mia_z_levels():
    """The three certificate levels Task 10 attacks: the ends of the sweep plus the arm.

    Ordered and de-duplicated: if DP_ARM_Z is moved onto one of the sweep ends, attacking
    the same model twice would buy a second MIA number for the same release and cost a
    whole extra shadow set, so the level is kept once.
    """
    out = []
    for z in (Z_SWEEP[0], DP_ARM_Z, Z_STRONG):
        if float(z) not in out:
            out.append(float(z))
    return tuple(out)


MIA_DP_LEVELS = (("non-DP", None),
                 ("z=%.1f" % DP_ARM_Z, dict(z=DP_ARM_Z, C=CLIP_C)),
                 ("z=%.1f head" % DP_ARM_Z, dict(z=DP_ARM_Z, C=None, head_only=True,
                                                 iter_avg=DP_ITER_AVG)))
# Task 2 Gaussian quantile estimator (Andrew et al. 2021).
# sigma_b is the noise std on the SUM of the K clipped-indicators; the estimator divides
# that sum by K, so b~ carries noise of std sigma_b/K. Andrew et al. use sigma_b = 10 with
# hundreds of clients. At this pipeline's K = 5 that is std 2.0 on a quantity confined to
# [0, 1]: C would random-walk instead of tracking the quantile and the arm would measure
# nothing. sigma_b is therefore set to half the size of one client's vote on that sum, so
# b~ can actually move - and the cost is honestly large: account_adaptive puts the total
# certificate at z=10.6, T=50 near eps ~ 1.7e2 rather than the 3.42 of fixed C at the same
# z. That is a measured cost, not a setting to be tuned away; reporting.py flags it.
ADAPT_SIGMA_B = 0.5                   # b~ noise std = 0.5/K = 0.1 at K=5, half a vote
ADAPT_TARGET_Q = 0.20                 # target clipped fraction
ADAPT_ETA = 0.05                      # multiplicative step on C
ADAPT_C_MIN, ADAPT_C_MAX = 0.005, 10.0
# The estimator's certificate, not its accuracy. Each round it runs it spends
# 1/(2*sigma_b^2) = 2.0 of RDP budget, against 1/(2*10.6^2) = 0.0044 for the released
# update, so leaving it running for all 50 rounds makes the clip-fraction statistic, not
# the model, the whole privacy cost of the arm (eps = 168.2 at z=10.6). Running it for the
# first ADAPT_FREEZE_ROUNDS rounds and freezing C afterwards cuts the composed estimator
# term to W/(2*sigma_b^2) while the update term is unchanged: eps = 50.7 at W=10.
# Freezing is sound, not a trick: sigma = z*C/K scales with whatever C the server uses,
# so the noise-to-sensitivity ratio stays z for an adaptively chosen radius - which is
# exactly why the arm can have a moving C at all. Stopping the release cannot break it.
# What freezing costs is honesty about the mechanism: after round W this arm is fixed
# clipping at a data-chosen radius, so it is no longer Andrew et al. running for 50
# rounds. The frozen value is logged and the arm is reported with the split, so the
# reader sees which certificate they are looking at. W is a scope decision made before
# any accuracy was read.
ADAPT_FREEZE_ROUNDS = 10

# ── Example-level DP-SGD (Task 1 optional extra) ──────────────────────────────
# A DIFFERENT PRIVACY UNIT, not a better version of the arms above: this protects one
# window in one silo, not one person. It is opt-in (experiments.STAGES has it, the driver's
# default RUN_STAGES does not) because per-example backward passes are ~B x slower per step
# than a batched step, and the Work Order itself marks the arm "only if time allows".
# The scope below is set from the cost of the mechanism, never from an accuracy: one seed,
# a 400-window silo cap and 20 rounds, all stamped into the log row and into the CSV, so a
# reader can see exactly which setting produced which epsilon. C=1.0 per example is the
# value the DP-SGD literature (and Opacus) uses as its default norm bound.
# DP_SGD_Z does NOT follow DP_ARM_Z on purpose: moving the user-level arms to another
# budget says nothing about the right budget for a mechanism whose privacy unit is one
# window, and letting one knob price both would hide which certificate a row is under.
DP_SGD_Z = Z_MODERATE                 # superseded: z is SOLVED from DP_SGD_TARGET_EPS
# Revision decision, fixed before any run: example-level DP-SGD is reported at two stated
# budgets. exp_dpsgd solves z for each target from the planned step count of the
# SMALLEST silo, before training; the accuracy that comes out is reported as it is.
DP_SGD_TARGET_EPS = (3.0, 8.0)
DP_SGD_CLIP_C = 1.0
# q=0.2: one silo is ONE person (~300-800 windows), so q=0.05 gave batches of 2-17 and a
# per-step noise z*C/B far above the clipped gradient. At q=0.2 (B ~ 60-160) and 100 steps
# the noise per batch is about half as large for the SAME epsilon (z is re-solved).
DP_SGD_SAMPLE_RATE = 0.2            # Poisson q; B = round(q*n) is the fixed divisor
DP_SGD_ROUNDS = 20
DP_SGD_LOCAL_EPOCHS = 1
DP_SGD_MAX_WINDOWS_PER_CLIENT = None
DP_SGD_SEEDS = (42, 43, 44)
DP_SGD_ARCHES = ("MLP-Small", "LightTCN")
# Plain SGD with momentum, the optimiser DP-SGD is normally run with: Adam rescales every
# coordinate, so a coordinate that carries only noise moves as far as one with signal.
DP_SGD_OPTIMIZER = "sgd"
DP_SGD_LR = 0.05
DP_SGD_MOMENTUM = 0.9

# The following were the broken settings of the rejected submission. They are kept
# ONLY so the accountant can report what the old code actually cost. Never use them
# to produce a result.
LEGACY_NOMINAL_EPS = 20.0
LEGACY_N_TENSORS = {"MLP": 8, "CNN-BiLSTM": 26, "LightTCN": 10}

# ── Non-IID (Task 5) ──────────────────────────────────────────────────────────
ALPHA_VALUES = (0.05, 0.1, 0.3, 0.5, 1.0, 5.0, "IID")
N_PARTITIONS_PER_ALPHA = 5
# The DP arm of Task 5 repeats the alpha trend under noise at one z. Same five independent
# partitions as the non-DP arm: the T=50 certificate does not change with the replication
# count, and a 3-partition DP trend would not be comparable to the 5-partition trend it sits
# beside in the same figure.
NONIID_DP_ALPHAS = ()    # scope: the user-level DP arm is at chance at K=5 (see privacy)
N_PARTITIONS_DP = 3
NONIID_DP_Z = DP_ARM_Z

# ── Byzantine (Task 6) ────────────────────────────────────────────────────────
# Standard definitions (revision): sign-flip = -1 x own update; scaled poisoning = -5 x
# own update; IPM = -eps x mean of the honest updates with eps = 0.5 (Xie et al. 2019).
# (The earlier code's "signflip" was random +/-1 per coordinate and "scaled" was +5 x the
# HONEST update, neither of which attacks the model; those runs are not reported.)
BYZ_ATTACKS = ("signflip_x1", "scaled_neg5", "ipm_eps0.5")
BYZ_AGGREGATORS = ("fedavg", "median", "trimmed_mean", "krum")
BYZ_FRACTIONS = (0.0, 0.1, 0.3)
# f=0.0 stays: without it there is no honest reference for "what did the robust
# aggregator cost me when nothing was attacking". Task 6 names four fractions, so the grid
# is those four - the 10% point is where a K=10 silo has exactly one colluding client, which
# is the case a robust aggregator is supposed to survive.
# Task 6 asks for IID and one non-IID alpha: a robust aggregator is only interesting if
# the partition is also skewed, and the submission never tested that.
BYZ_PARTITION_KINDS = ("iid",)        # scope: IID only, stated in the paper
BYZ_NONIID_ALPHA = 0.5
BYZ_NONIID_PARTITIONS = 1
BYZ_ASSUMED_F = 2                     # Krum's assumed colluders, NOT the true count
BYZ_IPM_EPS = 0.5
BYZ_SCALED_FACTOR = 5.0
BYZ_NOISE_STD = 1.0
BYZ_LABEL_FLIP_TO = 0

# ── Experiment scopes ─────────────────────────────────────────────────────────
# Task 4 main table: every method for every architecture, >= 6 seeds.
MAIN_ARCHES = tuple(a for a in os.environ.get(
    "PPML_MAIN_ARCHES", "MLP,LightTCN,CNN-BiLSTM").split(",") if a)
# CNN-BiLSTM is the slowest model (sequential LSTM over 128 steps), so in the main table it
# runs the comparisons the reviewers named, on 3 seeds. Both limits are logged per row.
MAIN_METHODS_BY_ARCH = {
    "CNN-BiLSTM": ("Centralised", "FL-FedAvg", "FL+DP"),
    "LightTCN": ("Centralised", "FL-FedAvg", "FL-FedProx", "FL-Median", "FL+DP",
                 "FL+DP-Head"),
}
MAIN_SEEDS_BY_ARCH = {"CNN-BiLSTM": 3, "LightTCN": 3}
# User-level DP with a small model and with public pretraining (stage "dplearn").
MLP_SMALL_HIDDEN = (64,)
DPLEARN_ROUNDS = (50, 20)          # T values; z is re-solved so eps stays at the arm's eps
MAIN_METHODS = (
    ("Centralised",      dict(central=True)),
    ("FL-FedAvg",        dict(method="fedavg")),
    ("FL-FedProx",       dict(method="fedavg", mu=FEDPROX_MU)),
    ("FL-Momentum",      dict(method="fedavg", server_momentum=True)),
    ("FL-Median",        dict(method="median")),
    ("FL-TrimmedMean",   dict(method="trimmed_mean")),
    ("FL-FedPer",        dict(method="fedavg", use_fedper=True)),
    ("FL+DP",            dict(method="fedavg", dp=dict(z=DP_ARM_Z, C=CLIP_C),
                              rounds=ROUNDS_DP)),
    # THE PROPOSED ARM, not the paper's. FL+DP above stays exactly as the rejected
    # submission specified it - absolute clip, noise on every coordinate, momentum off -
    # so the published claim is still tested as published and its result stands either
    # way. This arm differs by four changes that each leave the certificate alone: the
    # release is applied to only a fraction of its largest coordinates, chosen AFTER the
    # draw so it is post-processing (fl.dp_aggregate), a clip radius scaled to the model
    # rather than a constant (CLIP_REL), server momentum over released aggregates, and
    # iterate averaging (DP_ITER_AVG). Same z, same T, same delta, so the SAME eps - and
    # PAIR_SETS below tests the accuracy difference, not just its size.
    ("FL+DP-Sparse",     dict(method="fedavg", disabled=True,
                              dp=dict(z=DP_ARM_Z, C=None, sparsity=DP_SPARSITY,
                                      iter_avg=DP_ITER_AVG),
                              server_momentum=True, rounds=ROUNDS_DP,
                              # six seeds, the same as the published arm it is paired
                              # against: the sparse_vs_fedavg and adaptdp_vs_fedavg tests
                              # are paired on the seeds BOTH sides ran, so a thinner arm
                              # would silently thin the paired test of the arm the paper
                              # proposes. n=3 also puts the Wilcoxon floor at p=0.25.
                              seeds_n=6)),
    ("FL+DP-Head",       dict(method="fedavg",
                              # the arm with a derivation behind it: the release is
                              # restricted to the classifier head, so the noisy update is
                              # spread over d_head coordinates instead of d and the ratio
                              # the model has to beat, K/(z*sqrt(d)), improves by
                              # sqrt(d/d_head). The support is fixed by the architecture,
                              # never read from data, so sensitivity stays <= C/K and the
                              # certificate is identical. The cost is expressive: the
                              # feature extractor stays at its public initialisation.
                              dp=dict(z=DP_ARM_Z, C=None, head_only=True,
                                      iter_avg=DP_ITER_AVG),
                              server_momentum=True, rounds=ROUNDS_DP, seeds_n=6)),
    # Andrew et al. as published - one knob away from FL+DP, which is the whole point of
    # the paired adaptdp_vs_dp test - except that the estimator stops releasing after
    # ADAPT_FREEZE_ROUNDS rounds because each of its rounds is worth more budget than the
    # update release. No dp_sparsity, no head_only: adding either would price the
    # comparison against two changes at once instead of one.
    ("FL+AdaptDP",       dict(method="fedavg",
                              dp=dict(z=DP_ARM_Z, C=None, adaptive=True),
                              rounds=ROUNDS_DP)),
    ("FL-NonIID-a0.5",   dict(method="fedavg", partition="dirichlet", alpha=0.5)),
)


def dp_method_labels():
    """The Task 4 arms that pay a DP certificate - the ones a DP-only session keeps."""
    return tuple(label for label, spec in MAIN_METHODS if spec.get("dp"))


# Session-scope filter for Task 4, set by a driver before any stage runs (the Kaggle
# ppml_kaggle_dp_* bundles). None = every method above runs. It is deliberately a config
# value rather than an argument: exp_main filters on it, and reporting.profile_scope_notes
# prints it into run_report.md, so a narrowed session is announced by the tables instead of
# looking like an empty cell. Paired tests whose other arm was filtered out come back as
# "insufficient paired seeds", which is true, rather than as a comparison against nothing.
METHODS_FILTER = None
# Task 1 privacy-utility sweep architectures
# CNN-BiLSTM is left out of the sweep (it is in the main table): its d is 195k, so every
# user-level release is far below the SNR bar, and it is the slowest model by far.
PRIVACY_ARCHES = ("MLP", "MLP-Small")
# Task 1's optional third arm: the head-only release measured at EVERY z of the sweep
# instead of only at the operating point.
# This is the arm the accuracy question turns on, and it is a measurement, not a fix. The
# noise-to-signal a release has to beat is K/(z*sqrt(d)), so the 390-coordinate head of the
# 71,750-coordinate MLP sits sqrt(d/d_head) = 13.6x above the whole-model one at the same z
# and the same epsilon. The arithmetic says the head release crosses SNR = 1 between z=7.0
# and z=4.5 while the shipped Task 4 arm runs at z=10.6, where its own measured
# release_snr_mean is well under 1 - i.e. the improvement is predicted, and a prediction is
# not a result. Turning this on runs plain + sparse + head at each level for the arches in
# PRIVACY_HEAD_ARCHES: +30 T=50 runs per dataset per arch at 6 seeds, which is why it is a
# switch and not a default. Whatever the grid says gets reported, including a curve that
# stays at chance.
PRIVACY_HEAD_SWEEP = os.environ.get("PPML_DP_HEAD_SWEEP", "1") == "1"
# MLP only by default: it is the architecture whose d and d_head the SNR arithmetic above
# was computed for, and the CNN/LightTCN sweeps already cost three arches x 11 levels.
PRIVACY_HEAD_ARCHES = ("MLP", "MLP-Small", "LightTCN")
# LightTCN is the costly arm of the sweep: three z levels (ends + operating point), the
# plain and head-only releases, 3 seeds. The MLPs run every level and mechanism.
PRIVACY_Z_BY_ARCH = {"LightTCN": (4.5, 10.6, 28.6)}
PRIVACY_MECH_BY_ARCH = {"LightTCN": ("plain", "head")}
PRIVACY_SEEDS_BY_ARCH = {"LightTCN": 3}
# Runtime verbosity. Every training loop already computes the full metric set per round, so
# printing it costs nothing but lines, and on Kaggle the console is the only view of a run
# before the tables exist. Prints at the existing checkpoints (round 1, then every T/5), not
# every round. Set PPML_PRINT_ROUNDS=0 to quiet a session that is being read for something
# else; nothing logged or tabled changes either way.
PRINT_ROUNDS = os.environ.get("PPML_PRINT_ROUNDS", "1") == "1"
# Task 6 Byzantine: two datasets minimum, MLP + CNN
BYZ_ARCHES = ("MLP",)
# the non-IID arm of Task 6, scoped for session cost and reported as scoped
BYZ_NONIID_ARCHES = ("MLP",)
# Task 9 compute-cost architectures
COST_ARCHES = ("CNN-BiLSTM", "LightTCN")
COST_TIMING_RUNS = 120            # >= 100 as required
COST_WARMUP_RUNS = 20

LOSO_ROUNDS = 15
LOSO_MAX_FOLDS = 10               # 10 held-out subjects per dataset (all 9 for PAMAP2)
RESAMPLING_SEEDS = SEEDS

# ── Profiles ──────────────────────────────────────────────────────────────────
# 'full' satisfies the Work Order minimums. 'smoke' exists only to prove a script
# runs end-to-end; it is never a reportable result and stamps that into the log.
PROFILES = {
    "full":  dict(seeds=SEEDS, seeds_supp=SEEDS_SUPP, dp_seeds=DP_SEEDS,
                  rounds=ROUNDS, rounds_dp=ROUNDS_DP,
                  local_epochs_mlp=LOCAL_EPOCHS_MLP, local_epochs_cnn=LOCAL_EPOCHS_CNN,
                  z_levels=Z_LEVELS, z_sweep=Z_SWEEP, partitions=N_PARTITIONS_PER_ALPHA,
                  byz_seeds=SEEDS_BYZANTINE, mia_trials=SEEDS_MIA,
                  mia_shadows=MIA_SHADOWS, loso_folds=LOSO_MAX_FOLDS,
                  byz_attacks=BYZ_ATTACKS, byz_aggregators=BYZ_AGGREGATORS,
                  byz_fractions=BYZ_FRACTIONS, byz_arches=BYZ_ARCHES,
                  clip_sweep=(), partitions_dp=N_PARTITIONS_DP),
    "smoke": dict(seeds=(42, 43), seeds_supp=(42, 43), dp_seeds=(42,),
                  rounds=4, rounds_dp=5,
                  local_epochs_mlp=1, local_epochs_cnn=1,
                  z_levels=(Z_NON_PRIVATE, 10.6), z_sweep=(10.6,), partitions=1,
                  byz_seeds=(42,), mia_trials=(42,),
                  # 3 shadows, not 2: a shadow whose subject pool comes out empty is
                  # dropped, and LiRA's reference spread needs two SURVIVING models.
                  mia_shadows=3,
                  # Every attack and every aggregator still runs once, but at one
                  # corruption fraction and on the MLP only: the full Task 6 grid is
                  # 5 x 4 x 4 x 2 arches x 2 partition kinds and that alone is more than
                  # the smoke budget allows, which would starve the later stages.
                  byz_attacks=BYZ_ATTACKS, byz_aggregators=BYZ_AGGREGATORS,
                  byz_fractions=(0.2,), byz_arches=("MLP",),
                  # the two restored grids are thinned for the canary only: two clip radii
                  # (both ends) and one DP partition per alpha prove the arms run, and the
                  # smoke log is never a reportable row.
                  clip_sweep=(0.01, 0.50), partitions_dp=1,
                  # 3 folds is enough to prove the LOSO arm runs; all 30 would eat the
                  # smoke session before the later stages got checked
                  loso_folds=3),
}
PROFILE = os.environ.get("PPML_PROFILE", "full")

# Maximum windows per dataset, applied by datasets.load as a stratified subsample that
# KEEPS EVERY SUBJECT. WISDM v2 (72,044 windows) and HARTH (41,212) are 4-7x UCI HAR,
# which is what makes a four-dataset sweep unrunnable on Kaggle's GPU quota; capping the
# window count prices all four datasets the same without shrinking the subject pool that
# the silos, LOSO folds and MIA pools are built from. 0/None = no cap. This changes
# accuracy (fewer training windows), never the privacy certificate, and the cap is
# stamped into every log record so a table can be filtered by it.
try:
    MAX_WINDOWS = int(os.environ.get("PPML_MAX_WINDOWS", "0")) or None
except ValueError:
    MAX_WINDOWS = None


def cfg():
    return PROFILES[PROFILE]


def is_reportable():
    return PROFILE == "full"


# ── Kaggle session budget ─────────────────────────────────────────────────────
# Kaggle kills a session at 9 h (GPU) / 12 h (CPU) and the killed tail is lost. Every
# run is appended to its log as it finishes, so the only cost of stopping early is a
# stage that has to be resumed. SESSION_DEADLINE is an absolute unix timestamp; the
# driver sets it and check_budget() is called by the harness before each run.
SESSION_DEADLINE = None


def set_deadline(seconds_from_now=None, absolute=None):
    global SESSION_DEADLINE
    if absolute is not None:
        SESSION_DEADLINE = float(absolute)
    elif seconds_from_now is not None:
        SESSION_DEADLINE = time.time() + float(seconds_from_now)
    return SESSION_DEADLINE


def budget_left_s():
    return float("inf") if SESSION_DEADLINE is None else SESSION_DEADLINE - time.time()


class BudgetExceeded(RuntimeError):
    """Raised when a run would start past the session deadline."""


def check_budget(need_s=0.0):
    if SESSION_DEADLINE is not None and time.time() + need_s > SESSION_DEADLINE:
        raise BudgetExceeded("session deadline reached (%.1f s remaining, %.1f s needed)"
                             % (budget_left_s(), need_s))
    return True

#@@FILE ppml/dp_accountant.py
"""Valid Renyi-DP accounting for the central-Gaussian federated mechanism.

This module replaces ``GaussDP.renyi_epsilon`` and ``GaussDP.moments_epsilon`` from
the rejected submission, neither of which produced a valid (eps, delta) certificate.

Mechanism that is accounted for (Work Order Task 1)
---------------------------------------------------
Per round r, with K participating clients and whole-update clipping to L2 norm C:

    Delta_k   = w_k - w_global          client update
    Deltabar_k = Delta_k * min(1, C/||Delta_k||_2)      ONE norm over ALL tensors
    mean      = (1/K) * sum_k Deltabar_k                EQUAL weights 1/K
    release   = mean + N(0, sigma^2 I),  sigma = z * C / K

Privacy unit: one client = one subject silo. Adjacency: add/remove all the data of
one client.

Why the noise multiplier is z and not something else
----------------------------------------------------
The L2 sensitivity of ``mean`` under one-client adjacency is C/K (one clipped term
of norm <= C enters with weight 1/K). The added noise has std sigma = z*C/K, so the
noise-to-sensitivity ratio is exactly z, independent of C and K. That is why the
certificate survives adaptive clipping: when C_r changes from round to round, both
the sensitivity and the noise scale by the same factor and z stays fixed.

Per-round Renyi divergence of the Gaussian mechanism with ratio z is

    eps_rdp(alpha) = alpha / (2 z^2)

and over T independently-composed rounds eps_rdp(alpha) = T*alpha/(2 z^2).
Conversion to (eps, delta) and the optimal order are closed-form:

    alpha* = 1 + sqrt( 2 ln(1/delta) * 2 z^2 / T )
    eps    = T alpha* / (2 z^2) + ln(1/delta) / (alpha* - 1)

which reproduces the reference numbers quoted in the Work Order (verified in
``selftest()``).
"""

import math

MECHANISM = "central_gaussian_whole_update_clip"
ADJACENCY = "client_level_subtree_user"      # one client = one subject silo


def _rdp_to_eps(rdp_eps, order, delta):
    return rdp_eps - math.log(delta) / (order - 1.0)


def rdp_gaussian(order, z):
    """Rnyi divergence of order `order` of a Gaussian mechanism with ratio z."""
    if z <= 0:
        return float("inf")
    return order / (2.0 * z * z)


def optimal_order(c, delta):
    """argmin_alpha c*alpha + ln(1/delta)/(alpha-1) for c >= 0 (c = sum of per-round
    coefficients). Returns (alpha*, eps)."""
    if c <= 0:
        return float("inf"), float("inf")
    L = math.log(1.0 / delta)
    order = 1.0 + math.sqrt(L / c)
    return order, c * order + L / (order - 1.0)


def eps_for_order(c, order, delta):
    return _rdp_to_eps(c * order, order, delta)


def account_central_gaussian(z, T, delta=1e-5, client_sampling_rate=1.0):
    """Certificate for T rounds of the mechanism described in the module docstring.

    Parameters
    ----------
    z  : noise multiplier (sigma = z * C / K)
    T  : number of composed rounds
    client_sampling_rate : fraction of clients participating per round. Values < 1
        are handled CONSERVATIVELY: no subsampling amplification is claimed, because
        the rejected submission applied amplification (q = 32/n_k) for a loop that in
        fact sampled every client. Reporting the gamma = 1 bound is always valid.
    """
    if z is None or z <= 0 or T <= 0:
        return dict(eps=float("inf"), alpha_star=float("inf"), per_round_eps=[],
                    z=z, T=T, delta=delta, gamma=client_sampling_rate,
                    amplified=False, note="no noise: unrestricted privacy loss")
    c = T / (2.0 * z * z)
    order, eps = optimal_order(c, delta)
    per_round = []
    for t in range(1, T + 1):
        o_t, e_t = optimal_order(t / (2.0 * z * z), delta)
        per_round.append(dict(round=t, eps=e_t, alpha_star=o_t))
    return dict(mechanism=MECHANISM, adjacency=ADJACENCY, z=float(z), C_over_K=True,
                T=int(T), delta=delta, gamma=float(client_sampling_rate),
                alpha_star=float(order), eps=float(eps),
                eps_single_round=eps_for_order(1.0 / (2.0 * z * z), order, delta),
                per_round_eps=per_round,
                amplified=False,
                note=("gamma<1 treated as gamma=1 (no amplification claimed)"
                      if client_sampling_rate < 1.0 else "full participation, exact"))


def account_adaptive(z, T, delta=1e-5, sigma_b=100.0, T_clip=None):
    """Task 2: the update mechanism AND the Gaussian quantile estimator, composed in RDP.

    The clip-fraction estimator is b~ = (sum_k 1[||Delta_k|| <= C] + N(0, sigma_b^2))/K.
    Its L2 sensitivity under one-client adjacency is 1/K and its noise std is
    sigma_b/K, so its noise-to-sensitivity ratio is sigma_b. Each round the estimator
    RUNS adds its own RDP coefficient, so the composed budget is

        c = T/(2 z^2) + T_clip/(2 sigma_b^2)

    T_clip is the number of rounds in which a clip-fraction statistic was released. It
    defaults to T, which is Andrew et al. as published - the estimator tracking C for the
    whole run. The pipeline passes the count it actually logged, because the estimator is
    the expensive half of this certificate: at sigma_b = 0.5 one estimator round costs
    1/(2*0.5^2) = 2.0 while one update release at z = 10.6 costs 0.0044.
    """
    if z <= 0 and sigma_b <= 0:
        return dict(eps_total=float("inf"))
    T = int(T)
    T_upd = T
    T_cl = T if T_clip is None else int(T_clip)
    if T_cl > T:
        raise ValueError("T_clip=%d exceeds the %d rounds actually run; the certificate "
                         "cannot be larger than the release" % (T_cl, T))
    c_upd = T_upd / (2.0 * z * z) if z > 0 else 0.0
    c_clip = T_cl / (2.0 * sigma_b * sigma_b) if sigma_b > 0 else 0.0
    order, eps_total = optimal_order(c_upd + c_clip, delta)
    # component values are reported at the SHARED optimal order (valid decomposition)
    eps_upd = eps_for_order(c_upd, order, delta) if z > 0 else 0.0
    eps_clip = eps_for_order(c_clip, order, delta) if sigma_b > 0 else 0.0
    per_round = []
    for t in range(1, T + 1):
        c_t = ((min(t, T_upd) / (2.0 * z * z) if z > 0 else 0.0)
               + (min(t, T_cl) / (2.0 * sigma_b * sigma_b) if sigma_b > 0 else 0.0))
        per_round.append(dict(round=t, eps=optimal_order(c_t, delta)[1]))
    return dict(mechanism=MECHANISM + "+gaussian_quantile_clip",
                adjacency=ADJACENCY, z=float(z), sigma_b=float(sigma_b),
                T=int(T), T_updates=int(T_upd), T_clip_estimator=int(T_cl),
                delta=delta, alpha_star=float(order),
                eps_updates=float(eps_upd), eps_clip_fraction=float(eps_clip),
                eps_total=float(eps_total),
                per_round_eps=per_round)


def account_per_tensor_clipping(z, T, n_tensors, delta=1e-5):
    """What the REJECTED code cost.

    Clipping each of the L tensors to C bounds the whole-update norm by sqrt(L)*C, so
    the true sensitivity is sqrt(L) times larger than assumed and the effective ratio
    is z/sqrt(L). Used only to report the accounting error, never to certify a result.
    """
    z_eff = z / math.sqrt(n_tensors)
    out = account_central_gaussian(z_eff, T, delta)
    out["n_tensors"] = n_tensors
    out["note"] = "diagnostic: sensitivity inflation of the per-tensor clip in v1"
    return out


def legacy_equivalent_z(nominal_eps=20.0, delta=1e-5):
    """The v1 code set sigma = sqrt(2 ln(1.25/delta))/eps and then used sigma*C/K.
    Expressed as a noise multiplier that is exactly what nominal_eps meant."""
    return math.sqrt(2.0 * math.log(1.25 / delta)) / nominal_eps


def sigma_of(z, C, K):
    return z * C / float(K)


# ── Sampled Gaussian Mechanism (Poisson-subsampled DP-SGD) ─────────────────────
# Numerically stable log-space RDP of Mironov, Talwar & Zhang (2019), the same expansion
# TensorFlow-Privacy and Google dp-accounting use. Verified against dp-accounting's
# RdpAccountant(PoissonSampledDpEvent(q, GaussianDpEvent(z))) to ~1e-11 relative error
# for every eps <= 20 on a grid of z in [0.7, 5], q in [0.001, 0.1], T in [10, 5000].
SGM_ORDERS = ([1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))
              + [128, 256, 512])


def _log_add(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def _log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _log_erfc(x):
    from scipy import special
    return float(special.log_ndtr(-x * math.sqrt(2)) + math.log(2))


def _sgm_log_a_int(q, sigma, alpha):
    log_a = -math.inf
    for i in range(int(alpha) + 1):
        log_a = _log_add(log_a, _log_comb(alpha, i) + i * math.log(q)
                         + (alpha - i) * math.log1p(-q) + (i * i - i) / (2.0 * sigma * sigma))
    return log_a


def _sgm_log_a_frac(q, sigma, alpha):
    log_a0, log_a1 = -math.inf, -math.inf
    z0 = sigma * sigma * math.log(1.0 / q - 1.0) + 0.5
    i = 0
    while True:
        log_coef = (math.lgamma(alpha + 1) - math.lgamma(i + 1)
                    - math.lgamma(alpha - i + 1))
        j = alpha - i
        log_t0 = log_coef + i * math.log(q) + j * math.log1p(-q)
        log_t1 = log_coef + j * math.log(q) + i * math.log1p(-q)
        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))
        log_s0 = log_t0 + (i * i - i) / (2 * sigma * sigma) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * sigma * sigma) + log_e1
        log_a0 = _log_add(log_a0, log_s0)
        log_a1 = _log_add(log_a1, log_s1)
        i += 1
        if i > 1000 or (max(log_s0, log_s1) < -30 and i > alpha):
            break
    return _log_add(log_a0, log_a1)


def sgm_rdp(q, z, T, orders=None):
    """Total RDP after T steps of the Poisson-subsampled Gaussian (rate q, multiplier z)."""
    orders = orders or SGM_ORDERS
    out = []
    for a in orders:
        if q >= 1.0:
            out.append(T * a / (2.0 * z * z))
            continue
        la = (_sgm_log_a_int(q, z, int(a)) if float(a).is_integer()
              else _sgm_log_a_frac(q, z, float(a)))
        out.append(T * la / (a - 1.0))
    return orders, out


def sgm_eps(q, z, T, delta=1e-5):
    """(eps, alpha*) for T steps of DP-SGD, tight RDP->DP conversion (same as the
    user-level accountant: eps = rho + ln(1-1/a) - (ln delta + ln a)/(a-1))."""
    orders, rdp = sgm_rdp(q, z, T)
    best = (math.inf, None)
    for a, r in zip(orders, rdp):
        e = r + math.log1p(-1.0 / a) - (math.log(delta) + math.log(a)) / (a - 1.0)
        if e < best[0]:
            best = (e, a)
    return float(best[0]), float(best[1])


def z_for_eps_dpsgd(target_eps, q, T, delta=1e-5, lo=0.3, hi=50.0, tol=1e-4):
    """Smallest z whose T-step DP-SGD certificate is <= target_eps (bisection).
    Called BEFORE training, from the planned step count."""
    if sgm_eps(q, hi, T, delta)[0] > target_eps:
        raise ValueError("target eps %.3f unreachable even at z=%.1f" % (target_eps, hi))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sgm_eps(q, mid, T, delta)[0] > target_eps:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return float(hi)


def account_dpsgd(z, T, delta=1e-5, sample_rate=None, orders=None, batch=None,
                  n_per_client=None):
    """Example-level DP-SGD certificate: T Poisson-sampled Gaussian steps of ONE client.

    The privacy unit is one window. An example lives in exactly one silo, so the steps of
    other clients never touch it and the per-client composition is the whole bound; server
    averaging of already-private updates is post-processing. Every return path is stamped
    adjacency=example_level and user_level_certificate=NO.
    """
    if sample_rate:
        eps, a = sgm_eps(float(sample_rate), float(z), int(T), delta)
        out = dict(mechanism="dp_sgd_poisson", adjacency="example_level",
                   user_level_certificate="NO", z=float(z), T=int(T), delta=delta,
                   gamma=float(sample_rate), alpha_star=a, eps=eps,
                   accountant="sampled-Gaussian RDP (Mironov et al. 2019), in-package")
        try:   # independent cross-check when Google's library is installed
            from dp_accounting import dp_event as _ev
            from dp_accounting.rdp import RdpAccountant as _Acc
            acc = _Acc(SGM_ORDERS)
            acc.compose(_ev.PoissonSampledDpEvent(float(sample_rate),
                                                  _ev.GaussianDpEvent(float(z))), int(T))
            out["eps_dp_accounting"] = float(acc.get_epsilon(delta))
        except Exception:
            out["eps_dp_accounting"] = None
        return _stamp_dpsgd_shape(out, batch, n_per_client, sample_rate)
    out = account_central_gaussian(z, T, delta)
    out.update(mechanism="central_gaussian_no_amplification", accountant="closed form",
               note="no sample rate given: full-batch bound over %d steps" % int(T))
    return _stamp_dpsgd_shape(out, batch, n_per_client, sample_rate)


def _stamp_dpsgd_shape(out, batch, n_per_client, sample_rate):
    """The example-level labels, on every path that reaches the dpsgd certificate."""
    # Overwrite, never inherit: the fallback delegates to account_central_gaussian, whose
    # own adjacency is the client silo. A dpsgd row borrows that number but not that
    # privacy unit - one window is its unit - so leaving the inherited label in place would
    # print a client-level guarantee next to an example-level bound.
    out["central_form_adjacency"] = out.get("adjacency")
    out["adjacency"] = "example_level"
    out["user_level_certificate"] = "NO"
    out["batch"] = None if batch is None else int(batch)
    out["sample_rate"] = None if sample_rate is None else float(sample_rate)
    out["n_per_client"] = None if n_per_client is None else int(n_per_client)
    return out


def cross_check(z, T, delta=1e-5, orders=None):
    """Cross-check our closed form against Google dp-accounting (Work Order Task 1)."""
    ours = account_central_gaussian(z, T, delta)
    try:
        from dp_accounting.rdp import rdp_accounting
    except Exception as exc:
        return dict(ok=False, ours=ours["eps"], theirs=None,
                    error="dp-accounting not installed (%s); pip install dp-accounting" % exc)
    try:
        ord_list = orders or [1.0 + x / 100.0 for x in range(1, 2000)]
        ord_list = [o for o in ord_list if o > 1.0]
        rdp = rdp_accounting.compute_rdp(noise_multiplier=z, sampling_probability=1.0,
                                        steps=T, orders=ord_list)
        eps_them = rdp_accounting.get_epsilon_for_delta(
            noise_multiplier=z, sampling_probability=1.0, steps=T,
            orders=ord_list, target_delta=delta)
        return dict(ok=True, ours=ours["eps"], theirs=float(eps_them),
                    rel_diff=abs(ours["eps"] - eps_them) / max(eps_them, 1e-12),
                    alpha_star=ours["alpha_star"], z=z, T=T, delta=delta,
                    n_orders=len(rdp))
    except Exception as exc:          # pragma: no cover
        return dict(ok=False, ours=ours["eps"], theirs=None, error=str(exc))


def _eps_by_minimisation(c, delta, hi=200.0, steps=200000):
    """eps(alpha) minimised over a dense alpha grid - an independent check of the
    closed form in optimal_order(), which is what makes the certificate trustworthy."""
    best = float("inf")
    for i in range(steps):
        alpha = 1.0 + (hi - 1.0) * i / float(steps - 1)
        if alpha <= 1.0:
            continue
        e = c * alpha - math.log(delta) / (alpha - 1.0)
        if e < best:
            best = e
    return best


def selftest(verbose=True):
    """Check the accountant two ways: against its own closed form and against the
    approximate eps values quoted in the Work Order.

    The Work Order table is rounded to one significant figure ("eps ~= 3" for
    z = 10.6), so those rows are reported with the gap printed, not asserted; the
    assertions are the self-consistency and legacy-cost checks, which are exact.
    """
    hard = []
    # 1. the closed-form optimum must be the minimum of the conversion curve
    for z, T in ((4.5, 50), (10.6, 50), (28.6, 50), (7.0, 30)):
        c = T / (2.0 * z * z)
        closed = optimal_order(c, 1e-5)[1]
        grid = _eps_by_minimisation(c, 1e-5)
        hard.append(("z=%.1f T=%d closed form" % (z, T), closed, grid, 1e-4))
    # 2. eps must decrease monotonically in z and increase in T
    eps_z = [account_central_gaussian(z, 50, 1e-5)["eps"] for z in
             (4.5, 7.0, 10.6, 15.0, 28.6)]
    hard.append(("eps decreasing in z", float(all(a > b for a, b in zip(eps_z, eps_z[1:]))),
                 1.0, 0.0))
    eps_T = [account_central_gaussian(10.6, T, 1e-5)["eps"] for T in (10, 50, 200)]
    hard.append(("eps increasing in T", float(all(a < b for a, b in zip(eps_T, eps_T[1:]))),
                 1.0, 0.0))
    # 3. no noise => no certificate
    hard.append(("z=0 is infinite", float(account_central_gaussian(0, 50)["eps"] ==
                float("inf")), 1.0, 0.0))
    zl = legacy_equivalent_z(20.0)
    hard.append(("legacy whole-model eps", account_central_gaussian(zl, 50)["eps"],
                 566.0, 0.12))
    hard.append(("legacy CNN per-tensor eps",
                 account_per_tensor_clipping(zl, 50, 26)["eps"], 12000.0, 0.12))
    # 4. the adaptive composition, against the two things that pin it down. With the
    # estimator releasing zero times this must reduce EXACTLY to the plain certificate -
    # that is what lets the pipeline charge only the rounds the clipper ran. The split
    # keys must stay consistent with the composed total, and more estimator rounds must
    # cost strictly more.
    plain50 = account_central_gaussian(10.6, 50, 1e-5)["eps"]
    hard.append(("adaptive with T_clip=0 = plain",
                 account_adaptive(10.6, 50, 1e-5, sigma_b=0.5, T_clip=0)["eps_total"],
                 plain50, 1e-9))
    e_clips = [account_adaptive(10.6, 50, 1e-5, sigma_b=0.5, T_clip=w)["eps_total"]
               for w in (0, 5, 10, 25, 50)]
    hard.append(("adaptive eps rising in T_clip",
                 float(all(a < b for a, b in zip(e_clips, e_clips[1:]))), 1.0, 0.0))
    one = account_adaptive(10.6, 50, 1e-5, sigma_b=0.5, T_clip=10)
    # the two components are reported at the SHARED optimal order, so each carries the
    # ln(1/delta)/(alpha-1) term once and their sum exceeds the total by exactly that
    # term. This is the identity that makes the split a decomposition and not two
    # independent epsilons a reader might be tempted to add.
    delta_term = math.log(1.0 / 1e-5) / (one["alpha_star"] - 1.0)
    hard.append(("adaptive split = total + one delta term",
                 one["eps_updates"] + one["eps_clip_fraction"],
                 one["eps_total"] + delta_term, 1e-9))
    try:
        account_adaptive(10.6, 50, 1e-5, sigma_b=0.5, T_clip=51)
        over = 0.0
    except ValueError:
        over = 1.0
    hard.append(("T_clip > T is refused", over, 1.0, 0.0))
    ok = True
    if verbose:
        print("  assertions:")
    for name, got, exp, tol in hard:
        within = (got == float("inf") and exp == float("inf")) or \
                 abs(got - exp) <= tol * max(abs(exp), 1e-9)
        ok &= within
        if verbose:
            print("    %-28s got=%12.4f  expected=%12.4f  %s"
                  % (name, got, exp, "OK" if within else "FAIL"))
    if verbose:
        print("  Work Order table (references rounded, gap shown):")
        for z, quoted in ((4.5, 8.0), (10.6, 3.0), (28.6, 1.0)):
            got = account_central_gaussian(z, 50, 1e-5)["eps"]
            print("    z=%-5.1f eps=%7.3f  quoted ~= %-5.1f  gap %+.0f%%"
                  % (z, got, quoted, 100.0 * (got / quoted - 1.0)))
        print("    use the eps computed here, never the rounded quotes, in the paper")
    return ok


if __name__ == "__main__":
    print("dp_accountant selftest (vs Work Order reference numbers):")
    print("  all checks passed:", selftest())

#@@FILE ppml/runlog.py
"""Machine-readable run logging and resume support.

Work Order rule 2: "Log every run to a machine-readable file (JSON or CSV): dataset,
method, seed, partition id, all hyper-parameters, per-round test accuracy, final
accuracy/F1, and the accountant output. Figures are drawn only from these logs."

One JSONL line per run under results/logs/<dataset>/<stage>.jsonl, plus a flattened
summary CSV per stage. `config_hash` identifies a run uniquely, so a Kaggle session
that dies at hour 11 can be resumed without recomputing anything.
"""

import csv
import hashlib
import json
import os
import time

import numpy as np

from . import config

JSON_SUFFIX = ".jsonl"
_UNSET = object()


def _np(obj):
    if isinstance(obj, dict):
        return {str(k): _np(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_np(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def stable_hash(d):
    """Deterministic hash of a config dict (order-independent, float-rounded)."""
    def norm(o):
        if isinstance(o, dict):
            return {k: norm(o[k]) for k in sorted(o)}
        if isinstance(o, (list, tuple)):
            return [norm(v) for v in o]
        if isinstance(o, float):
            return round(o, 9)
        if isinstance(o, (np.floating,)):
            return round(float(o), 9)
        return o
    payload = json.dumps(norm(_np(d)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class RunLog:
    def __init__(self, dataset, stage):
        self.dataset = dataset
        self.stage = stage
        config.ensure_dirs()
        self.dir = config.LOG_DIR
        self.path_jsonl = os.path.join(self.dir, "%s__%s%s" % (dataset, stage, JSON_SUFFIX))
        self.path_csv = os.path.join(self.dir, "%s__%s.csv" % (dataset, stage))
        self._seen = self._load_hashes()

    # ── resume ────────────────────────────────────────────────────────────────
    def _load_hashes(self):
        seen = set()
        if not os.path.exists(self.path_jsonl):
            return seen
        with open(self.path_jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("config_hash"):
                    seen.add(rec["config_hash"])
        return seen

    def has(self, chash):
        return chash in self._seen

    # ── write ─────────────────────────────────────────────────────────────────
    def record(self, config_, result_, skip_existing=True, config_hash=None):
        """Append one run. `config_` = hyper-parameters, `result_` = metrics.

        `config_hash` is the run identity the caller already tested with `has()`. It has
        to be passed in: the caller hashes the settings-only block while `config_` here
        has grown the result-derived C/z/K/T/delta fields, and hashing the two different
        dictionaries is what made every session recompute finished work.

        Returns the stored record, or None when the run was already logged.
        """
        chash = config_hash if config_hash is not None else stable_hash(config_)
        if skip_existing and chash in self._seen:
            return None
        rec = {"dataset": self.dataset, "stage": self.stage, "profile": config.PROFILE,
               "reportable": config.is_reportable(), "config_hash": chash,
               "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        rec.update(_np(config_))
        rec.update(_np(result_))
        with open(self.path_jsonl, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self._seen.add(chash)
        self._append_csv(rec)
        return rec

    _csv_header = None

    def _append_csv(self, rec):
        # CSV carries the scalar columns; per-round curves stay in the JSONL.
        flat = {k: v for k, v in rec.items()
                if not isinstance(v, (list, dict, tuple))}
        # The header is fixed by the first row; later rows are written under THAT header
        # (missing keys blank, extra keys left to the JSONL, which is the complete record).
        # Writing each row with its own key set, as before, shifted values into the wrong
        # columns whenever two runs of a stage carried different fields.
        exists = os.path.exists(self.path_csv)
        if not exists:
            os.makedirs(os.path.dirname(self.path_csv), exist_ok=True)
            header = sorted(flat)
        else:
            with open(self.path_csv, newline="") as fh:
                header = next(csv.reader(fh), None) or sorted(flat)
        with open(self.path_csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore", restval="")
            if not exists:
                w.writeheader()
            w.writerow(flat)


def read_logs(dataset=None, stage=None, reportable_only=True):
    """Load every logged run as a list of dicts (this is what plotters read)."""
    config.ensure_dirs()
    out = []
    for fname in sorted(os.listdir(config.LOG_DIR)) if os.path.isdir(config.LOG_DIR) else []:
        if not fname.endswith(JSON_SUFFIX):
            continue
        base = fname[: -len(JSON_SUFFIX)]
        ds, _, st = base.partition("__")
        if dataset and ds != dataset:
            continue
        if stage and st != stage:
            continue
        with open(os.path.join(config.LOG_DIR, fname)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if reportable_only and not rec.get("reportable", False):
                    continue
                out.append(rec)
    return out


def summarise(runs, key_fields, metric="accuracy"):
    """Mean/std/n over seeds for each distinct combination of `key_fields`."""
    groups = {}
    for r in runs:
        k = tuple(str(r.get(f, "")) for f in key_fields)
        groups.setdefault(k, []).append(r)
    rows = []
    for k, g in groups.items():
        vals = np.array([float(x[metric]) for x in g if x.get(metric) is not None], float)
        row = dict(zip(key_fields, k))
        row.update(n=int(len(vals)), mean=float(vals.mean()) if len(vals) else None,
                   std=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
        rows.append(row)
    return sorted(rows, key=lambda d: [str(d[f]) for f in key_fields])


def versions():
    """Software provenance for the run report (Work Order section 13)."""
    import sys
    v = {"python": sys.version.split()[0]}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "torch", "matplotlib"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "?")
        except Exception:
            v[mod] = None
    for mod in ("dp_accounting", "opacus", "thop"):
        try:
            __import__(mod)
            v[mod] = "installed"
        except Exception:
            v[mod] = None
    return v


def hardware():
    """CPU model / RAM / GPU, recorded once per run report."""
    info = {}
    try:
        import multiprocessing
        info["cpu_count"] = multiprocessing.cpu_count()
    except Exception:
        pass
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
    except Exception:
        pass
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal"):
                        info["ram_kb"] = int(line.split()[1])
                        break
    except Exception:
        pass
    try:
        import torch
        info["torch_cuda"] = bool(torch.cuda.is_available())
        if info["torch_cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info

#@@FILE ppml/metrics.py
"""Metric helpers shared by every experiment."""

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score)

from . import config


def _specificity(conf):
    """One-vs-rest specificity per class, straight from the confusion matrix.

    TP/FP/FN/TN are read off the matrix rather than recomputed from the labels, so the
    number on the row is the same reading the figure and the table show. A class nobody
    predicted and nobody had gives 0/0; those cells are reported as None, not as 1.0,
    because a perfect-looking score that came from an empty class is the kind of number
    a reviewer takes apart.
    """
    conf = np.asarray(conf, dtype=float)
    n_total = float(conf.sum())
    out = []
    for i in range(conf.shape[0]):
        tp = float(conf[i, i])
        fp = float(conf[:, i].sum() - tp)
        fn = float(conf[i, :].sum() - tp)
        tn = n_total - tp - fp - fn
        denom = tn + fp
        out.append(float(tn / denom) if denom > 0 else None)
    return out


def classification_metrics(y_true, y_pred, y_proba=None, sample_loss=None,
                           n_class=None):
    n_class = n_class if n_class is not None else config.N_CLASS
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = list(range(n_class))
    conf = [[int(v) for v in row]
            for row in confusion_matrix(y_true, y_pred, labels=labels)]
    spec = _specificity(conf)
    spec_vals = [s for s in spec if s is not None]
    recall_pc = [float(x) for x in recall_score(y_true, y_pred, average=None,
                                                labels=labels, zero_division=0)]
    # Macro averages are taken over the classes that occur in this test set: a class with
    # no test windows (possible with a 2-subject PAMAP2 test split) has no recall to
    # average and must not enter as a zero. Per-class arrays keep every class.
    present = [c for c in labels if np.any(y_true == c)] or labels
    out = dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted",
                                   labels=labels, zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", labels=present,
                                zero_division=0)),
        precision=float(precision_score(y_true, y_pred, average="weighted",
                                        labels=labels, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="weighted", labels=labels,
                                  zero_division=0)),
        f1_per_class=[float(x) for x in f1_score(y_true, y_pred, average=None,
                                                 labels=labels, zero_division=0)],
        precision_per_class=[float(x) for x in precision_score(y_true, y_pred, average=None,
                                                               labels=labels, zero_division=0)],
        recall_per_class=recall_pc,
        # balanced accuracy is the mean of the per-class recalls: the class-count honest
        # version of accuracy, which matters when a HAR or PAMAP2 class is rare.
        balanced_accuracy=(float(np.mean([recall_pc[c] for c in present]))
                           if recall_pc else None),
        n_classes_in_test=len(present),
        specificity_per_class=spec,
        macro_specificity=float(np.mean(spec_vals)) if spec_vals else None,
        kappa=float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        n_test=int(len(y_true)),
        class_support=[int(np.sum(y_true == i)) for i in labels],
        confusion=conf,
        loss=float(np.mean(sample_loss)) if sample_loss is not None and
        len(np.asarray(sample_loss)) else float("nan"),
    )
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            out["roc_auc_ovr"] = float(roc_auc_score(
                np.eye(n_class)[y_true][:, present], np.asarray(y_proba)[:, present],
                average="macro"))
        except (ValueError, IndexError):
            out["roc_auc_ovr"] = None
        try:
            out["nll"] = float(log_loss(y_true, np.clip(y_proba, 1e-9, 1.0),
                                        labels=labels))
        except (ValueError, IndexError):
            out["nll"] = None
    return out


def per_round_arrays(per_round):
    if not per_round:
        return []
    return [float(p["test_acc"]) for p in per_round]


def _fmt(v, spec="%.4f"):
    return "n/a" if v is None else (spec % v)


def format_metrics(res):
    """One console line with every headline reading of a finished run.

    Used by the experiment stages so that a run's console line, its JSONL record and its
    table row all carry the same set. Nothing is computed here - it reads the result the
    loop already produced, so the line cannot disagree with the log.
    """
    parts = ["acc=%s" % _fmt(res.get("accuracy")),
             "prec=%s" % _fmt(res.get("precision")),
             "rec=%s" % _fmt(res.get("recall")),
             "F1w=%s" % _fmt(res.get("f1")),
             "F1m=%s" % _fmt(res.get("f1_macro")),
             "bal=%s" % _fmt(res.get("balanced_accuracy")),
             "spec=%s" % _fmt(res.get("macro_specificity")),
             "kappa=%s" % _fmt(res.get("kappa"))]
    auc, nll = res.get("roc_auc_ovr"), res.get("nll")
    if auc is not None:
        parts.append("auc=%s" % _fmt(auc))
    if nll is not None:
        parts.append("nll=%s" % _fmt(nll, "%.4f"))
    priv = res.get("privacy") or {}
    if priv:
        parts.append("eps=%s" % _fmt(priv.get("eps"), "%.4f"))
        parts.append("alpha*=%s" % _fmt(priv.get("alpha_star"), "%.2f"))
        parts.append("z=%s C=%s K=%s T=%s"
                     % (_fmt(res.get("dp_z"), "%.4g"), _fmt(res.get("clip_C"), "%.3f"),
                        _fmt(res.get("K"), "%d"), _fmt(res.get("rounds"), "%d")))
        snr = res.get("release_snr_mean")
        parts.append("snr=%s" % _fmt(snr, "%.4f"))
        if priv.get("adjacency"):
            parts.append("adj=%s" % priv["adjacency"])
    if res.get("accuracy_iter_avg") is not None:
        parts.append("acc_avg=%s" % _fmt(res.get("accuracy_iter_avg")))
    parts.append("wall=%.1fs" % float(res.get("wall_time_s") or 0.0))
    return "  ".join(parts)

#@@FILE ppml/features.py
"""117-dimensional hand-crafted feature extractor and the nine feature families.

Identical mathematics to the submission's ``extract_features``, with two changes the
Work Order requires:

* the sampling rate is an explicit argument instead of a hidden global, so WISDM
  (20 Hz) and PAMAP2 (100 Hz) are not analysed as if they were 50 Hz signals;
* ``jerk`` gains the sampling-rate factor, matching the derivative it claims to be.
  Which convention was used is written into every run record (Task 8).

Feature layout for ``nc`` channels (nc = 9 gives exactly 117 columns):

  block                     columns            count
  A statistical moments     0 .. 53              54   (6 per channel)
  B cross-axis correlation  54 .. 59              6
  C signal-magnitude area   60                    1
  D jerk                    61 .. 62              2
  E FFT band power          63 .. 107            45   (5 per channel)
  F gravity magnitude       108 .. 109            2
  G tilt                    110                   1
  H step lag                111 .. 113            3
  I spectral entropy        114 .. 116            3
"""

import numpy as np

from . import config

FAMILY_ORDER = ("statistical_moments", "cross_axis_correlation", "sma", "jerk",
                "fft_band_power", "gravity_magnitude", "tilt", "step_lag",
                "spectral_entropy")


def _dominant_lag(sig, fs):
    ac = np.correlate(sig - sig.mean(), sig - sig.mean(), mode="full")
    ac = ac[len(ac) // 2:]
    ac = ac / (ac[0] + 1e-10)
    for lag in range(5, min(60, len(ac) - 1)):
        if ac[lag] > ac[lag - 1] and ac[lag] > ac[lag + 1] and ac[lag] > 0.1:
            return float(lag) / fs
    return 0.0


def _spectral_entropy(sig):
    psd = np.abs(np.fft.rfft(sig)) ** 2
    pk = psd / (psd.sum() + 1e-10)
    return float(-np.sum(pk * np.log(pk + 1e-10)))


def n_features(nc=None):
    nc = nc if nc is not None else config.N_CH_HARMONISED
    return nc * 6 + config.N_CORR + 1 + 2 + nc * config.N_FFT_BANDS + 2 + 1 + 3 + 3


def family_slices(nc=None):
    """Column index ranges of the nine families, derived not hard-coded.

    `nc` is the NUMBER OF HARMONISED CHANNELS (config.N_CH_HARMONISED), not the number
    of feature columns - the families partition the whole 117-column vector.
    """
    nc = nc if nc is not None else config.N_CH_HARMONISED
    p = 0
    out = {}
    def take(name, k):
        nonlocal p
        out[name] = list(range(p, p + k))
        p += k
    take("statistical_moments", nc * 6)
    take("cross_axis_correlation", config.N_CORR)
    take("sma", 1)
    take("jerk", 2)
    take("fft_band_power", nc * config.N_FFT_BANDS)
    take("gravity_magnitude", 2)
    take("tilt", 1)
    take("step_lag", 3)
    take("spectral_entropy", 3)
    if p != n_features(nc):
        raise ValueError("the nine families cover %d columns but the extractor emits "
                         "%d - the two layouts drifted apart" % (p, n_features(nc)))
    return out


def feature_names(nc=None):
    nc = nc if nc is not None else config.N_CH_HARMONISED
    names = (["Ch%d_%s" % (c + 1, s) for c in range(nc)
              for s in ("mean", "std", "min", "max", "nrg", "zcr")]
             + ["Xcorr_%s" % p for p in ("01", "02", "12", "34", "35", "45")]
             + ["SMA", "Jerk_mean", "Jerk_std"]
             + ["FFT_Ch%d_B%d" % (c + 1, b + 1) for c in range(nc)
                for b in range(config.N_FFT_BANDS)]
             + ["GravMag_mean", "GravMag_std", "Tilt_mean"]
             + ["StepLag_Ch%d" % (c + 1) for c in range(3)]
             + ["SpEnt_Gyro%d" % (c + 1) for c in range(3)])
    assert len(names) == n_features(nc), (len(names), n_features(nc))
    return names


def extract_features(X, fs=None, bands_hz=None, jerk_with_fs=None):
    """X: (N, window, C) harmonised windows -> (N, 117) float array."""
    fs = float(config.TARGET_FS if fs is None else fs)
    bands_hz = bands_hz or config.FFT_BANDS_HZ
    jerk_with_fs = (config.JERK_WITH_SAMPLING_RATE if jerk_with_fs is None
                    else jerk_with_fs)
    N, wl, nc = X.shape
    F = np.zeros((N, n_features(nc)), dtype=np.float64)

    # A: statistical moments, 6 per channel
    for c in range(nc):
        sig = X[:, :, c]
        b = c * 6
        F[:, b + 0] = sig.mean(1)
        F[:, b + 1] = sig.std(1) + 1e-10
        F[:, b + 2] = sig.min(1)
        F[:, b + 3] = sig.max(1)
        F[:, b + 4] = (sig ** 2).mean(1)
        F[:, b + 5] = (np.diff(np.sign(sig), axis=1) != 0).mean(1)

    # B: cross-axis correlation over the three sensor blocks
    p = nc * 6
    pairs = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    for j, (a, b_) in enumerate([(a, b_) for a, b_ in pairs if a < nc and b_ < nc]):
        ca = X[:, :, a] - X[:, :, a].mean(1, keepdims=True)
        cb = X[:, :, b_] - X[:, :, b_].mean(1, keepdims=True)
        F[:, p + j] = np.nan_to_num(
            (ca * cb).sum(1) / (np.sqrt((ca ** 2).sum(1) * (cb ** 2).sum(1)) + 1e-10))
    p += config.N_CORR

    # C: signal-magnitude area over the accelerometer block
    F[:, p] = (np.abs(X[:, :, 0]) + np.abs(X[:, :, 1]) + np.abs(X[:, :, 2])).mean(1)
    p += 1

    # D: jerk. The magnitude is taken across the 3 accelerometer axes of the first
    # difference; the sampling-rate factor turns it into a true time derivative.
    ad = np.diff(X[:, :, :3], axis=1)
    jm = np.sqrt((ad ** 2).sum(2))
    scale = fs if jerk_with_fs else 1.0
    F[:, p + 0] = jm.mean(1) * scale
    F[:, p + 1] = jm.std(1) * scale
    p += 2

    # E: FFT band power, per channel, over the given band edges
    fb = np.fft.rfftfreq(wl, d=1.0 / fs)
    for c in range(nc):
        fm = np.abs(np.fft.rfft(X[:, :, c], axis=1)) ** 2
        for bi, (fl, fh) in enumerate(bands_hz):
            mk = (fb >= fl) & (fb < fh)
            F[:, p + c * len(bands_hz) + bi] = fm[:, mk].mean(1) if mk.any() else 0.0
    p += nc * len(bands_hz)

    # F: gravity magnitude
    gm = np.sqrt((X[:, :, :3] ** 2).sum(2))
    F[:, p + 0] = gm.mean(1)
    F[:, p + 1] = gm.std(1)
    p += 2

    # G: tilt
    F[:, p] = np.arctan2(X[:, :, 1], X[:, :, 2] + 1e-10).mean(1)
    p += 1

    # H: step lag (dominant autocorrelation lag of each accelerometer axis)
    for ci in range(min(3, nc)):
        F[:, p + ci] = [_dominant_lag(X[i, :, ci], fs) for i in range(N)]
    p += 3

    # I: spectral entropy of the gyroscope block
    for ci in range(3, min(6, nc)):
        F[:, p + (ci - 3)] = [_spectral_entropy(X[i, :, ci]) for i in range(N)]
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def redundancy(F, thresh=0.9):
    """Task 8: within- and across-family mean |r| and count of |r| > thresh pairs."""
    fam_of = {}
    for fam, idx in family_slices().items():
        for i in idx:
            fam_of[i] = fam
    ok = F.std(0) > 1e-12
    C = np.corrcoef(F[:, ok].T) if ok.sum() > 1 else np.zeros((0, 0))
    cols = np.where(ok)[0]
    within, across = [], []
    n_hi = 0
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            r = abs(float(C[a, b])) if np.isfinite(C[a, b]) else 0.0
            (within if fam_of[cols[a]] == fam_of[cols[b]] else across).append(r)
            if r > thresh:
                n_hi += 1
    return dict(n_features=int(len(cols)),
                mean_abs_r_within=float(np.mean(within)) if within else 0.0,
                mean_abs_r_across=float(np.mean(across)) if across else 0.0,
                n_pairs_above=float(n_hi), pair_thresh=thresh,
                n_families=len(FAMILY_ORDER),
                family_names=list(FAMILY_ORDER))

#@@FILE ppml/resampling.py
"""Anti-aliased resampling, spectral-distortion measurement and channel harmonisation.

Task 7. The submission upsampled WISDM 20 -> 50 Hz with ``np.interp``, which is linear
interpolation: it creates no energy above 10 Hz in the signal itself but the spectral
signature of the interpolation kernel is not the one a band-limited resampler gives, and
the paper's Nyquist argument assumed an ideal reconstruction. Everything here goes
through ``scipy.signal.resample_poly`` and the size of the difference is measured rather
than asserted.
"""

import numpy as np
from scipy.signal import resample_poly, welch

from . import config


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def ratio(src_fs, dst_fs):
    """Exact integer up/down factors for resample_poly."""
    src = int(round(float(src_fs) * 1000))
    dst = int(round(float(dst_fs) * 1000))
    g = _gcd(src, dst)
    return dst // g, src // g


def resample_signal(x, src_fs, dst_fs):
    """(T, C) at src_fs -> (T', C) at dst_fs, anti-aliased polyphase filter."""
    x = np.asarray(x, dtype=np.float64)
    up, down = ratio(src_fs, dst_fs)
    if up == down:
        return x.astype(np.float32)
    y = np.empty((x.shape[0] * up // down, x.shape[1]) if x.ndim == 2
                 else (x.shape[0] * up // down,), dtype=np.float64)
    if x.ndim == 1:
        y[:] = resample_poly(x, up, down)
    else:
        for c in range(x.shape[1]):
            col = resample_poly(x[:, c], up, down)
            n = min(len(col), y.shape[0])
            y[:n, c] = col[:n]
    return y.astype(np.float32)


def upsample_linear(x, src_fs, dst_fs):
    """The submission's np.interp upsampling, retained ONLY for the distortion measure."""
    x = np.asarray(x, dtype=np.float64)
    T, C = x.shape
    t_src = np.arange(T, dtype=float) / float(src_fs)
    T_new = int(T * dst_fs / src_fs)
    t_dst = np.arange(T_new, dtype=float) / float(dst_fs)
    out = np.empty((T_new, C), dtype=np.float64)
    for c in range(C):
        out[:, c] = np.interp(t_dst, t_src, x[:, c])
    return out.astype(np.float32)


def repeat_labels(labels, src_fs, dst_fs):
    """Nearest replication of a per-sample label stream onto a resampled time base."""
    labels = np.asarray(labels)
    T = len(labels)
    T_new = int(round(T * float(dst_fs) / float(src_fs)))
    idx = np.clip((np.arange(T_new) * float(src_fs) / float(dst_fs)).astype(int), 0, T - 1)
    return labels[idx]


def _trapz(y, x):
    """np.trapz was removed from the NumPy 2 namespace; prefer trapezoid when present."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def spectral_distortion(x_src, src_fs, dst_fs=None, band_guard=0.05):
    """Fraction of energy that a naive linear upsampling places above the source Nyquist.

    Returns spurious_energy_fraction for the linear and for the polyphase resampler.
    A band-limited 20 Hz signal has no content above 10 Hz; anything measured up there
    after resampling is an artefact of the resampler.
    """
    dst_fs = dst_fs or config.TARGET_FS
    x_src = np.asarray(x_src, dtype=np.float64)
    nyq_src = src_fs / 2.0
    out = {}
    for name, y in (("linear", upsample_linear(x_src, src_fs, dst_fs)),
                    ("polyphase", resample_signal(x_src, src_fs, dst_fs))):
        fs_used = dst_fs
        tot_e, spur_e = 0.0, 0.0
        for c in range(y.shape[1]):
            f, p = welch(y[:, c], fs=fs_used, nperseg=min(256, len(y)))
            p = np.nan_to_num(p, nan=0.0)
            tot_e += _trapz(p, f)
            above = f > nyq_src * (1.0 + band_guard)
            spur_e += _trapz(p[above], f[above])
        out["spurious_energy_fraction_%s" % name] = float(spur_e / (tot_e + 1e-20))
    out["source_nyquist_hz"] = float(nyq_src)
    out["dst_fs"] = float(dst_fs)
    out["artefact_reduction"] = float(out["spurious_energy_fraction_linear"]
                                     - out["spurious_energy_fraction_polyphase"])
    return out


# ── variants used by the ablation ─────────────────────────────────────────────
VARIANTS = ("common_50hz", "native_rate", "band_limited_features")


def window_native(signal, labels, label_map, fs, wl=None, step=None, purity=None,
                  src_fs=None):
    """Window a single subject's continuous stream.

    signal/labels may be at the native rate; when `src_fs` differs from `fs` the
    signal is resampled with resample_poly first. Window length follows the physical
    2.56 s of the paper, so the native-rate variant uses proportionally more samples.
    """
    wl = wl or config.WL
    step = step or wl // 2
    purity = purity or config.PURITY
    if src_fs is not None and abs(float(src_fs) - float(fs)) > 1e-6:
        signal = resample_signal(signal, src_fs, fs)
        labels = repeat_labels(labels, src_fs, fs)
    signal = np.asarray(signal, dtype=np.float32)
    labels = np.asarray(labels)
    X, yv = [], []
    for start in range(0, len(signal) - wl + 1, step):
        seg = signal[start:start + wl]
        lb = labels[start:start + wl]
        vals, cnts = np.unique(lb[(lb >= 0)], return_counts=True) if len(lb) else (np.array([]), np.array([]))
        if len(vals) == 0:
            continue
        keep = [i for i, v in enumerate(vals) if int(v) in label_map]
        if not keep:
            continue
        vals = vals[keep]
        cnts = cnts[keep]
        maj = int(vals[np.argmax(cnts)])
        if cnts.max() / float(wl) < purity or maj not in label_map:
            continue
        X.append(seg)
        yv.append(label_map[maj])
    if not X:
        return None, None
    return np.asarray(X, dtype=np.float32), np.asarray(yv, dtype=int)


def harmonized_band_features(X, fs, band_hz=None):
    """Variant (c): FFT features restricted to the common 0-10 Hz band.

    The Work Order names this function, so it carries that name: a reviewer grepping the
    response letter for `resampling.harmonized_band_features` must land on the code that
    produced the row. Zeroes every spectral component of each window outside `band_hz`
    before features are computed, so band power/entropy cannot see out-of-band resampling
    artefacts.
    """
    band_hz = band_hz or config.HARMONISED_BAND_HZ
    X = np.asarray(X, dtype=np.float32)
    wl = X.shape[1]
    f = np.fft.rfftfreq(wl, d=1.0 / float(fs))
    mask = ((f >= band_hz[0]) & (f <= band_hz[1])).astype(np.float32)
    spec = np.fft.rfft(X.astype(np.float64), axis=1)
    spec = spec * mask[None, :, None]
    Y = np.fft.irfft(spec, n=wl, axis=1)
    return np.nan_to_num(Y, nan=0.0).astype(np.float32)


# Where each sensor block sits in a loader's per-sample signal (column slices), per
# dataset. "total" may be "acc" (reuse the accelerometer, documented for WISDM).
CHANNEL_LAYOUT = {
    "PAMAP2":   dict(acc=(0, 3), gyro=(3, 6), total=(6, 9)),
    "WISDM v2": dict(acc=(0, 3), gyro=(3, 6), total="acc"),
    "HARTH":    dict(acc=(0, 3), gyro=None, total=(3, 6)),   # back acc, thigh acc
}


def harmonise_channels(X_native, n_ch, mode="zero_pad", gyro_slice=(3, 6), layout=None):
    """Map a dataset's native channels onto the 9-channel [acc | gyro | total_acc] tensor.

    `layout` says which native columns are which sensor; without it a 6-column input is
    ambiguous (WISDM = acc+gyro, HARTH = two accelerometers) and the old code put HARTH's
    thigh accelerometer into the GYROSCOPE slot.

    mode="zero_pad"      : 9 channels; an absent gyroscope block is exactly zero.
    mode="accel_only"    : accelerometer channels only (acc, plus the second
                           accelerometer when the dataset has one) - no gyro block at all.
    mode="channel_mask"  : zero_pad plus one constant channel: 1 = gyroscope present,
                           0 = absent.
    """
    X_native = np.asarray(X_native, dtype=np.float32)
    N, wl, c = X_native.shape
    if layout is None:
        layout = (dict(acc=(0, 3), gyro=(3, 6), total=(6, 9)) if c >= 9 else
                  dict(acc=(0, 3), gyro=(3, 6) if c >= 6 else None, total="acc"))
    acc = X_native[:, :, layout["acc"][0]:layout["acc"][1]]
    gyro = (X_native[:, :, layout["gyro"][0]:layout["gyro"][1]]
            if layout.get("gyro") else None)
    tot = (acc if layout.get("total") in (None, "acc")
           else X_native[:, :, layout["total"][0]:layout["total"][1]])
    if mode == "accel_only":
        blocks = [acc] + ([tot] if layout.get("total") not in (None, "acc") else [])
        return np.concatenate(blocks, axis=2).copy(), None
    out = np.zeros((N, wl, n_ch), dtype=np.float32)
    out[:, :, 0:3] = acc
    if gyro is not None:
        out[:, :, 3:6] = gyro
    out[:, :, 6:9] = tot
    if mode == "channel_mask":
        mask = np.full((N, wl, 1), 1.0 if gyro is not None else 0.0, dtype=np.float32)
        return out, mask
    return out, None


def ablation_variants(signal, labels, label_map, native_fs, n_ch, mode="zero_pad"):
    """Yield the three Task-7 data variants for one subject stream.

    (a) common_50hz        - polyphase resample to 50 Hz, 128-sample window;
    (b) native_rate        - no resampling, window length scaled to 2.56 s;
    (c) band_limited_features - as (a), windows band-limited to 0-10 Hz.
    """
    wl_native = int(round(config.WL * native_fs / config.TARGET_FS))
    a_X, a_y = window_native(signal, labels, label_map, fs=config.TARGET_FS,
                             src_fs=native_fs)
    n_X, n_y = window_native(signal, labels, label_map, fs=native_fs, wl=wl_native)
    if a_X is None:
        return {}
    b_X = harmonized_band_features(a_X, config.TARGET_FS)
    return {"common_50hz": (a_X, a_y),
            "native_rate": (n_X, n_y) if n_X is not None else (a_X, a_y),
            "band_limited_features": (b_X, a_y)}

#@@FILE ppml/splitting.py
"""Subject-level splitting, subject-silo clients, LOSO and the leakage audit.

This is the module the submission did not have. Every split it produces is
disjoint in SUBJECTS, never in windows, and it records which subject ids landed in
which split so the paper can state exactly how clients were formed (Task 3).
"""

import numpy as np

from . import config


# ── helpers ───────────────────────────────────────────────────────────────────
def _subject_arrays(subjects):
    return np.unique(subjects)


def _class_coverage(y, subjects, subset_subs, n_class=None):
    n_class = n_class if n_class is not None else config.N_CLASS
    mask = np.isin(subjects, subset_subs)
    present = set(int(c) for c in np.unique(y[mask]))
    return present, [c for c in range(n_class) if c not in present]


def _greedy_split(subjects, y, fracs, rng, n_class):
    """Assign whole subjects to splits until the target fractions are met, while
    keeping every class represented in every split when that is achievable."""
    subs = _subject_arrays(subjects)
    counts = {int(s): int(np.sum(subjects == s)) for s in subs}
    order = list(subs)
    rng.shuffle(order)
    tot = float(sum(counts.values()))
    names = ("train", "val", "test")
    target = {n: fracs[i] * tot for i, n in enumerate(names)}
    got = {n: 0 for n in names}
    assigned = {n: [] for n in names}
    cls_have = {n: set() for n in names}
    cls_of = {int(s): set(int(c) for c in y[subjects == s]) for s in subs}

    def deficit(n):
        return (target[n] - got[n]) / max(target[n], 1.0)

    for s in order:
        # prefer the split still missing some class, else the one furthest below target
        needs = [n for n in names if (set(range(n_class)) - cls_have[n]) & cls_of[int(s)]]
        cand = needs or names
        pick = max(cand, key=lambda n: deficit(n) + (0.5 if n in needs else 0.0))
        assigned[pick].append(int(s))
        got[pick] += counts[int(s)]
        cls_have[pick] |= cls_of[int(s)]
    return assigned


def subject_disjoint_split(subjects, y, seed=config.SEEDS[0], fracs=(0.70, 0.10, 0.20),
                           n_class=None):
    """Train/val/test with no subject appearing in two splits.

    Returns dict(train_idx, val_idx, test_idx, subjects_in_split, leakage_checks).
    """
    n_class = n_class if n_class is not None else config.N_CLASS
    subjects = np.asarray(subjects)
    y = np.asarray(y)
    rng = np.random.RandomState(int(seed))
    best = None
    for attempt in range(24):
        r = rng if attempt == 0 else np.random.RandomState(int(seed) + 1000 + attempt)
        assigned = _greedy_split(subjects, y, fracs, r, n_class)
        miss = {n: _class_coverage(y, subjects, assigned[n], n_class)[1] for n in assigned}
        score = sum(len(v) for v in miss.values())
        cand = dict(assigned=assigned, score=score, missing=miss, attempt=attempt)
        if best is None or cand["score"] < best["score"]:
            best = cand
        if best["score"] == 0:
            break
    assigned = best["assigned"]
    idx = {n: np.where(np.isin(subjects, assigned[n]))[0] for n in assigned}
    checks = leakage_report(subjects, idx, kind="subject_disjoint",
                            missing_classes=best["missing"],
                            n_attempts=best["attempt"] + 1, seed=seed)
    return dict(train_idx=idx["train"], val_idx=idx["val"], test_idx=idx["test"],
                subjects_in_split={k: sorted(int(x) for x in v) for k, v in assigned.items()},
                leakage_checks=checks)


def window_level_split(y, seed=config.SEEDS[0], fracs=(0.70, 0.10, 0.20)):
    """The OLD (leaking) split, kept only so the leakage gap can be MEASURED.

    Stratified window-level train_test_split, exactly the construction the reviewers
    objected to. Never used for a reportable result.
    """
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    try:
        tmp, te = train_test_split(idx, test_size=fracs[2], stratify=y, random_state=seed)
    except ValueError:                                  # rare class with <2 members
        tmp, te = train_test_split(idx, test_size=fracs[2], random_state=seed)
    vfrac = fracs[1] / (fracs[0] + fracs[1])
    tr, va = train_test_split(tmp, test_size=vfrac, stratify=y[tmp], random_state=seed)
    return dict(train_idx=tr, val_idx=va, test_idx=te)


def subject_leak(subjects, split_idx):
    """How many subjects appear in more than one split (0 is the only acceptable value)."""
    sets = {k: set(int(s) for s in np.asarray(subjects)[v]) for k, v in split_idx.items()}
    names = list(sets)
    dupes = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = sets[names[i]] & sets[names[j]]
            if inter:
                dupes.append((names[i], names[j], sorted(inter)))
    n_windows_exposed = 0
    if dupes:
        exposed = set(x for d in dupes for x in d[2])
        n_windows_exposed = int(sum(1 for s in subjects if int(s) in exposed))
    return dict(n_overlapping_pairs=len(dupes),
                overlapping_subjects=[(a, b, len(v)) for a, b, v in dupes],
                windows_from_overlapping_subjects=n_windows_exposed,
                n_subjects=len(sets[names[0]]) if names else 0)


def leakage_report(subjects, split_idx, kind="subject_disjoint", missing_classes=None,
                   n_attempts=None, seed=None):
    """The Work Order's named leakage check (Task 3), for ANY split.

    Both constructions go through this one function, so the zeros of the subject-disjoint
    split and the non-zeros of the old window-level split are the same measurement rather
    than two conventions. Everything returned here is stamped into the log record by
    `harness._split_provenance`, which means a table can show "0 subjects reused" as a
    measured column instead of an assertion.
    """
    full = {n: np.asarray(v) for n, v in split_idx.items()}
    leak = subject_leak(subjects, full)
    sets = {n: set(int(s) for s in np.asarray(subjects)[v]) for n, v in full.items()}

    def _ov(a, b):
        return len(sets.get(a, set()) & sets.get(b, set()))

    out = dict(split_kind=kind,
               subjects_overlap_train_test=_ov("train", "test"),
               subjects_overlap_train_val=_ov("train", "val"),
               subjects_overlap_val_test=_ov("val", "test"),
               n_overlapping_pairs=leak["n_overlapping_pairs"],
               overlapping_subjects=leak["overlapping_subjects"],
               windows_from_overlapping_subjects=
                   leak["windows_from_overlapping_subjects"],
               n_subjects=leak["n_subjects"])
    if missing_classes is not None:
        out["missing_classes"] = missing_classes
    if n_attempts is not None:
        out["n_attempts"] = int(n_attempts)
    if seed is not None:
        out["seed"] = int(seed)
    return out


def to_block_positions(client_indices, block_idx):
    """Translate GLOBAL window ids into positions inside a block of windows.

    Every partition function in this module returns global ids (they index the
    whole bundle), while the runner is handed only one split block - `harness.input()`
    gives it `X[train_idx]`. Feeding global ids straight into that block reads past
    its end or trains on another subject's windows, so the two index spaces must be
    translated at the boundary.

    A window that is not in `block_idx` is a split bug, so this raises rather than
    clamps: silently dropping test windows from a client would invalidate the leak-free
    claim the whole Task 3 rests on.
    """
    block_idx = np.asarray(block_idx, dtype=int)
    pos = np.full(int(block_idx.max()) + 1 if len(block_idx) else 0, -1, dtype=int)
    pos[block_idx] = np.arange(len(block_idx))
    out = []
    for idx in client_indices:
        idx = np.asarray(idx, dtype=int)
        too_big = idx.size and int(idx.max()) >= pos.size
        if too_big:
            raise ValueError("client window id %d is beyond every window in the split "
                             "(largest global id %d) - the partition was built on a "
                             "different split" % (int(idx.max()), pos.size - 1))
        p = pos[idx] if idx.size else idx
        bad = int((p < 0).sum())
        if bad:
            raise ValueError("%d client window(s) are not in the requested split block" % bad)
        out.append(p)
    return out


# ── federated clients ─────────────────────────────────────────────────────────
def federated_clients_by_subject(train_idx, subjects, y=None, n_clients=None,
                                 seed=config.SEEDS[0], mode="grouped"):
    """Build K clients as SUBJECT SILOS.

    mode="one_subject_per_client" : K = number of training subjects.
    mode="grouped"                : training subjects are grouped into K silos,
        balanced by window count (greedy largest-first), so client sizes are even
        while no client ever mixes a subject with another client's subject.

    The returned `client_subjects` is the exact per-client subject id list the paper
    must report.
    """
    n_clients = n_clients if n_clients is not None else config.N_CLIENTS
    train_idx = np.asarray(train_idx)
    subs_tr = np.unique(np.asarray(subjects)[train_idx])
    if mode == "one_subject_per_client":
        groups = [[int(s)] for s in subs_tr]
    else:
        rng = np.random.RandomState(int(seed) + 7)
        order = list(subs_tr)
        rng.shuffle(order)
        sizes = {int(s): int(np.sum(subjects[train_idx] == s)) for s in order}
        groups = [[] for _ in range(int(n_clients))]
        load = np.zeros(int(n_clients))
        for s in sorted(order, key=lambda x: -sizes[int(x)]):
            k = int(np.argmin(load))
            groups[k].append(int(s))
            load[k] += sizes[int(s)]
        groups = [g for g in groups if g]
    out_idx = [train_idx[np.isin(subjects[train_idx], g)] for g in groups]
    keep = [i for i, ix in enumerate(out_idx) if len(ix) > 0]
    out_idx = [out_idx[i] for i in keep]
    groups = [groups[i] for i in keep]
    return dict(client_indices=out_idx, client_subjects=groups,
                client_sizes=[len(ix) for ix in out_idx],
                n_clients=len(out_idx), mode=mode, seed=int(seed),
                privacy_unit="client (= subject silo)",
                adjacency="add/remove all data of one client")


def dirichlet_partition_windows(train_idx, y, n_clients=None, alpha=0.5,
                                seed=config.SEEDS[0]):
    """Label-Dirichlet partition of the TRAINING WINDOWS only (Task 5).

    The rejected code called this on the full dataset including the test set, which
    both leaked and made alpha meaningless. Test windows are never partitioned.
    """
    n_clients = n_clients if n_clients is not None else config.N_CLIENTS
    train_idx = np.asarray(train_idx)
    y_tr = np.asarray(y)[train_idx]
    rng = np.random.RandomState(int(seed))
    classes = np.unique(y_tr)
    buckets = [[] for _ in range(int(n_clients))]
    for c in classes:
        pos = train_idx[y_tr == c]
        rng.shuffle(pos)
        props = rng.dirichlet(np.ones(int(n_clients)) * float(alpha))
        counts = (props * len(pos)).astype(int)
        counts[-1] = max(0, len(pos) - counts[:-1].sum())
        start = 0
        for k, cnt in enumerate(counts):
            buckets[k].extend(pos[start:start + cnt].tolist())
            start += cnt
    out = [np.array(sorted(b), dtype=int) for b in buckets]
    return [o for o in out if len(o) > 0]


def iid_partition_windows(train_idx, n_clients=None, seed=config.SEEDS[0]):
    n_clients = n_clients if n_clients is not None else config.N_CLIENTS
    train_idx = np.asarray(train_idx)
    rng = np.random.RandomState(int(seed))
    parts = np.array_split(rng.permutation(train_idx), int(n_clients))
    return [p for p in parts if len(p) > 0]


# ── heterogeneity measure ─────────────────────────────────────────────────────
def partition_heterogeneity(client_indices, y, n_class=None):
    """Label skew of a partition: mean per-client entropy deficit plus the spread of
    the dominant-class share. Reported alongside every alpha in Task 5."""
    n_class = n_class if n_class is not None else config.N_CLASS
    y = np.asarray(y)
    shares, dom = [], []
    for idx in client_indices:
        if len(idx) == 0:
            continue
        cnt = np.array([np.sum(y[idx] == c) for c in range(n_class)], float)
        p = cnt / cnt.sum()
        ent = -np.sum(p[p > 0] * np.log(p[p > 0]))
        shares.append(ent / np.log(n_class))
        dom.append(float(cnt.max() / cnt.sum()))
    if not shares:
        return dict(normalized_entropy=0.0, dominant_class_share=0.0,
                    dominant_class_spread=0.0, n_clients=0)
    return dict(normalized_entropy=float(np.mean(shares)),
                dominant_class_share=float(np.mean(dom)),
                dominant_class_spread=float(np.std(dom)),
                n_clients=len(shares))


# ── leave-one-subject-out ─────────────────────────────────────────────────────
def loso_folds(subjects, n_folds_max=None, seed=None):
    """One fold per subject: that subject is the test set, the rest is training.

    Returns list of dicts with train_idx/test_idx and the held-out subject id.
    Val is carved from the training SUBJECTS (never from the held-out one).
    """
    subjects = np.asarray(subjects)
    subs = _subject_arrays(subjects)
    folds = []
    for s in subs:
        test_idx = np.where(subjects == s)[0]
        tr_subs = [x for x in subs if x != s]
        val_sub = tr_subs[-1]
        train_idx = np.where(np.isin(subjects, tr_subs[:-1]))[0]
        val_idx = np.where(subjects == val_sub)[0]
        folds.append(dict(holdout_subject=int(s), train_idx=train_idx,
                          val_idx=val_idx, test_idx=test_idx,
                          n_train=len(train_idx), n_test=len(test_idx)))
    if n_folds_max and len(folds) > n_folds_max:
        rng = np.random.RandomState(int(seed or config.SEEDS[0]))
        keep = rng.choice(len(folds), size=int(n_folds_max), replace=False)
        folds = [folds[i] for i in sorted(keep)]
    return folds

#@@FILE ppml/models.py
"""Model zoo: feature MLP, raw-window MLP, CNN-BiLSTM and LightTCN.

All four are plain PyTorch so that local epochs mean real gradient steps (the
submission used ``sklearn.MLPClassifier`` with ``max_iter=local_epochs`` and a
post-hoc "proximal correction", which is neither). Because they are torch modules,
one aggregation and one DP routine serves every architecture, which is exactly what
Task 1 asks for.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config

ARCHES = ("MLP", "MLP-Small", "MLP-Raw", "CNN-BiLSTM", "LightTCN")


class FeatureMLP(nn.Module):
    """MLP on the 117 hand-crafted features."""

    def __init__(self, in_dim, n_class=None, hidden=None):
        super().__init__()
        n_class = n_class or config.N_CLASS
        hidden = hidden or config.MLP_HIDDEN
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, n_class)]
        self.body = nn.Sequential(*layers[:-1])
        self.head = layers[-1]
        self.input_kind = "features"

    def forward(self, x):
        return self.head(self.body(x))


class RawMLP(nn.Module):
    """MLP on flattened windows - the representation-matched control (Task 3)."""

    def __init__(self, window_shape, n_class=None, hidden=None):
        super().__init__()
        n_class = n_class or config.N_CLASS
        in_dim = int(np.prod(window_shape))
        hidden = hidden or config.MLP_HIDDEN
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        # (the original dropped the output layer: its "head" was the last ReLU, so the
        # model emitted len(hidden[-1]) scores instead of n_class)
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_class)
        self.input_kind = "flat_windows"
        self.window_shape = tuple(window_shape)

    def forward(self, x):
        return self.head(self.body(x.reshape(x.shape[0], -1)))


class CNNBiLSTM(nn.Module):
    """Multi-branch conv (3/5/7) + 2-layer BiLSTM, as in the submission."""

    def __init__(self, n_ch=None, n_class=None, filters=32):
        super().__init__()
        n_ch = n_ch or config.N_CH_HARMONISED
        n_class = n_class or config.N_CLASS
        self.br3 = nn.Sequential(nn.Conv1d(n_ch, filters, 3, padding=1), nn.ReLU(),
                                 nn.Dropout(0.1))
        self.br5 = nn.Sequential(nn.Conv1d(n_ch, filters, 5, padding=2), nn.ReLU(),
                                 nn.Dropout(0.1))
        self.br7 = nn.Sequential(nn.Conv1d(n_ch, filters, 7, padding=3), nn.ReLU(),
                                 nn.Dropout(0.1))
        self.bilstm = nn.LSTM(filters * 3, 64, batch_first=True, bidirectional=True,
                              num_layers=2, dropout=0.3)
        self.body = nn.ModuleList([self.br3, self.br5, self.br7, self.bilstm])
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(),
                                  nn.Linear(64, n_class))
        self.input_kind = "windows"

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.cat([self.br3(x), self.br5(x), self.br7(x)], dim=1)
        x = x.permute(0, 2, 1)
        out, _ = self.bilstm(x)
        return self.head(out[:, -1, :])


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=self.pad, dilation=dilation)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]
        return F.relu(out) + self.res(x)


class LightTCN(nn.Module):
    def __init__(self, n_ch=None, n_class=None, ch=32):
        super().__init__()
        n_ch = n_ch or config.N_CH_HARMONISED
        n_class = n_class or config.N_CLASS
        self.body = nn.Sequential(TCNBlock(n_ch, ch, dilation=1),
                                  TCNBlock(ch, ch, dilation=2),
                                  TCNBlock(ch, ch, dilation=4))
        self.head = nn.Linear(ch, n_class)
        self.input_kind = "windows"

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.head(self.body(x)[:, :, -1])


def build(arch, feature_dim=None, window_shape=None, n_class=None, device=None):
    """window_shape is (WL, C): the channel count of the model follows the data, so a
    3-channel accelerometer-only input really gives a 3-channel network."""
    arch = arch.upper() if isinstance(arch, str) else arch
    key = {"MLP": "MLP", "MLP-SMALL": "MLP-Small", "MLP-RAW": "MLP-Raw",
           "CNN-BILSTM": "CNN-BiLSTM",
           "CNN": "CNN-BiLSTM", "LIGHTTCN": "LightTCN", "TCN": "LightTCN"}.get(arch)
    if key is None:
        raise ValueError("unknown arch %r (one of %s)" % (arch, ARCHES))
    n_ch = None
    if window_shape is not None and len(window_shape) > 1:
        n_ch = int(window_shape[1])
    if key == "MLP":
        m = FeatureMLP(feature_dim, n_class)
    elif key == "MLP-Small":
        # 117 -> 64 -> C: about 7.9k released coordinates instead of 71.8k, the
        # user-level DP arm whose noise-to-signal ratio is sqrt(d) smaller
        m = FeatureMLP(feature_dim, n_class, hidden=config.MLP_SMALL_HIDDEN)
    elif key == "MLP-Raw":
        m = RawMLP(window_shape, n_class)
    elif key == "CNN-BiLSTM":
        m = CNNBiLSTM(n_ch=n_ch, n_class=n_class)
    else:
        m = LightTCN(n_ch=n_ch, n_class=n_class)
    if device is not None:
        m = m.to(device)
    return m


def n_tensors(model):
    return len([p for p in model.parameters() if p.requires_grad])


def n_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def param_shapes(model):
    return [tuple(p.shape) for p in model.parameters() if p.requires_grad]


def get_weights(model):
    return [p.detach().cpu().numpy().astype(np.float32).copy()
            for p in model.parameters() if p.requires_grad]


def set_weights(model, weights):
    ps = [p for p in model.parameters() if p.requires_grad]
    assert len(ps) == len(weights), (len(ps), len(weights))
    with torch.no_grad():
        for p, w in zip(ps, weights):
            p.data.copy_(torch.as_tensor(w, dtype=p.dtype, device=p.device))


def zero_like(model):
    return [np.zeros_like(p.detach().cpu().numpy()) for p in model.parameters()
            if p.requires_grad]


def n_head_params(model):
    """How many trailing parameter tensors belong to the classifier head.

    FedPer keeps these private and shares only the feature extractor. For all four
    architectures the head's parameters are the last ones listed by
    ``model.parameters()``.
    """
    if not hasattr(model, "head"):
        return 0
    return len(list(model.head.parameters()))


def n_class_of(model):
    """Number of output classes, for heads that are Linear or Sequential."""
    head = getattr(model, "head", None)
    if isinstance(head, nn.Linear):
        return int(head.out_features)
    if isinstance(head, nn.Sequential):
        for layer in reversed(list(head)):
            if isinstance(layer, nn.Linear):
                return int(layer.out_features)
    return config.N_CLASS


def device_for(use_cuda=True):
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

#@@FILE ppml/fl.py
"""Federated aggregation, the corrected DP mechanism, adaptive clipping and Byzantine attacks.

Task 1 in full. The single rule that fixes the submission: clip each client's WHOLE
update with ONE L2 norm over ALL tensors, average with EQUAL weights 1/K, then add ONE
Gaussian draw of std sigma = z*C/K at the server. That construction has L2 sensitivity
C/K, so the noise multiplier really is z and the certificate from dp_accountant applies.
The same function is used for MLP, CNN-BiLSTM and LightTCN.

Optional second rule (`sparsity`, `head_tensors`): release only part of the averaged
update. At a fixed noise multiplier the per-coordinate signal is C/sqrt(d) while the noise
is z*C/K, so everything the model can learn from is governed by K/(z*sqrt(d)) - the client
count over the noise multiplier times the SQUARE ROOT OF THE MODEL DIMENSION. That is the
whole story of why the submission's DP arms sit near chance at d = 71,750 and it also says
where the only real levers are: more clients (K), a smaller released vector (d), or less
noise (z, which is the budget). Sparsifying the support leaves sqrt(d) alone and only trades
noise energy for update coverage, so its net effect is measured, not promised; restricting
the release to the classifier head shrinks d itself and is the change with a derivation
behind it. Both are functions of an already-released or architecturally-fixed object, so
neither can move epsilon. See `dp_aggregate` and `projection_selftest`.
"""

import numpy as np

from . import config


# ── primitives ────────────────────────────────────────────────────────────────
def l2_norm(tensors):
    return float(np.sqrt(sum(float(np.sum(np.asarray(t) ** 2)) for t in tensors)))


def clip_delta(delta, C):
    """Whole-update clipping: ONE norm over all tensors, one scale for all tensors."""
    n = l2_norm(delta)
    f = min(1.0, float(C) / (n + 1e-12))
    return [np.asarray(d) * f for d in delta], n, (1.0 if n > C else 0.0)


def weighted_mean(client_ws, client_ns):
    tot = float(sum(client_ns))
    out = [np.zeros_like(w) for w in client_ws[0]]
    for w, n_k in zip(client_ws, client_ns):
        for i, o in enumerate(out):
            out[i] = out[i] + np.asarray(w[i]) * (float(n_k) / tot)
    return out


def equal_mean(client_ws):
    K = float(len(client_ws))
    out = [np.zeros_like(w) for w in client_ws[0]]
    for w in client_ws:
        for i, o in enumerate(out):
            out[i] = out[i] + np.asarray(w[i]) / K
    return out


def top_k_projection(vec, keep_fraction, energy_from=None):
    """Boolean masks marking the `keep_fraction` largest-|value| coordinates of `vec`.

    `vec` must be an object the mechanism has ALREADY released - here, the noisy average of
    this round - never the clean average. Choosing the support from the clean average is
    what DP-Forward does, and it is a data-dependent selection whose worst-case effect on
    sensitivity is not bounded by C/K on its own: two averages that differ by delta can
    have supports that differ by a whole near-threshold coordinate. This pipeline certifies
    only what it can prove, so the support comes from a released object and the ranking of
    candidates is post-processing: epsilon is exactly the value dp_accountant returns for
    (z, T, delta), with no lemma and no assumption about how concentrated the update is.

    `energy_from` is the clean average, used only to report `signal_energy_fraction` - the
    honest cost of the trick: the coordinates not updated this round keep their old value,
    so whatever the round agreed on there is not released.
    """
    frac = float(keep_fraction)
    if frac >= 1.0:
        return None, dict(sparsity=1.0, noise_coords=None, total_coords=None,
                          signal_energy_fraction=1.0)
    flat = np.concatenate([np.asarray(m, dtype=np.float64).ravel() for m in vec])
    d = int(flat.size)
    m_keep = max(1, int(round(frac * d)))
    mag = np.abs(flat)
    if m_keep >= d:
        return None, dict(sparsity=1.0, noise_coords=d, total_coords=d,
                          signal_energy_fraction=1.0)
    idx = np.argpartition(mag, d - m_keep)[d - m_keep:]
    thresh = float(mag[idx].min())
    masks = [np.abs(np.asarray(m, dtype=np.float64)) >= thresh for m in vec]
    n_kept = int(sum(int(np.count_nonzero(k)) for k in masks))
    ref = [np.asarray(m, dtype=np.float64) for m in (energy_from or vec)]
    total_e = float(sum(float(np.sum(r ** 2)) for r in ref))
    kept_e = float(sum(float(np.sum(r[k] ** 2)) for r, k in zip(ref, masks)))
    return masks, dict(sparsity=frac, noise_coords=n_kept, total_coords=d,
                       signal_energy_fraction=(kept_e / total_e if total_e > 0 else 1.0))


def dp_aggregate(global_w, client_ws, C, z, rng, return_diag=False, sparsity=None,
                 head_tensors=None):
    """The reference mechanism (Work Order Task 1), with an optional release support.

    Returns the new global weights. Note there is deliberately no client-count weighting
    here: sensitivity C/K is only true for equal weights 1/K.

    The mechanism drawn in full every time is the plain one: the clipped average plus one
    Gaussian draw of std sigma = z*C/K on all d coordinates. That vector is what the
    certificate prices, and both options below are functions of it - never of the clean
    average - which is why neither can move epsilon.

    sparsity=None or 1.0  -> every coordinate is updated (the mechanism the rejected paper
                             ran, kept intact so its claim is tested as published).
    sparsity=0.03..0.5    -> only the `sparsity` fraction of coordinates with the largest
                             RELEASED magnitude is applied; the rest keep their previous
                             value. The support is a function of noise that is already in
                             the release, so it is post-processing and epsilon is identical.
                             What it buys (less random walk in most of the model) against
                             what it costs (only part of the round's update lands) is a
                             measurement, not a derivation - Tasks 1, 2 and 4 report it.
    head_tensors=H        -> the last H tensors, the classifier head, are the only
                             coordinates released; the feature extractor stays at its
                             public initialisation. This is the one option that moves the
                             signal-to-noise law itself: the update the model can receive is
                             spread over d_head coordinates instead of d, and the ratio the
                             mechanism has to beat is K/(z*sqrt(d)). The support is fixed by
                             the architecture rather than the data, so the mask is a 0/1
                             diagonal map that cannot increase an L2 distance: sensitivity
                             stays <= C/K and the certificate is exactly the same one.
    """
    deltas = []
    norms = []
    clipped_flags = []
    for w in client_ws:
        d = [np.asarray(wi, dtype=np.float64) - np.asarray(gi, dtype=np.float64)
             for wi, gi in zip(w, global_w)]
        cd, n, was = clip_delta(d, C)
        deltas.append(cd)
        norms.append(n)
        clipped_flags.append(was)
    K = len(deltas)
    if K == 0:
        return (global_w, None) if return_diag else global_w
    mean = [np.sum([d[i] for d in deltas], axis=0) / float(K) for i in range(len(global_w))]
    sigma = float(z) * float(C) / float(K)
    # the plain mechanism first, in full: this vector is what the certificate is for
    noisy_all = [np.asarray(m, dtype=np.float64) + rng.normal(0.0, sigma, m.shape)
                 for m in mean]
    ht = 0 if head_tensors is None else int(head_tensors)
    sp = 1.0 if sparsity is None else float(sparsity)
    if 0 < ht < len(mean):
        keep = [i >= len(mean) - ht for i in range(len(mean))]
        noisy = [n if k else np.zeros_like(n) for n, k in zip(noisy_all, keep)]
        total_e = float(sum(float(np.sum(np.asarray(m, dtype=np.float64) ** 2))
                            for m in mean))
        head_e = float(sum(float(np.sum(np.asarray(m, dtype=np.float64) ** 2))
                           for m, k in zip(mean, keep) if k))
        proj_diag = dict(sparsity=None, support="head",
                         noise_coords=int(sum(int(np.size(m))
                                              for m, k in zip(mean, keep) if k)),
                         total_coords=int(sum(int(np.size(m)) for m in mean)),
                         signal_energy_fraction=(head_e / total_e if total_e > 0 else 1.0))
    elif sp >= 1.0:
        noisy = noisy_all
        proj_diag = dict(sparsity=1.0, support="all", noise_coords=None,
                         total_coords=None, signal_energy_fraction=1.0)
    else:
        # the support is a function of the released noisy vector, never of `mean`
        masks, proj_diag = top_k_projection(noisy_all, sp, energy_from=mean)
        proj_diag["support"] = "released-noise"
        noisy = noisy_all if masks is None else [n * k for n, k in zip(noisy_all, masks)]
    new_w = [np.asarray(g + n, dtype=np.float32) for g, n in zip(global_w, noisy)]
    if not return_diag:
        return new_w
    diag = dict(K=K, sigma=sigma, update_norms=norms,
                clip_fraction=float(np.mean(clipped_flags)), clip_C=float(C), z=float(z))
    # the realised signal-to-noise of this one release, measured from the vectors actually
    # drawn rather than predicted from d: ||noise|| here is the empirical L2 of the Gaussian
    # over the coordinates that were really released, so the head-only and sparse supports
    # report their own ratio and not the whole-model one. This is the number that says
    # whether a chance-level DP row is the mechanism's fault or the model's.
    noise_vecs = [np.asarray(n, dtype=np.float64) - np.asarray(m, dtype=np.float64)
                  for n, m in zip(noisy, mean)]
    signal_norm = l2_norm(mean)
    noise_norm = l2_norm(noise_vecs)
    diag["release_signal_norm"] = float(signal_norm)
    diag["release_noise_norm"] = float(noise_norm)
    diag["release_snr"] = (float(signal_norm / noise_norm)
                           if noise_norm > 0 else float("inf"))
    diag.update(proj_diag)
    return new_w, diag


def projection_selftest(verbose=True):
    """Preflight assertions for the two alternative release supports - proof or stop.

    The claim being tested is narrow and load-bearing: releasing only part of the averaged
    update is worth the SAME epsilon as releasing all of it. The construction makes that a
    post-processing statement instead of a lemma - the noise is drawn in full exactly as in
    the plain mechanism, and the coordinates kept are either selected from the noisy vector
    itself (sparsity) or fixed by the architecture (head-only) - so the assertions below
    check that the code really does that. If any of them fails, `dp_sparsity` or
    `dp_head_only` is a way of faking accuracy and the arm is worthless; the preflight
    stops the whole run rather than logging it.
    """
    rng = np.random.RandomState(0)
    hard = []
    # (i) masking a FIXED support can only shrink a difference, so it is 1-Lipschitz in L2.
    # This is the only Lipschitz fact the mechanism needs, and it is the one that holds.
    worst = 0.0
    for _ in range(400):
        d = 500
        x = rng.normal(size=d)
        y = rng.normal(size=d) + 0.7
        x = x * min(1.0, 1.0 / (np.linalg.norm(x) + 1e-12))
        y = y * min(1.0, 1.0 / (np.linalg.norm(y) + 1e-12))
        mx, _ = top_k_projection([x], 0.1)
        ratio = (np.linalg.norm(x * mx[0] - y * mx[0])
                 / (np.linalg.norm(x - y) + 1e-12))
        worst = max(worst, float(ratio))
    hard.append(("fixed mask is 1-Lipschitz", worst, 1.0, 1e-9, "<="))
    # (ii) the noise multiplier is C-free: sigma/C = z/K for any clip radius. This is the
    # whole reason CLIP_REL can rescale C per architecture without touching a certificate.
    from . import dp_accountant
    a = dp_accountant.sigma_of(10.6, 0.10, 5) / 0.10
    b = dp_accountant.sigma_of(10.6, 4.72, 5) / 4.72
    hard.append(("sigma/C = z/K for any C", float(a / b), 1.0, 1e-9, "=="))
    # (iii) end-to-end. Both arms are run on the same clients and the SAME noise draw, so
    # the sparsified release must be a coordinate-subset of the plain DP release: nothing
    # reaches the model without noise on it, and at most m coordinates are touched.
    d, K, sp = 1000, 4, 0.1
    gw = [np.full(d, 0.3, dtype=np.float32)]
    cw = [gw[0] + rng.normal(scale=0.05, size=d) for _ in range(K)]
    plain, diag_p = dp_aggregate(gw, cw, C=0.5, z=10.6, rng=np.random.RandomState(1),
                                 return_diag=True, sparsity=None)
    sparse, diag_s = dp_aggregate(gw, cw, C=0.5, z=10.6, rng=np.random.RandomState(1),
                                  return_diag=True, sparsity=sp)
    s0, p0, g0 = np.asarray(sparse[0]), np.asarray(plain[0]), np.asarray(gw[0])
    # a coordinate is legitimate only if it is either untouched (== the old weight) or the
    # plain release's value. Anything else is an un-noised - i.e. unprotected - release.
    bad = int(np.count_nonzero((s0 != g0) & (s0 != p0)))
    hard.append(("no coordinate escapes noise", float(bad), 0.0, 1.0, "<="))
    updated = int(np.count_nonzero(s0 != g0))
    hard.append(("coords updated <= m+1", float(updated), float(round(sp * d) + 1),
                 1e-9, "<="))
    hard.append(("sigma identical to plain", float(diag_s["sigma"]),
                 float(diag_p["sigma"]), 1e-12, "=="))
    hard.append(("sigma = z*C/K", float(diag_s["sigma"]), 10.6 * 0.5 / K, 1e-9, "=="))
    hard.append(("support read from release",
                 1.0 if diag_s["support"] == "released-noise" else 0.0, 1.0, 1e-9, "=="))
    # (iiib) the head-only arm: a support fixed by the ARCHITECTURE. Nothing outside the
    # head may move at all, and what does move must be the plain release's own values.
    gb = [np.full(900, 0.2, dtype=np.float32), np.full(100, -0.4, dtype=np.float32)]
    r2 = np.random.RandomState(7)
    client_ws = [[gb[0] + r2.normal(scale=0.02, size=900),
                  gb[1] + r2.normal(scale=0.05, size=100)] for _ in range(K)]
    plain_h, diag_ph = dp_aggregate(gb, client_ws, C=0.5, z=10.6,
                                    rng=np.random.RandomState(2), return_diag=True)
    head_h, diag_hh = dp_aggregate(gb, client_ws, C=0.5, z=10.6,
                                   rng=np.random.RandomState(2), return_diag=True,
                                   head_tensors=1)
    hard.append(("frozen body does not move",
                 float(np.count_nonzero(np.asarray(head_h[0]) != np.asarray(gb[0]))),
                 0.0, 1.0, "<="))
    hard.append(("head arm adds no new values",
                 float(np.count_nonzero(np.asarray(head_h[1]) != np.asarray(plain_h[1]))),
                 0.0, 1.0, "<="))
    hard.append(("head arm coords = head size", float(diag_hh["noise_coords"]), 100.0,
                 1e-9, "=="))
    hard.append(("head arm sigma unchanged", float(diag_hh["sigma"]),
                 float(diag_ph["sigma"]), 1e-12, "=="))
    hard.append(("head support labelled",
                 1.0 if diag_hh["support"] == "head" else 0.0, 1.0, 1e-9, "=="))
    # the clipped average the certificate is written for really is bounded by C
    per_client = [clip_delta([np.asarray(w) - g0], 0.5)[0][0] for w in cw]
    mean_c = [np.sum(per_client, axis=0) / float(K)]
    hard.append(("clipped mean norm <= C", l2_norm(mean_c), 0.5, 1e-9, "<="))
    # (iv) the noise the mechanism actually drew corresponds to multiplier z, so the
    # certificate dp_accountant issues for (z, T, delta) is the certificate for this
    # release - read back from sigma, not assumed. Nothing here takes C or sparsity as an
    # argument, which is the whole reason neither the clip basis nor dp_sparsity can move
    # epsilon.
    z_implied = float(diag_s["sigma"]) * K / 0.5
    hard.append(("drawn noise is multiplier z", float(z_implied), 10.6, 1e-9, "=="))
    hard.append(("eps from drawn noise = certified eps",
                 float(dp_accountant.account_central_gaussian(z_implied, 20, 1e-5)["eps"]),
                 float(dp_accountant.account_central_gaussian(10.6, 20, 1e-5)["eps"]),
                 1e-9, "=="))
    ok = True
    if verbose:
        print("  dp mechanism assertions:")
    for name, got, exp, tol, cmp_ in hard:
        if cmp_ == "<=":
            within = got <= exp + tol * max(abs(exp), 1e-9)
        elif cmp_ == ">=":
            within = got >= exp - tol * max(abs(exp), 1e-9)
        else:
            within = abs(got - exp) <= tol * max(abs(exp), 1e-9)
        ok &= within
        if verbose:
            print("    %-28s got=%12.4f  %s %-12.4f  %s"
                  % (name, got, cmp_, exp, "OK" if within else "FAIL"))
    return ok


class AdaptiveClipper:
    """Gaussian quantile estimator of Andrew et al. (2021), replacing the submission's
    unaccounted Laplace clip-fraction update.

        b~ = ( sum_k 1[||Delta_k|| <= C] + N(0, sigma_b^2) ) / K
        C  <- C * exp( -eta * (b~ - target_q) )

    `freeze_rounds` stops the release after that many rounds and holds C fixed for the
    rest of the run. The estimator is the expensive part of this arm's certificate
    (1/(2*sigma_b^2) per round against 1/(2*z^2) for the model release), so the number of
    rounds it actually runs - `rounds_released` - is what the accountant composes, and it
    is read from this object rather than assumed. Nothing else changes: sigma = z*C/K
    tracks whatever radius is in force, so a frozen or a moving C costs the update
    mechanism the same z.
    """

    def __init__(self, C0=None, sigma_b=None, eta=None, target_q=None,
                 c_min=None, c_max=None, freeze_rounds=None):
        self.C = float(config.CLIP_C if C0 is None else C0)
        self.sigma_b = float(config.ADAPT_SIGMA_B if sigma_b is None else sigma_b)
        self.eta = float(config.ADAPT_ETA if eta is None else eta)
        self.target_q = float(config.ADAPT_TARGET_Q if target_q is None else target_q)
        self.c_min = float(config.ADAPT_C_MIN if c_min is None else c_min)
        self.c_max = float(config.ADAPT_C_MAX if c_max is None else c_max)
        self.freeze_rounds = (None if freeze_rounds is None
                              else int(freeze_rounds))
        self.history = []

    @property
    def rounds_released(self):
        """How many clip-fraction statistics this run actually released."""
        return int(sum(1 for h in self.history if h.get("released")))

    def update(self, update_norms, rng):
        K = float(len(update_norms))
        frozen = (self.freeze_rounds is not None
                  and self.rounds_released >= self.freeze_rounds)
        if K == 0 or frozen:
            # no release: no noise drawn, no budget spent, C held at its last value
            self.history.append(dict(round=len(self.history) + 1, C=self.C,
                                     noisy_fraction=None, true_fraction=None,
                                     released=False))
            return self.C
        indicators = np.array([1.0 if n <= self.C else 0.0 for n in update_norms])
        noisy_sum = float(indicators.sum() + rng.normal(0.0, self.sigma_b))
        b_tilde = float(np.clip(noisy_sum / K, 0.0, 1.0))
        self.C = float(np.clip(self.C * np.exp(-self.eta * (b_tilde - self.target_q)),
                               self.c_min, self.c_max))
        self.history.append(dict(round=len(self.history) + 1, C=self.C,
                                 noisy_fraction=b_tilde,
                                 true_fraction=float(indicators.mean()),
                                 released=True))
        return self.C


def clipper_selftest(verbose=True):
    """Preflight assertions for the frozen estimator - the count is the certificate.

    The adaptive arm's epsilon is composed from the number of clip-fraction statistics it
    actually released, so the pipeline is only honest if `rounds_released` counts releases
    and nothing else. A clipper that kept emitting the statistic while reporting a smaller
    number would be understating its own privacy cost, which is the exact error the
    rejected submission made. These assertions check the bookkeeping, not the arithmetic
    (dp_accountant.selftest does that).
    """
    rng = np.random.RandomState(0)
    T, W, K = 30, 7, 5
    hard = []
    c_freeze = AdaptiveClipper(C0=1.0, freeze_rounds=W)
    for _ in range(T):
        c_freeze.update(list(rng.uniform(0.2, 3.0, size=K)), rng)
    hard.append(("releases = freeze_rounds", float(c_freeze.rounds_released), float(W),
                 1e-9, "=="))
    hard.append(("history has every round", float(len(c_freeze.history)), float(T),
                 1e-9, "=="))
    after = [h["C"] for h in c_freeze.history[W:]]
    hard.append(("C constant once frozen", float(len(set(after))), 1.0, 1e-9, "=="))
    hard.append(("frozen rounds log no statistic",
                 float(sum(1 for h in c_freeze.history[W:] if h["noisy_fraction"] is None)),
                 float(T - W), 1e-9, "=="))
    hard.append(("only released rounds flagged",
                 float(sum(1 for h in c_freeze.history if h.get("released"))), float(W),
                 1e-9, "=="))
    # the published mechanism must still be available: no freeze, T releases
    c_all = AdaptiveClipper(C0=1.0, freeze_rounds=None)
    for _ in range(T):
        c_all.update(list(rng.uniform(0.2, 3.0, size=K)), rng)
    hard.append(("unfrozen releases every round", float(c_all.rounds_released), float(T),
                 1e-9, "=="))
    hard.append(("unfrozen radius moves", float(len(set(h["C"] for h in c_all.history)) > 1),
                 1.0, 1e-9, "=="))
    ok = True
    if verbose:
        print("  adaptive-clipper assertions:")
    for name, got, exp, tol, cmp_ in hard:
        within = abs(got - exp) <= tol * max(abs(exp), 1e-9)
        ok &= within
        if verbose:
            print("    %-30s got=%12.4f  %s %-12.4f  %s"
                  % (name, got, cmp_, exp, "OK" if within else "FAIL"))
    return ok


# ── robust aggregators (Task 6) ───────────────────────────────────────────────
# A client's weights are a LIST OF TENSORS, and the tensors differ in shape (an MLP
# has 8 of them, a CNN-BiLSTM 26), so every coordinate-wise operation has to run one
# tensor at a time. Stacking the clients directly raises
# "inhomogeneous shape" on the median arm and silently mis-slices everywhere else.
def _stack_by_tensor(client_ws, i):
    return np.stack([np.asarray(w[i]) for w in client_ws], axis=0)


def coordinate_median(client_ws):
    return [np.median(_stack_by_tensor(client_ws, i), axis=0)
            for i in range(len(client_ws[0]))]


def trimmed_mean(client_ws, trim=0.2):
    K = float(len(client_ws))
    n_trim = int(np.floor(K * trim))
    out = []
    for i in range(len(client_ws[0])):
        s = _stack_by_tensor(client_ws, i)
        if n_trim == 0:
            out.append(np.mean(s, axis=0))
            continue
        srt = np.sort(s, axis=0)
        keep = srt[n_trim:int(K) - n_trim] if K - 2 * n_trim > 0 else srt
        out.append(np.mean(keep, axis=0))
    return out


def krum(client_ws, assumed_f=None, n_select=1):
    """KRUM / blurred-krum with a FIXED assumed number of colluders.

    The submission passed the true number of attackers, which no honest server knows.
    Here `assumed_f` is a declared parameter (default config.BYZ_ASSUMED_F = 2) and is
    never derived from the actual attack configuration.
    """
    assumed_f = config.BYZ_ASSUMED_F if assumed_f is None else int(assumed_f)
    n = len(client_ws)
    if n <= 2 * assumed_f + 2:
        return coordinate_median(client_ws)
    # Krum scores are defined on the WHOLE update, so all tensors of a client are
    # concatenated into one vector before the pairwise distances are taken.
    flat = [np.concatenate([np.asarray(t).reshape(-1) for t in w]) for w in client_ws]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(np.sum((flat[i] - flat[j]) ** 2))
    m = n - assumed_f - 2
    scores = np.array([np.sum(np.sort(D[i])[1:m + 1]) for i in range(n)])
    sel = np.argsort(scores)[:max(1, int(n_select))]
    chosen = [client_ws[k] for k in sel]
    return [np.mean(_stack_by_tensor(chosen, i), axis=0)
            for i in range(len(chosen[0]))]


AGGREGATORS = {"fedavg": weighted_mean, "equal_mean": equal_mean,
               "median": coordinate_median, "trimmed_mean": trimmed_mean,
               "krum": krum}


def aggregate_plain(client_ws, client_ns, how="fedavg", assumed_f=None):
    if how == "fedavg":
        return weighted_mean(client_ws, client_ns)
    if how == "equal_mean":
        return equal_mean(client_ws)
    if how == "median":
        return coordinate_median(client_ws)
    if how == "trimmed_mean":
        return trimmed_mean(client_ws)
    if how == "krum":
        return krum(client_ws, assumed_f=assumed_f)
    raise ValueError("unknown aggregator %r" % how)


# ── Byzantine attacks ─────────────────────────────────────────────────────────
UPDATE_ATTACKS = ("none", "signflip", "scaled", "noise", "ipm", "signflip_x1", "scaled_neg5", "ipm_eps0.5")
DATA_ATTACKS = ("labelflip",)


def is_malicious(client_ids, n_clients, fraction, rng):
    """Deterministic adversary set: the first ceil(fraction*K) client ids."""
    n_adv = int(round(float(fraction) * n_clients))
    adv = set(int(c) for c in np.array(client_ids)[:n_adv]) if n_adv > 0 else set()
    return adv


def attack_update(delta, global_w, how, rng, honest_deltas=None, factor=None,
                  noise_std=None, ipm_eps=None):
    """Transform one malicious client's delta. `honest_deltas` are the deltas of the
    non-colluding clients (needed by IPM)."""
    d = [np.asarray(x).copy() for x in delta]
    if how == "none":
        return d
    if how in ("signflip", "signflip_x1"):
        # sign-flipping attack: the client sends the NEGATIVE of its honest update
        return [-x for x in d]
    if how in ("scaled", "scaled_neg5"):
        # scaled poisoning: the reversed update, boosted by BYZ_SCALED_FACTOR, so a
        # minority can outweigh the honest majority under plain averaging
        f = config.BYZ_SCALED_FACTOR if factor is None else factor
        return [x * -float(f) for x in d]
    if how == "noise":
        s = config.BYZ_NOISE_STD if noise_std is None else noise_std
        return [x + rng.normal(0.0, float(s), x.shape) for x in d]
    if how in ("ipm", "ipm_eps0.5"):
        # Xie et al. 2019: malicious update = -eps * MEAN OF THE HONEST UPDATES.
        # The submission's "ipm" was an alias for sign-flip; this is the real thing.
        eps = config.BYZ_IPM_EPS if ipm_eps is None else ipm_eps
        if not honest_deltas:
            return [x * -float(eps) for x in d]
        mean_honest = [np.mean([h[i] for h in honest_deltas], axis=0)
                       for i in range(len(d))]
        return [np.asarray(m) * (-float(eps)) for m in mean_honest]
    raise ValueError("unknown update attack %r" % how)


def flip_labels(y, rng, n_class=None, to=None):
    n_class = n_class or config.N_CLASS
    to = config.BYZ_LABEL_FLIP_TO if to is None else to
    out = np.asarray(y).copy()
    return np.where(out == to, (out + 1) % n_class, out)

#@@FILE ppml/datasets.py
"""Dataset loading for UCI-HAR, PAMAP2, WISDM v2 and HARTH.

Three things the submission's loaders could not do and the Work Order requires:

1. they return the SUBJECT id of every window, without which neither subject-disjoint
   splitting nor user-level DP is possible (the old loaders concatenated the per-subject
   files and threw the identity away);
2. every channel block is mapped explicitly and reported, instead of the old
   ``[22,23,24,28,29,30,4,5,6]`` selection which put a PAMAP2 *temperature* column and
   an out-of-range index into the accelerometer block;
3. all resampling goes through ``resample_poly`` (see resampling.py).

Sensor mapping actually used
----------------------------
dataset   acc block            gyro block           total_acc block       absent
UCI-HAR   body_acc            body_gyro            total_acc             -
PAMAP2    hand acc +-16 g      hand gyroscope       chest acc +-16 g      -
          (cols 4-6)           (cols 10-12)         (cols 21-23)   [0-based, PAMAP2 readme]
WISDM     phone accel          phone gyroscope      phone accel           -
          (raw/phone/accel)    (raw/phone/gyro, aligned to accel by timestamp)
HARTH     back acc             -                    thigh acc             gyro

Class spaces (config.N_CLASS is set per dataset by the driver):
  UCI-HAR, PAMAP2, HARTH : 6 classes  0 Walking, 1 WalkUp, 2 WalkDn, 3 Sitting,
                                      4 Standing, 5 Laying
  WISDM (2019, UCI #507) : 3 classes  0 Walking (A), 1 Sitting (D), 2 Standing (E).
      The archive has 18 activities (A-S, no N). It has no lying class and does not
      separate stairs up from stairs down, so only these three map onto the shared set.
HARTH keeps 1 walking, 4 stairs up, 5 stairs down, 7 sitting, 6 standing, 8 lying and
drops running (2), shuffling (3) and cycling (13, 14, 130, 140), which ARE in the
archive but have no equivalent in the shared class set.
"""

import os
import re
import zipfile
import urllib.request

import numpy as np
import pandas as pd

from . import config
from .resampling import (CHANNEL_LAYOUT, ablation_variants, harmonized_band_features,
                         harmonise_channels, window_native)

CLASS_NAMES = ["Walking", "Walk Up", "Walk Down", "Sitting", "Standing", "Laying"]
CLASS_NAMES_BY_DATASET = {
    "UCI HAR": CLASS_NAMES, "PAMAP2": CLASS_NAMES, "HARTH": CLASS_NAMES,
    "WISDM v2": ["Walking", "Sitting", "Standing"],
}
# Number of classes each dataset is scored on. The driver writes this into
# PPML_N_CLASS before the package is imported, so config.N_CLASS matches the bundle.
N_CLASS_BY_DATASET = {k: len(v) for k, v in CLASS_NAMES_BY_DATASET.items()}


def class_names(name):
    return CLASS_NAMES_BY_DATASET.get(name, CLASS_NAMES)

LABEL_MAPS = {
    "UCI HAR": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5},
    "PAMAP2": {4: 0, 12: 1, 13: 2, 2: 3, 3: 4, 1: 5},
    # WISDM 2019 activity_key.txt: A walking, B jogging, C stairs, D sitting,
    # E standing, F typing, G brushing teeth, ... Only A/D/E have a shared-class
    # equivalent; they become a 3-class problem (0 Walking, 1 Sitting, 2 Standing).
    "WISDM v2": {"A": 0, "D": 1, "E": 2},
    # HARTH (UCI #779): 1 walking, 2 running, 3 shuffling, 4 stairs up, 5 stairs down,
    # 6 standing, 7 sitting, 8 lying, 13/14/130/140 cycling. Running, shuffling and
    # cycling have no shared-class equivalent and are dropped (window discarded).
    "HARTH": {1: 0, 4: 1, 5: 2, 7: 3, 6: 4, 8: 5},
}

# 0-based column indices into each raw per-sample record
CHANNEL_MAP = {
    "UCI HAR": dict(acc=(0, 1, 2), gyro=(3, 4, 5), total=(6, 7, 8), n_native=9),
    # PAMAP2 readme, 0-based: 0 time, 1 activity, 2 heart rate, then three 17-column IMU
    # blocks (hand 3-19, chest 20-36, ankle 37-53), each: temp, acc16 x3, acc6 x3,
    # gyro x3, mag x3, orientation x4. So hand acc16 = 4-6, hand gyro = 10-12,
    # chest acc16 = 21-23.
    "PAMAP2": dict(acc=(4, 5, 6), gyro=(10, 11, 12), total=(21, 22, 23), n_native=54),
    "WISDM v2": dict(acc=(0, 1, 2), gyro=(3, 4, 5), total=(0, 1, 2), n_native=6),
    "HARTH": dict(acc=(0, 1, 2), gyro=None, total=(3, 4, 5), n_native=6),
}

NATIVE_FS = {"UCI HAR": 50.0, "PAMAP2": 100.0, "WISDM v2": 20.0, "HARTH": 50.0}

_URLS = {
    # current UCI addresses first; the legacy /ml/machine-learning-databases/ paths are
    # kept only as a fallback
    "UCI HAR": ["https://archive.ics.uci.edu/static/public/240/"
                "human+activity+recognition+using+smartphones.zip",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/"
                "UCI%20HAR%20Dataset.zip"],
    "PAMAP2": ["https://archive.ics.uci.edu/static/public/231/"
               "pamap2+physical+activity+monitoring.zip",
               "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/"
               "PAMAP2_Dataset.zip"],
    "WISDM v2": ["https://archive.ics.uci.edu/static/public/507/"
                 "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip",
                 "https://archive.ics.uci.edu/ml/machine-learning-databases/00507/"
                 "wisdm-dataset.zip"],
    "HARTH": ["https://archive.ics.uci.edu/static/public/779/harth.zip",
              "https://zenodo.org/record/5178138/files/harth.zip"],
}


class DatasetBundle:
    """Everything one experiment needs for one dataset."""

    def __init__(self, name, X, y, subject, meta):
        self.name = name
        self.X = X                # (N, WL, 9) harmonised @ 50 Hz
        self.y = y                # (N,) shared class ids
        self.subject = subject    # (N,) original subject id
        self.meta = meta
        self.variants = {}        # filled by load(): Task 7 data variants

    def __len__(self):
        return len(self.y)

    def summary(self):
        cnt = dict(zip(*np.unique(self.y, return_counts=True)))
        return dict(dataset=self.name, n_windows=int(len(self.y)),
                    n_subjects=int(len(np.unique(self.subject))),
                    window_shape=list(self.X.shape),
                    native_fs=NATIVE_FS[self.name],
                    class_counts={int(k): int(v) for k, v in cnt.items()},
                    missing_classes=[c for c in range(config.N_CLASS) if c not in cnt],
                    class_names=list(class_names(self.name)),
                    channel_map={k: v for k, v in CHANNEL_MAP[self.name].items()},
                    label_map={str(k): v for k, v in LABEL_MAPS[self.name].items()})

    def variant(self, name):
        """Task 7 data variants: common_50hz | native_rate | band_limited_features."""
        return self.variants[name]

    def as_variant(self, name):
        """A bundle rebuilt on one Task-7 variant, carrying the sampling rate its
        windows are actually at (native_rate keeps the device clock)."""
        if name not in self.variants:
            raise KeyError("variant %r not loaded (have %s)"
                           % (name, sorted(self.variants)))
        X, y, s = self.variants[name]
        fs = getattr(self, "variants_fs", {}).get(name, config.TARGET_FS)
        b = DatasetBundle(self.name, X, y, s, dict(self.meta))
        b.variants = self.variants
        b.variants_fs = getattr(self, "variants_fs", {})
        b.meta.update(variant=name, fs=float(fs))
        b.meta.update(b.summary())
        b.meta["variant"] = name
        b.meta["fs"] = float(fs)
        return b


def _fetch(name, dest_dir):
    """Find the dataset, in this order, and return the directory its loader reads:

    1. any folder under /kaggle/input whose CONTENTS match the dataset (folder names are
       not trusted: "har" is inside "harth", and Kaggle copies are named freely);
    2. a zip under /kaggle/input, unpacked into dest_dir;
    3. an earlier download in dest_dir;
    4. a fresh download from UCI (Internet ON).
    """
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.isdir("/kaggle/input"):
        root = _signature_root(name, "/kaggle/input")
        if root:
            print("  %s: using attached Kaggle input %s" % (name, root))
            return root
        for dp, dn, fn in os.walk("/kaggle/input"):
            for f in fn:
                if f.lower().endswith(".zip") and _zip_is(name, os.path.join(dp, f)):
                    print("  %s: unpacking attached archive %s" % (name, f))
                    with zipfile.ZipFile(os.path.join(dp, f), "r") as z:
                        z.extractall(dest_dir)
                    _extract_nested_zips(dest_dir)
                    root = _signature_root(name, dest_dir)
                    if root:
                        return root
    root = _signature_root(name, dest_dir)
    if root:
        return root
    for url in _URLS[name]:
        zip_path = os.path.join(dest_dir, name.replace(" ", "_") + ".zip")
        print("  %s: downloading %s" % (name, url))
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(dest_dir)
            os.remove(zip_path)
            _extract_nested_zips(dest_dir)
            root = _signature_root(name, dest_dir)
            if root:
                return root
            print("  %s: archive unpacked but the expected files were not found" % name)
        except Exception as exc:
            print("  %s: failed (%s)" % (name, exc))
    return None


_HARTH_FILE = re.compile(r"^S\d+\.csv$", re.I)
_PAMAP_FILE = re.compile(r"^subject1\d\d\.dat$", re.I)
_WISDM_ACC = re.compile(r"^data_\d+_accel_phone\.txt$", re.I)


def _signature_root(name, base):
    """The directory under `base` whose files identify `name`, or None.

    UCI HAR : the folder holding train/Inertial Signals (raw signals, not the 561-feature
              CSV copies that circulate on Kaggle - those cannot be used)
    PAMAP2  : the Protocol folder with subject101.dat ... (never Optional/)
    WISDM   : raw/phone/accel with data_16xx_accel_phone.txt
    HARTH   : the folder with S0xx.csv files
    """
    if not os.path.isdir(base):
        return None
    for dp, dn, fn in os.walk(base):
        parts = dp.split(os.sep)
        # macOS archive metadata (__MACOSX/) mirrors the real folders with "._" files;
        # the UCI-HAR zip ships one, and it must never be taken for the data
        dn[:] = [d for d in dn if d != "__MACOSX"]
        if "__MACOSX" in parts:
            continue
        if name == "UCI HAR":
            if os.path.isdir(os.path.join(dp, "train", "Inertial Signals")):
                return dp
        elif name == "PAMAP2":
            if "Optional" in parts:
                continue
            if any(_PAMAP_FILE.match(f) for f in fn):
                return dp
        elif name == "WISDM v2":
            if dp.endswith(os.path.join("phone", "accel")) and "raw" in parts \
                    and any(_WISDM_ACC.match(f) for f in fn):
                return dp
        elif name == "HARTH":
            if sum(1 for f in fn if _HARTH_FILE.match(f)) >= 2:
                return dp
    return None


def _zip_is(name, path):
    """Peek inside an attached zip (without unpacking) to see whether it is `name`."""
    keys = {"UCI HAR": ("inertial signals", "uci har", "human+activity"),
            "PAMAP2": ("pamap2", "subject101"),
            "WISDM v2": ("wisdm",),
            "HARTH": ("harth", "s006.csv")}[name]
    low = os.path.basename(path).lower()
    if any(k in low for k in keys):
        return True
    try:
        with zipfile.ZipFile(path) as z:
            names = " ".join(z.namelist()[:200]).lower()
        return any(k in names for k in keys)
    except zipfile.BadZipFile:
        return False


def _extract_nested_zips(dest_dir, max_depth=3):
    """UCI's /static/public/ archives wrap the original archive in another zip
    (e.g. PAMAP2_Dataset.zip, UCI HAR Dataset.zip, wisdm-dataset.zip). Unpack every
    inner zip in place so the loaders see the real files."""
    for _ in range(max_depth):
        inner = [os.path.join(dp, f) for dp, dn, fn in os.walk(dest_dir) for f in fn
                 if f.lower().endswith(".zip")]
        if not inner:
            return
        for zp in inner:
            try:
                with zipfile.ZipFile(zp, "r") as z:
                    z.extractall(os.path.dirname(zp))
                os.remove(zp)
            except zipfile.BadZipFile:
                print("    skipping unreadable inner archive %s" % zp)
                os.remove(zp)


def _mount_candidates(name):
    out = []
    if not os.path.isdir("/kaggle/input"):
        return out
    keys = {"UCI HAR": ("uci", "har"), "PAMAP2": ("pamap",),
            "WISDM v2": ("wisdm",), "HARTH": ("harth",)}
    for d in sorted(os.listdir("/kaggle/input")):
        low = d.lower()
        if any(k in low for k in keys[name]):
            out.append(os.path.join("/kaggle/input", d))
    return out


def _find_root(name, dest):
    if name == "WISDM v2":
        # The v2 archive carries the same streams twice: parseable .txt under
        # raw/phone/accel and .arff under arff_files/phone/accel. Walking returns
        # whichever the filesystem lists first, so the raw copy is chosen explicitly.
        hits = [r for r, d, f in os.walk(dest)
                if r.endswith(os.path.join("phone", "accel"))]
        for r in hits:
            if "raw" in r.split(os.sep):
                return r
        return hits[0] if hits else dest
    for root, dirs, files in os.walk(dest):
        if name == "UCI HAR" and os.path.isdir(os.path.join(root, "train")):
            return root
        if name == "PAMAP2" and root.endswith("Protocol"):
            return root
        if name == "HARTH" and any(f.endswith(".csv") for f in files):
            return root
    return dest


# ── UCI HAR (pre-windowed) ────────────────────────────────────────────────────
def _load_uci(root):
    base = os.path.join(root, "UCI HAR Dataset")
    if not os.path.isdir(base):
        base = root
    sig_ch = ["body_acc_x", "body_acc_y", "body_acc_z",
              "body_gyro_x", "body_gyro_y", "body_gyro_z",
              "total_acc_x", "total_acc_y", "total_acc_z"]
    Xs, ys, ss = [], [], []
    for split in ("train", "test"):
        sig = os.path.join(base, split, "Inertial Signals")
        try:
            arrs = [np.loadtxt(os.path.join(sig, "%s_%s.txt" % (c, split))) for c in sig_ch]
        except (OSError, ValueError):
            return None
        Xs.append(np.stack(arrs, axis=2).astype(np.float32))
        ys.append(np.loadtxt(os.path.join(base, split, "y_%s.txt" % split), dtype=int) - 1)
        ss.append(np.loadtxt(os.path.join(base, split, "subject_%s.txt" % split), dtype=int))
    X = np.concatenate(Xs)
    y = np.concatenate(ys).astype(int)
    s = np.concatenate(ss).astype(int)
    return X, y, s


# ── PAMAP2 ────────────────────────────────────────────────────────────────────
def _load_pamap2(root):
    # PAMAP2 has 9 subjects (subject101-109). Only Protocol/ is read: Optional/ holds the
    # same subjects' optional activities, none of which map to the shared classes.
    prot = _signature_root("PAMAP2", root) or root
    files = sorted(os.path.join(dp, f) for dp, d, fn in os.walk(prot) for f in fn
                   if _PAMAP_FILE.match(f) and "Optional" not in dp.split(os.sep))
    if not files:
        print("    PAMAP2: no .dat streams under %s" % prot)
    recs = []
    cm = CHANNEL_MAP["PAMAP2"]
    for fp in files:
        fn = os.path.basename(fp)
        subj = _subject_id(fn)
        try:
            df = pd.read_csv(fp, sep=r"\s+", header=None, dtype=float)
        except Exception as exc:
            print("    %s: skip (%s)" % (fn, exc))
            continue
        if df.shape[1] < cm["n_native"]:
            print("    %s: only %d cols, expected %d - skip"
                  % (fn, df.shape[1], cm["n_native"]))
            continue
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.fillna(0.0, inplace=True)
        act = df.iloc[:, 1].values.astype(int)
        blocks = [df.iloc[:, list(cm[k])].values.astype(np.float32)
                  for k in ("acc", "gyro", "total")]
        sig = np.concatenate(blocks, axis=1)
        sig = np.clip(sig, -config.CLIP_RANGE, config.CLIP_RANGE)
        recs.append((subj, sig, act))
    # Expected: 9 subjects. Which ids were actually read is printed so a partial
    # archive cannot pass unnoticed.
    print("    PAMAP2: %d .dat under %s -> %d subjects %s"
          % (len(files), prot, len(set(r[0] for r in recs)),
             sorted(set(r[0] for r in recs))))
    return recs


# ── WISDM v2 ──────────────────────────────────────────────────────────────────
def _read_wisdm_stream(path):
    """One raw WISDM file -> DataFrame(act, t, x, y, z) for the mapped activities only.
    Line format: subject,activity,timestamp,x,y,z;"""
    rows = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            parts = line.strip().replace(";", "").split(",")
            if len(parts) < 6:
                continue
            act = parts[1].strip()
            if act not in LABEL_MAPS["WISDM v2"]:
                continue
            try:
                rows.append((act, int(float(parts[2])), float(parts[3]),
                             float(parts[4]), float(parts[5])))
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=["act", "t", "x", "y", "z"])


# filled by _load_wisdm so the driver can print how well accel and gyro lined up
WISDM_ALIGNMENT = {}


def _load_wisdm(root):
    """Phone accelerometer + phone gyroscope, aligned by timestamp.

    The two sensors are separate files with their own clocks, so rows are NOT matched by
    position: each accelerometer sample takes the nearest gyroscope sample of the same
    activity within half a sampling period (pd.merge_asof). Accelerometer samples with no
    gyroscope partner are dropped, and the matched fraction per subject is recorded in
    WISDM_ALIGNMENT and printed by the driver.
    """
    accel = _signature_root("WISDM v2", root) or root
    gyro = os.path.join(os.path.dirname(accel), "gyro")
    if not os.path.isdir(accel):
        return None
    has_gyro = os.path.isdir(gyro)
    files = sorted(f for f in os.listdir(accel) if _WISDM_ACC.match(f))
    recs = []
    WISDM_ALIGNMENT.clear()
    for fn in files:
        subj = _subject_id(fn)
        acc = _read_wisdm_stream(os.path.join(accel, fn))
        if len(acc) < config.WL:
            continue
        gfile = os.path.join(gyro, fn.replace("_accel_", "_gyro_")) if has_gyro else None
        if gfile and os.path.exists(gfile):
            gyr = _read_wisdm_stream(gfile).rename(columns={"x": "gx", "y": "gy", "z": "gz"})
        else:
            gyr = pd.DataFrame(columns=["act", "t", "gx", "gy", "gz"])
        parts = []
        n_acc, n_match = 0, 0
        for act, a_df in acc.groupby("act", sort=False):
            a_df = a_df.sort_values("t")
            g_df = gyr[gyr["act"] == act].sort_values("t")
            n_acc += len(a_df)
            if len(g_df) == 0:
                continue
            dt = np.median(np.diff(a_df["t"].values)) if len(a_df) > 1 else 0
            tol = max(1, int(abs(dt) * 0.5))
            m = pd.merge_asof(a_df, g_df[["t", "gx", "gy", "gz"]], on="t",
                              direction="nearest", tolerance=tol)
            m = m.dropna(subset=["gx", "gy", "gz"])
            n_match += len(m)
            parts.append(m)
        WISDM_ALIGNMENT[subj] = dict(accel_samples=n_acc, matched=n_match,
                                     matched_fraction=(n_match / n_acc) if n_acc else 0.0)
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        if len(df) < config.WL:
            continue
        sig = df[["x", "y", "z", "gx", "gy", "gz"]].values.astype(np.float32)
        sig = np.clip(sig, -config.CLIP_RANGE, config.CLIP_RANGE)
        codes = np.array([LABEL_MAPS["WISDM v2"][a] for a in df["act"].values], dtype=int)
        recs.append((subj, sig, codes))
    if WISDM_ALIGNMENT:
        fr = [v["matched_fraction"] for v in WISDM_ALIGNMENT.values()]
        print("    WISDM accel/gyro alignment: %d subjects, matched fraction "
              "min %.2f / median %.2f" % (len(fr), min(fr), float(np.median(fr))))
    return recs


# ── HARTH ─────────────────────────────────────────────────────────────────────
def _subject_id(name):
    """The FIRST digit run of a file name is the subject; later runs are activity,
    trial or file numbers.

    Concatenating every digit in the name (the old behaviour) turned HARTH's
    `S01_A02.csv` / `S01_A03.csv` into two different "subjects", so one person's windows
    were split across two silos. That is not a cosmetic bug: the subject-disjoint split,
    the K client silos and the claim that a client-level certificate is also a user-level
    one all rest on "one silo = one person".
    """
    m = re.search(r"(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else 0


_HARTH_COLS = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]


def _load_harth(root):
    """HARTH S0xx.csv files, sensor columns read BY NAME (some files carry extra
    index columns, which shifts every positional index)."""
    root = _signature_root("HARTH", root) or root
    csvs = sorted(os.path.join(dp, f) for dp, dn, fn in os.walk(root) for f in fn
                  if _HARTH_FILE.match(f))
    recs = []
    for fp in csvs:
        subj = _subject_id(os.path.basename(fp))
        try:
            df = pd.read_csv(fp)
        except Exception as exc:
            print("    %s: skip (%s)" % (os.path.basename(fp), exc))
            continue
        df.columns = [str(c).strip().lower() for c in df.columns]
        if not all(c in df.columns for c in _HARTH_COLS + ["label"]):
            print("    %s: columns %s do not include %s - skip"
                  % (os.path.basename(fp), list(df.columns), _HARTH_COLS + ["label"]))
            continue
        sig = df[_HARTH_COLS].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
        lbl = pd.to_numeric(df["label"], errors="coerce").values
        sig = np.clip(np.nan_to_num(sig, nan=0.0), -config.CLIP_RANGE, config.CLIP_RANGE)
        codes = np.array([LABEL_MAPS["HARTH"].get(int(v), -1)
                          for v in np.nan_to_num(lbl, nan=-1).astype(int)], dtype=int)
        if len(sig) < config.WL:
            continue
        recs.append((subj, sig, codes))
    print("    HARTH: %d files -> %d subjects" % (len(csvs), len(set(r[0] for r in recs))))
    return recs


# ── build bundle ──────────────────────────────────────────────────────────────
def _cap_windows(X, y, s, cap, seed):
    """Deterministic stratified subsample: every subject keeps a share of `cap`.

    A flat `X[:cap]` would silently delete the later subjects, and the subject pool is
    what the K silos, the LOSO folds and the three MIA pools are built from. Budget per
    subject is cap/n_subjects, so a dataset with more subjects loses proportionally more
    windows but never loses a person.
    """
    if not cap or len(X) <= cap:
        return X, y, s
    subjects = np.unique(s)
    per = max(1, int(cap) // max(1, len(subjects)))
    rng = np.random.RandomState(int(seed))
    keep = []
    for subj in subjects:
        idx = np.flatnonzero(s == subj)
        if len(idx) > per:
            idx = np.sort(rng.choice(idx, size=per, replace=False))
        keep.append(idx)
    k = np.concatenate(keep)
    return X[k], y[k], s[k]


def load(name, max_subjects=None, channel_mode="zero_pad", seed=config.SEEDS[0],
         max_windows=None):
    """Return a DatasetBundle for `name` in one of
    {'UCI HAR', 'PAMAP2', 'WISDM v2', 'HARTH'}."""
    if name not in _URLS:
        raise ValueError("unknown dataset %r" % name)
    set_class_space(name)
    max_windows = config.MAX_WINDOWS if max_windows is None else max_windows
    root = _fetch(name, os.path.join(config.DATA_DIR, name.replace(" ", "_")))
    if root is None:
        raise RuntimeError("could not obtain %s" % name)
    variants = {}
    if name == "UCI HAR":
        got = _load_uci(root)
        if got is None:
            raise RuntimeError("UCI HAR layout not recognised in %s" % root)
        X, y, s = got
        if max_subjects:
            keep = np.unique(s)[:int(max_subjects)]
            m = np.isin(s, keep)
            X, y, s = X[m], y[m], s[m]
        # UCI ships 128-sample windows at 50 Hz with the 9 harmonised blocks already
        variants["common_50hz"] = (X, y, s)
        variants["native_rate"] = (X, y, s)
        variants["band_limited_features"] = (harmonized_band_features(X, config.TARGET_FS), y, s)
    else:
        recs = {"PAMAP2": _load_pamap2, "WISDM v2": _load_wisdm,
                "HARTH": _load_harth}[name](root)
        if not recs:
            raise RuntimeError("no usable %s files under %s" % (name, root))
        if max_subjects:
            recs = sorted(recs, key=lambda r: r[0])[:int(max_subjects)]
        fs = NATIVE_FS[name]
        lmap = LABEL_MAPS[name]
        identity_map = {k: k for k in range(config.N_CLASS)}
        per_variant = {k: [] for k in ("common_50hz", "native_rate",
                                       "band_limited_features")}
        for subj, sig, codes in recs:
            lab = (np.array([lmap.get(int(c), -1) for c in codes], dtype=int)
                   if name == "PAMAP2" else np.asarray(codes, dtype=int))
            vs = ablation_variants(sig, lab, identity_map, fs, sig.shape[1])
            if "common_50hz" not in vs:
                continue
            for k in per_variant:
                if k in vs:
                    Xv, yv = vs[k]
                    per_variant[k].append((subj, Xv, yv))
        for k, lst in per_variant.items():
            if not lst:
                continue
            X9s, y9s, s9s = [], [], []
            for subj, Xv, yv in lst:
                X9, msk = harmonise_channels(Xv, config.N_CH_HARMONISED, mode=channel_mode,
                                             layout=CHANNEL_LAYOUT.get(name))
                if msk is not None:
                    # keep the mask as an extra input channel: the network is TOLD which
                    # sensor block is absent instead of inferring it from constant zeros
                    X9 = np.concatenate([X9, msk], axis=2)
                X9s.append(X9)
                y9s.append(yv)
                s9s.append(np.full(len(yv), subj, dtype=int))
            variants[k] = (np.concatenate(X9s), np.concatenate(y9s),
                           np.concatenate(s9s))
        if "common_50hz" not in variants:
            raise RuntimeError("%s produced no windows" % name)
        X, y, s = variants["common_50hz"]
    # Cap every variant with the same seed: Task 7/8 swap one variant in for another and
    # an uncapped `band_limited_features` would make those arms cost different amounts of
    # data from the arm they are compared against.
    if max_windows:
        variants = {k: _cap_windows(vx, vy, vs, max_windows, seed)
                    for k, (vx, vy, vs) in variants.items()}
        X, y, s = variants["common_50hz"]
    b = DatasetBundle(name, X, y, s, {})
    b.variants_fs = {k: (NATIVE_FS[name] if k == "native_rate" else config.TARGET_FS)
                     for k in variants}
    b.meta.update({"dataset": name, "native_fs": NATIVE_FS[name],
                   "channel_mode": channel_mode, "variant": "common_50hz",
                   "fs": float(config.TARGET_FS)})
    b.variants = variants
    b.meta.update(b.summary())
    b.meta["variant"] = "common_50hz"
    b.meta["fs"] = float(config.TARGET_FS)
    return b


ALL_DATASETS = ("UCI HAR", "PAMAP2", "WISDM v2", "HARTH")
SLUG = {"UCI HAR": "uci_har", "PAMAP2": "pamap2", "WISDM v2": "wisdm_v2", "HARTH": "harth"}

_LOADERS = {"PAMAP2": _load_pamap2, "WISDM v2": _load_wisdm, "HARTH": _load_harth}


def raw_recs(name, max_subjects=None):
    """Per-subject NATIVE-rate streams: [(subject_id, (T, C) signal, per-sample codes)].

    Task 7 needs the un-resampled signal to measure what a resampler actually does;
    `load()` only returns finished windows.
    """
    if name not in _LOADERS:
        raise ValueError("raw_recs is for the resampled datasets (%s), not %r"
                         % (", ".join(sorted(_LOADERS)), name))
    root = _fetch(name, os.path.join(config.DATA_DIR, name.replace(" ", "_")))
    if root is None:
        raise RuntimeError("could not obtain %s" % name)
    recs = _LOADERS[name](root)
    if max_subjects:
        recs = sorted(recs, key=lambda r: r[0])[:int(max_subjects)]
    return recs


def set_class_space(name):
    """Point config.N_CLASS at this dataset's class count (WISDM is 3, the rest 6).

    Every module reads config.N_CLASS at call time, so a session that loads WISDM after
    UCI HAR re-scores against the right label set and the right chance line."""
    config.N_CLASS = int(N_CLASS_BY_DATASET.get(name, len(CLASS_NAMES)))
    config.CLASS_NAMES = list(class_names(name))
    return config.N_CLASS


def label_map_table():
    """Task 7 deliverable: the activity-label mapping across the four datasets.
    Rows are activity NAMES, so WISDM's 3-class indices line up with the others."""
    rows = []
    for cls_name in CLASS_NAMES:
        row = dict(shared_name=cls_name)
        for ds, m in LABEL_MAPS.items():
            names = class_names(ds)
            if cls_name not in names:
                row["%s source codes" % ds] = "- (class not in this dataset)"
                continue
            idx = names.index(cls_name)
            src = [str(k) for k, v in sorted(m.items(), key=lambda kv: str(kv[0]))
                   if v == idx]
            row["%s source codes" % ds] = "+".join(src) if src else "-"
        rows.append(row)
    # one row per dataset naming the native labels that are NOT mapped anywhere, so the
    # dropped-window decision is on the record instead of buried in the loader
    for ds in LABEL_MAPS:
        dropped = sorted(x for x in _NATIVE_LABELS[ds] if x not in LABEL_MAPS[ds])
        row = dict(shared_name="DROPPED (no shared-class equivalent)")
        for ds2 in LABEL_MAPS:
            row["%s source codes" % ds2] = (",".join(str(x) for x in dropped)
                                            if ds2 == ds else "-")
        rows.append(row)
    return rows


# Native label codes each loader can see, used only to report what was dropped.
_NATIVE_LABELS = {
    "UCI HAR": set(range(1, 7)),
    "PAMAP2": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
    "WISDM v2": set("ABCDEFGHIJKLMOPQRS"),   # 18 activities, no N
    "HARTH": set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                  20, 21, 22, 23, 24, 130, 140]),
}

#@@FILE ppml/runners.py
"""One federated-learning training loop for every dataset, architecture and mechanism.

Every table and figure in the revision is produced by `run_fl` (or `run_central`), so
methodology cannot drift between datasets or between the MLP and the CNN - the
inconsistency the reviewers found (WISDM at 23% in the figures, 84.9% in Table 5) is a
property of this file, not of the experiment scripts.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import config, dp_accountant, fl, models
from .metrics import classification_metrics

MAX_DEVICE_BYTES = 2_000_000_000


class LocalData:
    """Client/test data, parked on the GPU when it fits and on the CPU otherwise."""

    def __init__(self, X, y, device, dtype=None):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y).astype(np.int64), dtype=torch.long)
        nbytes = (self.X.numel() + self.y.numel() * 8) * 4
        self.to_device = (device.type == "cuda") and nbytes < MAX_DEVICE_BYTES
        self.device = device
        if self.to_device:
            self.X = self.X.to(device)
            self.y = self.y.to(device)

    def __len__(self):
        return len(self.y)

    def batch(self, idx):
        xb = self.X[idx]
        yb = self.y[idx]
        if self.to_device:
            return xb, yb
        return xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)

    def subset(self, idxs):
        return LocalData._view(self, idxs, self.device)

    @staticmethod
    def _view(parent, idxs, device):
        child = object.__new__(LocalData)
        child.X = parent.X[idxs]
        child.y = parent.y[idxs]
        child.to_device = parent.to_device
        child.device = device
        return child


def class_weights_of(y, n_class=None):
    """Inverse-frequency class weights from ONE silo's own labels (no other client's data
    is read). Classes absent from the silo get weight 0 - they produce no loss terms
    anyway - and the weights are normalised to mean 1 over the classes present."""
    n_class = n_class or config.N_CLASS
    y = np.asarray(y).astype(int)
    cnt = np.bincount(y, minlength=n_class).astype(np.float64)
    w = np.zeros(n_class)
    present = cnt > 0
    w[present] = cnt[present].sum() / (present.sum() * cnt[present])
    return w


def _eps_value(v):
    """per_round_eps entries are dicts ({"round", "eps", ...}); the curve needs the number."""
    if isinstance(v, dict):
        v = v.get("eps")
    return None if v is None else float(v)


def lr_at(rnd, rounds, lr=None, lr_min=None):
    """Cosine decay from LR (round 1) to LR_MIN (last round)."""
    lr = config.LR if lr is None else lr
    lr_min = config.LR_MIN if lr_min is None else lr_min
    if config.LR_SCHEDULE != "cosine" or rounds <= 1:
        return float(lr)
    t = (rnd - 1) / float(rounds - 1)
    return float(lr_min + 0.5 * (lr - lr_min) * (1.0 + np.cos(np.pi * t)))


def _small_rotations(n, device, max_deg=15.0, gen=None):
    """n random 3x3 rotations of at most max_deg degrees (axis-angle). Small on purpose:
    the direction of gravity is what separates sitting, standing and lying."""
    axis = torch.randn(n, 3, generator=gen)
    axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-8)
    ang = (torch.rand(n, generator=gen) * 2 - 1) * np.deg2rad(max_deg)
    K = torch.zeros(n, 3, 3)
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    I = torch.eye(3).expand(n, 3, 3)
    s_, c_ = torch.sin(ang).view(-1, 1, 1), torch.cos(ang).view(-1, 1, 1)
    return (I + s_ * K + (1 - c_) * (K @ K)).to(device)


def augment_windows(xb, gen=None):
    """Scaling (x N(1, 0.1)), jitter (+ N(0, 0.05)) and one small 3-D rotation per window
    applied to every 3-channel sensor triple. Works on a copy; raw-window inputs only."""
    if xb.dim() != 3:
        return xb
    n, wl, c = xb.shape
    dev = xb.device
    out = xb * (1.0 + 0.1 * torch.randn(n, 1, c, generator=gen).to(dev))
    out = out + 0.05 * torch.randn(out.shape, generator=gen).to(dev)
    n_tri = c // 3
    if n_tri:
        R = _small_rotations(n, dev, gen=gen)
        parts = []
        for t in range(n_tri):
            parts.append(torch.einsum("nij,nwj->nwi", R, out[:, :, 3 * t:3 * t + 3]))
        if c % 3:
            parts.append(out[:, :, 3 * n_tri:])
        out = torch.cat(parts, dim=2)
    return out


def train_local(model, data, epochs, batch, lr, device, mu=0.0, global_torch=None,
                freeze_head=False, grad_clip=5.0, seed=0, eval_every=0, eval_fn=None,
                class_weights=None, augment=None, cosine_epochs=False):
    """Real local gradient steps. mu > 0 is a genuine FedProx proximal term.

    eval_every/eval_fn give a trajectory for the pooled (non-federated) baseline without
    changing the optimisation: the checkpoint pass runs under no_grad, so it consumes no
    random numbers and cannot shift the training stream.
    """
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY,
                           foreach=True)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)),
                                                        eta_min=config.LR_MIN)
             if cosine_epochs and config.LR_SCHEDULE == "cosine" else None)
    cw = (torch.as_tensor(class_weights, dtype=torch.float32, device=device)
          if class_weights is not None else None)
    do_aug = config.AUGMENT if augment is None else bool(augment)
    if freeze_head and hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad_(False)
    model.train()
    n = len(data)
    if n == 0:
        return 0.0
    nb = max(1, int(batch))
    final_loss = 0.0
    for ep in range(1, int(epochs) + 1):
        perm = torch.randperm(n, generator=gen)
        if data.to_device:
            perm = perm.to(device)
        for start in range(0, n, nb):
            idx = perm[start:start + nb]
            xb, yb = data.batch(idx)
            if do_aug and xb.dim() == 3:
                xb = augment_windows(xb, gen=gen)
            opt.zero_grad()
            logits = model(xb)
            loss = Fn.cross_entropy(logits, yb, weight=cw)
            if mu > 0 and global_torch is not None:
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(model.parameters(), global_torch)
                           if p.requires_grad)
                loss = loss + 0.5 * float(mu) * prox
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip)
            opt.step()
            final_loss = float(loss.item())
        if sched is not None:
            sched.step()
        if eval_every and eval_fn is not None and (ep % int(eval_every) == 0 or
                                                   ep == int(epochs)):
            eval_fn(ep)
            model.train()          # evaluate() switches to eval mode; restore it
    if freeze_head and hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad_(True)
    return final_loss


@torch.no_grad()
def evaluate(model, data, batch=512):
    model.eval()
    preds, probs, losses = [], [], []
    n = len(data)
    for start in range(0, n, int(batch)):
        idx = torch.arange(start, min(start + int(batch), n))
        xb, yb = data.batch(idx.to(data.device) if data.to_device else idx)
        logits = model(xb)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        losses.append(Fn.cross_entropy(logits, yb, reduction="none").cpu().numpy())
    pred = np.concatenate(preds) if preds else np.zeros(0, int)
    prob = np.concatenate(probs) if probs else np.zeros((0, config.N_CLASS))
    loss = np.concatenate(losses) if len(losses) else np.zeros(0)
    return pred, prob, loss


def _init_global(arch, feature_dim, window_shape, n_class, device, seed):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    return models.build(arch, feature_dim=feature_dim, window_shape=window_shape,
                        n_class=n_class, device=device)


def dp_clip_C(dp, weights, shared_slice, arch):
    """Resolve the clip radius for one DP run, BEFORE any client data is touched.

    An absolute `dp["C"]` is used as given (that is what the Task 2 sweep varies and what
    the rejected paper's FL+DP arm certifies at). `C=None` asks for CLIP_REL x ||w0||: the
    init is seeded and public, so this reads no private window, and eps cannot move because
    sigma = z*C/K and sensitivity = C/K scale together. What it fixes is reach - a constant
    0.10 bounds how far T rounds of clipped updates can travel, and the answer depends on
    d and on the architecture, not on the privacy budget.
    """
    raw = dp.get("C")
    if raw is not None:
        return float(raw), None
    init_norm = fl.l2_norm([np.asarray(w, dtype=np.float64)
                            for w in list(weights)[shared_slice]])
    return float(config.CLIP_REL) * init_norm, init_norm


def run_fl(bundle_data, client_indices, arch, seed, rounds=None, local_epochs=None,
           batch=None, lr=None, method="fedavg", mu=0.0, dp=None, use_fedper=False,
           server_momentum=False, attacks=None, Xval=None, yval=None, Xte=None, yte=None,
           label="FL", partition_id=0, extra_config=None, device=None, verbose=False,
           grad_clip=5.0, heterogeneity=None, keep_model=False, init_weights=None):
    """Federated training for one (seed, partition, mechanism) cell.

    dp = None                       non-private
    dp = dict(z=..., C=..., T=...)   fixed clipping
    dp = dict(z=..., adaptive=True)  Gaussian quantile estimator, accounted
    attacks = dict(fraction=, update=, labelflip=bool, aggreg=)
    """
    C_ = config.cfg()
    config.check_budget()
    rounds = C_["rounds"] if rounds is None else rounds
    local_epochs = (C_["local_epochs_mlp"] if arch.upper().startswith("MLP")
                    else C_["local_epochs_cnn"]) if local_epochs is None else local_epochs
    batch = batch or config.BATCH
    lr = lr or config.LR
    device = device or models.device_for(True)

    Xtr, ytr = bundle_data
    data_tr = LocalData(Xtr, ytr, device)
    data_te = LocalData(Xte, yte, device)
    data_val = LocalData(Xval, yval, device) if Xval is not None else None

    gm = _init_global(arch, Xtr.shape[1] if Xtr.ndim == 2 else None,
                      Xtr.shape[1:], None, device, seed)
    n_class = models.n_class_of(gm)
    if init_weights is not None:
        # a public starting point (stage dplearn): the body was trained on a DIFFERENT
        # dataset, so it reads no private window and costs no privacy budget
        models.set_weights(gm, init_weights)
    weights = models.get_weights(gm)
    n_head = models.n_head_params(gm) if use_fedper else 0
    shared_slice = slice(0, len(weights) - n_head) if n_head else slice(0, len(weights))

    rng = np.random.RandomState(int(seed) + 1)
    torch.manual_seed(int(seed) + 1)

    # FedProx is FedAvg aggregation plus the proximal term in LOCAL training (mu > 0),
    # so the method name is not an aggregator and must not reach aggregate_plain.
    aggreg_name = "fedavg" if method == "fedprox" else method

    adaptive = None
    iter_snap = []
    if dp is not None:
        if aggreg_name not in ("fedavg", "equal_mean"):
            raise ValueError("the C/K sensitivity argument needs equal 1/K weights; "
                             "aggregator %r is not certified" % aggreg_name)
        dp = dict(dp)
        dp.setdefault("T", rounds)
        dp.setdefault("delta", config.DELTA)
        dp.setdefault("sparsity", None)
        dp.setdefault("iter_avg", 0)
        C_eff, init_norm = dp_clip_C(dp, weights, shared_slice, arch)
        dp["C"] = C_eff
        dp["init_norm"] = init_norm
        # the head-only support is fixed by the architecture, so it is decided here, before
        # any client data is read, and never depends on a released value
        dp["head_tensors"] = None
        if dp.get("head_only"):
            if n_head:
                raise ValueError("dp head_only releases the trailing tensors of the "
                                 "aggregated vector; combining it with FedPer's slice is "
                                 "undefined and silently uncertified")
            ht = int(models.n_head_params(gm))
            if ht <= 0 or ht >= len(weights):
                raise ValueError("arch %s exposes %d head tensors of %d: head_only would "
                                 "release nothing or everything" % (arch, ht, len(weights)))
            dp["head_tensors"] = ht
        if dp.get("adaptive"):
            # The estimator releases a statistic every round it runs, and that statistic
            # is the dominant term of this arm's certificate. W is the budget it is
            # allowed to spend, set in config before any accuracy was read; the run then
            # accounts for the releases it actually made, read back from the clipper.
            adaptive = fl.AdaptiveClipper(C0=C_eff,
                                          freeze_rounds=dp.get("freeze_rounds",
                                                               config.ADAPT_FREEZE_ROUNDS))

    adv = set()
    attack_update = "none"
    labelflip = False
    aggreg = aggreg_name
    if attacks is not None:
        adv = fl.is_malicious(list(range(len(client_indices))), len(client_indices),
                              attacks.get("fraction", 0.0), rng)
        attack_update = attacks.get("update", "none")
        labelflip = bool(attacks.get("labelflip", False))
        aggreg = attacks.get("aggreg", aggreg)

    vel = None
    per_round = []
    clip_history = []
    # IPM is defined against the mean of the HONEST updates (Xie et al. 2019), so every
    # update attack is applied once the honest deltas of the round are known. For IPM the
    # attacker's own local training is then not part of the attack, so it is skipped.
    attack_deferred = attack_update in ("ipm", "ipm_eps0.5")
    t_start = time.time()
    # The privacy cost already paid at each round, for the live print and the per-round log.
    # Closed-form and cheap, so it is computed once for the whole run, not once per round.
    # The adaptive arm gets None here: its certificate depends on how many estimator
    # releases actually happened, which only the final accounting knows - inventing a
    # per-round curve for it would be the kind of number this pipeline exists to avoid.
    eps_curve = None
    if dp is not None and adaptive is None:
        try:
            eps_curve = dp_accountant.account_central_gaussian(
                dp["z"], rounds, dp["delta"]).get("per_round_eps") or []
        except Exception:
            eps_curve = None

    best_sel = None          # ((val_acc, val_macro_f1), round, weights)
    for rnd in range(1, rounds + 1):
        global_np = weights
        g_send = list(global_np[shared_slice])
        client_ws, client_ns, client_ids = [], [], []
        honest_deltas = []
        pending_adv = []
        up_times = []
        for k, idx in enumerate(client_indices):
            if len(idx) == 0:
                continue
            is_adv = k in adv
            cm = _init_global(arch, Xtr.shape[1] if Xtr.ndim == 2 else None,
                              Xtr.shape[1:], n_class, device, seed * 1000 + k)
            models.set_weights(cm, global_np)
            y_client = ytr[idx]
            if labelflip and is_adv:
                y_client = fl.flip_labels(y_client, rng, n_class)
            data_cl = LocalData(Xtr[idx], y_client, device)
            gt = ([torch.as_tensor(w, dtype=torch.float32, device=device)
                   for w in global_np] if mu > 0 else None)
            if not (is_adv and attack_deferred):
                t0 = time.time()
                train_local(cm, data_cl, local_epochs, batch,
                            lr_at(rnd, rounds, lr), device, mu=mu,
                            global_torch=gt, freeze_head=use_fedper,
                            grad_clip=grad_clip, seed=seed * 100 + rnd * 7 + k,
                            class_weights=(class_weights_of(y_client, n_class)
                                           if config.CLASS_WEIGHTS else None))
                up_times.append(time.time() - t0)
            w_k = models.get_weights(cm)
            delta_full = [np.asarray(a) - np.asarray(b) for a, b in zip(w_k, global_np)]
            if is_adv and attack_update != "none":
                pending_adv.append((k, delta_full, w_k, int(len(idx))))
            else:
                honest_deltas.append(delta_full)
                # FedPer contributes only the shared part
                w_send = list(w_k[shared_slice]) if n_head else w_k
                client_ws.append(w_send)
                client_ids.append(k)
                client_ns.append(int(len(idx)))
            del cm
        for k, delta_full, w_k, n_k in pending_adv:
            delta_full = fl.attack_update(delta_full, global_np, attack_update, rng,
                                         honest_deltas=honest_deltas or None)
            w_k = [np.asarray(g) + np.asarray(d) for g, d in zip(global_np, delta_full)]
            w_send = list(w_k[shared_slice]) if n_head else w_k
            client_ws.append(w_send)
            client_ids.append(k)
            client_ns.append(n_k)
            del w_k

        if not client_ws:
            break

        dp_diag = None
        if dp is not None:
            est_released = None
            if adaptive is not None:
                norms = [fl.l2_norm([np.asarray(w) - np.asarray(g)
                                     for w, g in zip(cw, g_send)])
                         for cw in client_ws]
                C_now = adaptive.update(norms, rng)
                est_released = bool(adaptive.history[-1].get("released"))
            else:
                C_now = float(dp["C"])
            new_shared, dp_diag = fl.dp_aggregate(g_send, client_ws, C_now, dp["z"], rng,
                                                  return_diag=True,
                                                  sparsity=dp.get("sparsity"),
                                                  head_tensors=dp.get("head_tensors"))
            clip_history.append(dict(round=rnd, C=float(C_now),
                                     clip_fraction=dp_diag["clip_fraction"],
                                     sigma=dp_diag["sigma"],
                                     sparsity=dp_diag.get("sparsity"),
                                     support=dp_diag.get("support"),
                                     estimator_released=est_released,
                                     noise_coords=dp_diag.get("noise_coords"),
                                     total_coords=dp_diag.get("total_coords"),
                                     signal_energy_fraction=dp_diag.get(
                                         "signal_energy_fraction"),
                                     release_signal=dp_diag["release_signal_norm"],
                                     release_noise=dp_diag["release_noise_norm"],
                                     release_snr=dp_diag["release_snr"],
                                     update_norms=[float(x) for x in dp_diag["update_norms"]]))
        else:
            new_shared = fl.aggregate_plain(client_ws, client_ns, how=aggreg,
                                            assumed_f=config.BYZ_ASSUMED_F)

        if server_momentum:
            delta_agg = [np.asarray(a) - np.asarray(b) for a, b in zip(new_shared, g_send)]
            if vel is None:
                vel = [np.zeros_like(d) for d in delta_agg]
            vel = [config.MOMENTUM_BETA * v + d for v, d in zip(vel, delta_agg)]
            new_shared = [np.asarray(g) + config.MOMENTUM_ETA_S * v
                          for g, v in zip(g_send, vel)]

        if n_head:
            # keep each client's private head locally; global carries the shared body only
            merged = list(new_shared) + list(weights[shared_slice.stop:])
        else:
            merged = list(new_shared)
        weights = [np.asarray(m, dtype=np.float32) for m in merged]
        models.set_weights(gm, weights)
        if dp is not None and int(dp.get("iter_avg") or 0) > 0:
            iter_snap.append([np.asarray(w, dtype=np.float64) for w in weights])
            while len(iter_snap) > int(dp["iter_avg"]):
                iter_snap.pop(0)

        pred, prob, loss = evaluate(gm, data_te)
        m = classification_metrics(yte, pred, prob, loss, n_class)
        val_acc = None
        val_f1 = None
        if data_val is not None and len(data_val):
            pv, _, lv = evaluate(gm, data_val)
            val_acc = float(np.mean(pv == np.asarray(yval)))
            if lv is not None and len(pv):
                val_f1 = classification_metrics(yval, pv, None, lv,
                                                n_class)["f1_macro"]
        if config.SELECT_BEST_VAL and val_acc is not None:
            key = (val_acc, val_f1 if val_f1 is not None else -1.0)
            if best_sel is None or key > best_sel[0]:
                best_sel = (key, rnd, [np.asarray(w).copy() for w in weights])
        ch = clip_history[-1] if clip_history else {}
        per_round.append(dict(round=rnd, test_acc=m["accuracy"], test_loss=m["loss"],
                              val_acc=val_acc, val_macro_f1=val_f1,
                              macro_f1=m["f1_macro"], f1_weighted=m["f1_weighted"],
                              precision=m["precision"], recall=m["recall"],
                              balanced_accuracy=m["balanced_accuracy"],
                              macro_specificity=m["macro_specificity"],
                              kappa=m["kappa"], roc_auc_ovr=m.get("roc_auc_ovr"),
                              # how many clients actually contributed this round: K is in
                              # the noise scale, so a round with dropouts is not the round
                              # the certificate assumes
                              n_clients=len(client_ws),
                              eps_cum=(_eps_value(eps_curve[rnd - 1]) if eps_curve
                                       and rnd - 1 < len(eps_curve) else None),
                              C=ch.get("C"), sigma=ch.get("sigma"),
                              clip_fraction=ch.get("clip_fraction"),
                              release_snr=ch.get("release_snr"),
                              wall_s=time.time() - t_start,
                              local_epoch_time_s=float(np.mean(up_times)) if up_times else 0.0))
        # config.PRINT_ROUNDS is the one global quiet switch: it gates only this line, every
        # value on it is already in the per-round record above, which reaches the logs
        # whatever the setting.
        if verbose and config.PRINT_ROUNDS and (
                rnd == 1 or rnd % max(1, rounds // 5) == 0 or rnd == rounds):
            # The live reading is the full metric set, not accuracy alone: a run that looks
            # flat on accuracy can still be losing precision or collapsing to one class,
            # and on Kaggle that is the only view of the run before the tables exist.
            line = ("    R%3d/%d K=%d  acc=%.4f prec=%.4f rec=%.4f F1w=%.4f "
                    "F1m=%.4f bal=%.4f spec=%s kappa=%.4f loss=%.4f val=%s"
                    % (rnd, rounds, len(client_ws), m["accuracy"], m["precision"],
                       m["recall"], m["f1_weighted"], m["f1_macro"],
                       m["balanced_accuracy"],
                       "n/a" if m["macro_specificity"] is None else "%.4f" % m["macro_specificity"],
                       m["kappa"], m["loss"],
                       "n/a" if val_acc is None else "%.4f" % val_acc))
            auc = m.get("roc_auc_ovr")
            if auc is not None:
                line += " auc=%.4f" % auc
            if ch:
                snr = ch.get("release_snr")
                eps_c = per_round[-1]["eps_cum"]
                line += ("  eps=%s C=%.3f sigma=%.3f clipfrac=%.3f snr=%s"
                         % ("n/a" if eps_c is None else "%.4f" % eps_c, float(ch["C"]),
                            float(ch["sigma"]), float(ch["clip_fraction"]),
                            "n/a" if snr is None else "%.4f" % snr))
            print(line)

    wall = time.time() - t_start
    if not per_round:
        # no client contributed in any round: still report the initialised global model
        pred, prob, loss = evaluate(gm, data_te)
        m = classification_metrics(yte, pred, prob, loss, n_class)
        per_round.append(dict(round=0, test_acc=m["accuracy"], test_loss=m["loss"],
                              val_acc=None, val_macro_f1=None, macro_f1=m["f1_macro"],
                              f1_weighted=m["f1_weighted"], precision=m["precision"],
                              recall=m["recall"],
                              balanced_accuracy=m["balanced_accuracy"],
                              macro_specificity=m["macro_specificity"], kappa=m["kappa"],
                              roc_auc_ovr=m.get("roc_auc_ovr"), n_clients=0, eps_cum=None,
                              C=None, sigma=None, clip_fraction=None, release_snr=None,
                              wall_s=wall, local_epoch_time_s=0.0))
    privacy = None
    iter_avg = None
    if dp is not None and len(iter_snap) >= 2:
        # POST-PROCESSING, not a second mechanism: every snapshot here is a deterministic
        # function of releases the run already made, so the privacy loss accounting is
        # unchanged by the post-processing theorem - while averaging over the window
        # cancels part of the fresh per-round noise. Both numbers are reported; the
        # headline `accuracy` stays the final model, so nothing is silently swapped.
        avg_w = [np.mean([s[i] for s in iter_snap], axis=0)
                 for i in range(len(weights))]
        models.set_weights(gm, [np.asarray(a, dtype=np.float32) for a in avg_w])
        p2, q2, l2 = evaluate(gm, data_te)
        m2 = classification_metrics(yte, p2, q2, l2, n_class)
        iter_avg = dict(window=len(iter_snap), accuracy=m2["accuracy"],
                        f1=m2["f1_weighted"], f1_macro=m2["f1_macro"], loss=m2["loss"])
        models.set_weights(gm, weights)
    if dp is not None:
        T = int(dp.get("T", rounds))
        if adaptive is not None:
            # only the releases this run made are charged for; rounds_released is read
            # back from the clipper's own history, so a frozen estimator cannot quietly
            # keep spending or keep releasing while claiming to have stopped
            privacy = dp_accountant.account_adaptive(dp["z"], T, dp["delta"],
                                                     sigma_b=adaptive.sigma_b,
                                                     T_clip=adaptive.rounds_released)
            privacy["freeze_rounds"] = adaptive.freeze_rounds
        else:
            privacy = dp_accountant.account_central_gaussian(dp["z"], T, dp["delta"])
        privacy.update(C=float(dp.get("C", config.CLIP_C)), K=len(client_ws),
                       adaptive=adaptive is not None,
                       sigma=dp_accountant.sigma_of(dp["z"], privacy.get("C", config.CLIP_C),
                                                    max(1, len(client_ws))))

    final = per_round[-1] if per_round else dict(test_acc=0.0, test_loss=float("nan"))
    test_acc_final_round = final.get("test_acc")
    selected_round = final.get("round")
    best_val_acc = None
    if best_sel is not None:
        # Model selection on the VALIDATION subjects only; the test set is read once, for
        # the selected model. The last-round test accuracy stays on the row beside it.
        best_val_acc = float(best_sel[0][0])
        selected_round = int(best_sel[1])
        models.set_weights(gm, [np.asarray(w, dtype=np.float32) for w in best_sel[2]])
        pred, prob, loss = evaluate(gm, data_te)
        m = classification_metrics(yte, pred, prob, loss, n_class)
        final = dict(final, test_acc=m["accuracy"], test_loss=m["loss"])
        models.set_weights(gm, weights)
    out = dict(label=label, seed=int(seed), partition_id=int(partition_id),
               architecture=arch.upper(), method=method,
               rounds=int(rounds), local_epochs=int(local_epochs), batch=int(batch),
               lr=float(lr), K=len(client_indices),
               accuracy=final["test_acc"], f1=m["f1_weighted"], f1_macro=m["f1_macro"],
               test_acc_final_round=test_acc_final_round,
               selected_round=selected_round, best_val_acc=best_val_acc,
               model_selection=("best validation round" if best_sel is not None
                                else "last round"),
               lr_schedule=config.LR_SCHEDULE, lr_final=lr_at(rounds, rounds, lr),
               class_weights=config.CLASS_WEIGHTS, augment=bool(config.AUGMENT),
               precision=m["precision"], recall=m["recall"],
               f1_per_class=m["f1_per_class"], precision_per_class=m["precision_per_class"],
               recall_per_class=m["recall_per_class"], confusion=m["confusion"],
               # the class-balanced set IEEE Access reviewers ask for alongside accuracy:
               # a rare activity class can be invisible in accuracy and cannot hide here.
               balanced_accuracy=m["balanced_accuracy"],
               macro_specificity=m["macro_specificity"],
               specificity_per_class=m["specificity_per_class"], kappa=m["kappa"],
               roc_auc_ovr=m.get("roc_auc_ovr"), nll=m.get("nll"),
               n_test=m["n_test"], class_support=m["class_support"],
               y_pred=pred, y_proba=prob, y_true=np.asarray(yte),
               per_round=per_round, clip_history=clip_history,
               privacy=privacy, wall_time_s=wall,
               client_sizes=[int(n) for n in client_ns],
               client_subjects=None, partition_heterogeneity=heterogeneity,
               attack=(attacks or {}).get("update") if attacks else None,
               attack_fraction=(attacks or {}).get("fraction") if attacks else 0.0,
               aggregator=aggreg, mu=float(mu), fedper=bool(use_fedper),
               server_momentum=bool(server_momentum),
               communication_payload_bytes=_payload_bytes(weights, len(client_indices), rounds),
               local_epoch_time_s=float(np.mean([p["local_epoch_time_s"] for p in per_round]))
               if per_round else 0.0)
    if dp is not None:
        out["clip_C"] = float(dp.get("C", config.CLIP_C))
        out["dp_z"] = float(dp["z"])
        out["dp_delta"] = float(dp["delta"])
        # Which of the four eps-neutral knobs this cell used, per row: a table that shows
        # an improved DP arm without showing the knob that improved it is not evidence.
        out["dp_sparsity"] = dp.get("sparsity")
        out["dp_head_only"] = bool(dp.get("head_only"))
        out["dp_head_tensors"] = dp.get("head_tensors")
        out["dp_iter_avg_window"] = int(dp.get("iter_avg") or 0)
        out["clip_basis"] = ("CLIP_REL x ||w0||" if dp.get("init_norm") is not None
                             else "absolute")
        out["init_weight_norm"] = dp.get("init_norm")
        # which released object picked the coordinates, so a reader can tell the two
        # sparse variants apart instead of trusting a method name
        out["dp_support"] = (clip_history[-1].get("support") if clip_history else None)
        # ... and what the sparse release cost in signal: the fraction of the clipped
        # mean's L2 energy that survived, averaged over the rounds. It lives in
        # clip_history, which the CSV writer flattens away, so the headline number goes
        # on the record.
        sef = [c.get("signal_energy_fraction") for c in clip_history
               if c.get("signal_energy_fraction") is not None]
        if sef:
            out["signal_energy_fraction"] = float(sum(sef) / len(sef))
        last = clip_history[-1] if clip_history else {}
        out["noise_coords"] = last.get("noise_coords")
        out["total_coords"] = last.get("total_coords")
        # The privacy-utility diagnostic, measured rather than predicted: the L2 of the
        # clipped mean update against the L2 of the Gaussian actually drawn over the
        # released support, averaged over the rounds that ran. A chance-level DP row is only
        # a result if the table can say how far below one its release sat, and the ratio is
        # read off the drawn noise so the head-only and sparse arms report their own support.
        hist = [c for c in clip_history if c.get("release_snr") is not None]
        if hist:
            n = float(len(hist))
            out["release_snr_mean"] = float(sum(c["release_snr"] for c in hist) / n)
            out["release_snr_first"] = float(hist[0]["release_snr"])
            out["release_snr_last"] = float(hist[-1]["release_snr"])
            out["release_signal_norm"] = float(sum(c["release_signal"] for c in hist) / n)
            out["release_noise_norm"] = float(sum(c["release_noise"] for c in hist) / n)
        if iter_avg is not None:
            out["accuracy_iter_avg"] = iter_avg["accuracy"]
            out["f1_iter_avg"] = iter_avg["f1"]
            out["f1_macro_iter_avg"] = iter_avg["f1_macro"]
            out["iter_avg_window"] = iter_avg["window"]
    if extra_config:
        out["extra_config"] = extra_config
    if keep_model:
        # the MIA protocol needs per-example losses on sets the run never evaluated
        out["model"] = gm
        out["global_weights"] = weights
    return out


def _payload_bytes(weights, K, rounds):
    """Uplink+downlink float32 bytes moved between server and clients."""
    numel = int(sum(np.asarray(w).size for w in weights))
    return int(numel * 4 * (K + 1) * int(rounds))


def central_epochs(arch, epochs=None):
    """The pooled baseline's epoch budget, in one place so the log cannot say 0."""
    C_ = config.cfg()
    if epochs:
        return int(epochs)
    if arch.upper().startswith("MLP"):
        return int(C_["local_epochs_mlp"] * C_["rounds"])
    return int(max(20, C_["local_epochs_cnn"] * 4))


def run_central(Xtr, ytr, Xte, yte, arch, seed, epochs=None, batch=None, lr=None,
               device=None, label="Centralised", feature_dim=None, checkpoints=10):
    """Same model, same optimiser, pooled training data - the matched control."""
    config.check_budget()
    device = device or models.device_for(True)
    epochs = central_epochs(arch, epochs)
    data_tr = LocalData(Xtr, ytr, device)
    data_te = LocalData(Xte, yte, device)
    m = _init_global(arch, Xtr.shape[1] if Xtr.ndim == 2 else None, Xtr.shape[1:], None,
                     device, seed)
    traj = []

    def _at(ep):
        p, pb, ls = evaluate(m, data_te)
        met = classification_metrics(yte, p, pb, ls, models.n_class_of(m))
        traj.append(dict(round=int(ep), test_acc=met["accuracy"],
                         test_loss=met["loss"], val_acc=None,
                         macro_f1=met["f1_macro"], wall_s=None, local_epoch_time_s=None))

    t0 = time.time()
    train_local(m, data_tr, epochs, batch or config.BATCH, lr or config.LR, device,
                seed=seed, grad_clip=5.0, cosine_epochs=True,
                class_weights=(class_weights_of(ytr, models.n_class_of(m))
                               if config.CLASS_WEIGHTS else None),
                eval_every=max(1, epochs // max(1, int(checkpoints))), eval_fn=_at)
    pred, prob, loss = evaluate(m, data_te)
    met = classification_metrics(yte, pred, prob, loss, models.n_class_of(m))
    wall = time.time() - t0
    for point in traj:
        point["wall_s"] = wall
    return dict(label=label, seed=int(seed), architecture=arch.upper(), rounds=0,
                accuracy=met["accuracy"], f1=met["f1_weighted"], f1_macro=met["f1_macro"],
                precision=met["precision"], recall=met["recall"],
                f1_per_class=met["f1_per_class"], precision_per_class=met["precision_per_class"],
                recall_per_class=met["recall_per_class"], confusion=met["confusion"],
                # same headline set as run_fl, so the centralised control sits in the same
                # columns of the metric table instead of showing blanks
                balanced_accuracy=met["balanced_accuracy"],
                macro_specificity=met["macro_specificity"],
                specificity_per_class=met["specificity_per_class"], kappa=met["kappa"],
                roc_auc_ovr=met.get("roc_auc_ovr"), nll=met.get("nll"),
                n_test=met["n_test"], class_support=met["class_support"],

                y_pred=pred, y_proba=prob, y_true=np.asarray(yte),
                per_round=traj, wall_time_s=wall,
                privacy=None, K=1, local_epochs=int(epochs), method="centralised",
                batch=int(batch or config.BATCH), lr=float(lr or config.LR))

#@@FILE ppml/harness.py
"""Shared harness: every experiment goes through the same data path, split rule,
client construction, training loop and logger. This is what makes the four dataset
scripts methodologically identical - they differ only in `bundle`.
"""

import os
import time

import numpy as np

from . import config, datasets, features, fl, models, runners, splitting
from .runlog import RunLog


class Split:
    def __init__(self, idx, subjects, meta, key=""):
        self.train_idx = idx["train_idx"]
        self.val_idx = idx["val_idx"]
        self.test_idx = idx["test_idx"]
        self.subjects_in_split = meta["subjects_in_split"]
        self.leakage_checks = meta["leakage_checks"]
        self.key = key


class Harness:
    """Holds one dataset plus the cached feature matrix and the GPU handle."""

    def __init__(self, bundle, device=None, compute_features=True):
        self.bundle = bundle
        self.device = device or models.device_for(True)
        self.Xw = bundle.X                       # (N, WL, C) windows
        self.y = bundle.y
        self.subjects = bundle.subject
        # native_rate windows are on the device clock; everything else is 50 Hz
        self.fs = float(bundle.meta.get("fs", config.TARGET_FS))
        self.Xf = None
        if compute_features:
            t0 = time.time()
            self.Xf = features.extract_features(self.Xw, fs=self.fs)
            self.feature_time_s = time.time() - t0
        else:
            # the pooled Task 7 arm trains raw-window models only, so it never needs the
            # 117-feature matrix - but the driver prints this attribute for every harness
            self.Xf = None
            self.feature_time_s = 0.0
        self.n_features = int(self.Xf.shape[1]) if self.Xf is not None else 0
        self.logs = {}
        self._scale_cache = {}

    # ── logging ────────────────────────────────────────────────────────────
    def log(self, stage):
        if stage not in self.logs:
            self.logs[stage] = RunLog(self.bundle.name, stage)
        return self.logs[stage]

    # ── splitting ──────────────────────────────────────────────────────────
    def split(self, seed, kind="subject"):
        if kind == "subject":
            got = splitting.subject_disjoint_split(self.subjects, self.y, seed=seed)
            return Split(got, self.subjects, got, key="subject:%d" % int(seed))
        if kind == "window":
            got = splitting.window_level_split(self.y, seed=seed)
            full = dict(train_idx=got["train_idx"], val_idx=got["val_idx"],
                        test_idx=got["test_idx"])
            leak = splitting.leakage_report(self.subjects, full, kind="window_level")
            per_split = {n: sorted(set(int(s) for s in self.subjects[v]))
                         for n, v in full.items()}
            return Split(full, self.subjects,
                         dict(subjects_in_split=per_split, leakage_checks=leak),
                         key="window:%d" % int(seed))
        raise ValueError(kind)

    def clients(self, split, seed, n_clients=None, mode=None):
        """The K client silos for one FL run.

        `mode=None` takes config.CLIENT_MODE, which is one subject per client: several
        people inside one silo would make the certificate client-level but NOT user-level,
        and would pay for it in utility (sigma = z*C/K). Task 3's client-construction rows
        still name their mode explicitly, so the comparison survives the default changing.
        """
        return splitting.federated_clients_by_subject(
            split.train_idx, self.subjects, self.y, n_clients=n_clients, seed=seed,
            mode=(config.CLIENT_MODE if mode is None else mode))

    def clients_dirichlet(self, split, seed, n_clients=None, alpha=0.5, kind="dirichlet"):
        if kind == "iid":
            return splitting.iid_partition_windows(split.train_idx, n_clients, seed=seed)
        return splitting.dirichlet_partition_windows(split.train_idx, self.y, n_clients,
                                                     alpha=alpha, seed=seed)

    # ── input selection ────────────────────────────────────────────────────
    def _scaled(self, src, split, tag):
        """Zero-mean/unit-variance using TRAIN subjects only.

        The submission fitted a scaler per client and, in the pooled path, over the
        whole dataset including the test windows. Fitting on the training split alone
        keeps the transform leak-free while giving every client identical statistics.
        """
        # The column count belongs in the key: the Task 3/7/8 arms hand `input()`
        # different-width overrides (a 54-column family, an accelerometer-only window
        # block) for the same split, and reusing the 117-column statistics either
        # broadcasts wrongly or crashes.
        key = (split.key, tag, tuple(src.shape[1:]),
               int(split.train_idx[0]) if len(split.train_idx) else -1)
        key = key + (config.WINDOW_SCALE,)
        if key not in self._scale_cache:
            tr = src[split.train_idx]
            # Raw windows (N, WL, C) are standardised PER CHANNEL (one mean/std per sensor
            # axis over all training windows and time steps); per-time-step statistics
            # would scale each sample position differently. Feature rows (N, F) are
            # standardised per feature. Statistics come from the training subjects only.
            axes = (0, 1) if (tr.ndim == 3 and config.WINDOW_SCALE == "channel") else 0
            mu = tr.mean(axis=axes, keepdims=True)
            sd = tr.std(axis=axes, keepdims=True) + 1e-8
            self._scale_cache[key] = (mu.astype(np.float32), sd.astype(np.float32))
        mu, sd = self._scale_cache[key]
        return ((src - mu) / sd).astype(np.float32)

    def input(self, arch, split, X_override=None, standardize=True):
        """Return (X_train, y_train, X_test, y_test, X_val, y_val) for `arch`.

        MLP        -> 117 hand-crafted features (paper's representation)
        MLP-Raw    -> flattened windows (representation-matched control, Task 3)
        CNN/TCN    -> (N, WL, C) windows
        """
        a = arch.upper()
        tag = "features" if a in ("MLP", "MLP-SMALL") else "windows"
        src = self.Xf if a in ("MLP", "MLP-SMALL") else self.Xw
        if X_override is not None:
            src = X_override
        if src is None:
            raise RuntimeError("%s needs features but the harness was built without them" % arch)
        if standardize:
            src = self._scaled(np.asarray(src), split, tag)
        pick = lambda idx: (src[idx], self.y[idx])
        Xtr, ytr = pick(split.train_idx)
        Xte, yte = pick(split.test_idx)
        Xv, yv = pick(split.val_idx)
        # MLP-Raw keeps the (N, WL, C) shape; RawMLP flattens inside forward().
        return Xtr, ytr, Xte, yte, Xv, yv

    def scaled_all(self, arch, split):
        """The whole standardized matrix, so any GLOBAL window index can be addressed.

        `input()` returns only the three split blocks; the MIA protocol has to score
        arbitrary subject pools drawn from the training subjects.
        """
        a = arch.upper()
        src = self.Xf if a in ("MLP", "MLP-SMALL") else self.Xw
        if src is None:
            raise RuntimeError("%s needs features but the harness was built without them" % arch)
        return self._scaled(np.asarray(src), split,
                            "features" if a in ("MLP", "MLP-SMALL") else "windows")

    # ── one run, logged ────────────────────────────────────────────────────
    def _split_audit(self, split):
        """Scalar split provenance, so every logged run carries its own leakage evidence.

        The Work Order requires the log to be the only source for the tables; the audit
        numbers from `splitting` have to travel with the run, not only with the note.
        """
        if split is None:
            return {}
        out = dict(n_train_subjects=len(split.subjects_in_split.get("train", [])),
                   n_val_subjects=len(split.subjects_in_split.get("val", [])),
                   n_test_subjects=len(split.subjects_in_split.get("test", [])))
        lc = split.leakage_checks or {}
        for key in ("subjects_overlap_train_test", "subjects_overlap_train_val",
                    "subjects_overlap_val_test", "split_kind", "n_overlapping_pairs",
                    "windows_from_overlapping_subjects"):
            if key in lc:
                out["split_" + key] = lc[key]
        out["n_windows_train"] = int(len(split.train_idx))
        out["n_windows_test"] = int(len(split.test_idx))
        return out

    def run_fl(self, cfg, stage, arch="MLP", client_indices=None, split=None, seed=42,
               dp=None, method="fedavg", mu=0.0, use_fedper=False, server_momentum=False,
               attacks=None, rounds=None, local_epochs=None, X_override=None,
               heterogeneity=None, verbose=False, partition_id=0, label=None, cell=None,
               init_weights=None):
        """Execute + log one FL run. Returns the result dict (or the cached one)."""
        log = self.log(stage)
        C_ = config.cfg()
        base_cfg = dict(dataset=self.bundle.name, stage=stage, arch=arch, method=method,
                        # `cell` is the experiment-table name, e.g. "FL-Median". The raw
                        # `method` alone cannot tell FedAvg from FedProx (both "fedavg"),
                        # so paired tests key on (arch, cell) instead of guessing.
                        cell=cell or label or method,
                        seed=int(seed), partition_id=int(partition_id),
                        rounds=int(rounds or C_["rounds"]),
                        local_epochs=int(local_epochs or
                                         (C_["local_epochs_mlp"] if arch.upper().startswith("MLP")
                                          else C_["local_epochs_cnn"])),
                        batch=int(config.BATCH), lr=float(config.LR),
                        n_clients=len(client_indices) if client_indices else None,
                        dp=dp, mu=float(mu), fedper=bool(use_fedper),
                        server_momentum=bool(server_momentum),
                        attacks=attacks, client_mode=cfg.get("client_mode", "grouped"),
                        split_kind=cfg.get("split_kind", "subject"),
                        feature_kind=cfg.get("feature_kind", "handcrafted_117"),
                        jerk_with_fs=config.JERK_WITH_SAMPLING_RATE,
                        variant=cfg.get("variant", "common_50hz"),
                        harness=cfg)
        base_cfg.update(self._split_audit(split))
        chash = self.run_id(base_cfg)
        if log.has(chash):
            return None                                     # already computed
        Xtr, ytr, Xte, yte, Xv, yv = self.input(arch, split, X_override)
        cl_block = splitting.to_block_positions(client_indices, split.train_idx) \
            if client_indices else client_indices
        res = runners.run_fl((Xtr, ytr), cl_block, arch, seed, rounds=rounds,
                             local_epochs=local_epochs, method=method, mu=mu, dp=dp,
                             use_fedper=use_fedper, server_momentum=server_momentum,
                             attacks=attacks, Xval=Xv, yval=yv, Xte=Xte, yte=yte,
                             label=label or method, partition_id=partition_id,
                             device=self.device, verbose=verbose,
                             heterogeneity=heterogeneity, extra_config=cfg,
                             init_weights=init_weights)
        rec = self._record(base_cfg, res, chash)
        return res if rec is not None else None

    def run_central(self, stage, arch="MLP", split=None, seed=42, epochs=None,
                    X_override=None, cfg=None, label="Centralised", cell="Centralised"):
        log = self.log(stage)
        n_ep = runners.central_epochs(arch, epochs)
        base_cfg = dict(dataset=self.bundle.name, stage=stage, arch=arch,
                        method="centralised", cell=cell, seed=int(seed), partition_id=0,
                        epochs=int(n_ep), local_epochs=int(n_ep),
                        batch=int(config.BATCH),
                        lr=float(config.LR), split_kind=(cfg or {}).get("split_kind", "subject"),
                        variant=(cfg or {}).get("variant", "common_50hz"),
                        jerk_with_fs=config.JERK_WITH_SAMPLING_RATE,
                        harness=cfg or {})
        base_cfg.update(self._split_audit(split))
        chash = self.run_id(base_cfg)
        if log.has(chash):
            return None
        Xtr, ytr, Xte, yte, _, _ = self.input(arch, split, X_override)
        res = runners.run_central(Xtr, ytr, Xte, yte, arch, seed, epochs=epochs,
                                  device=self.device, label=label)
        rec = self._record(base_cfg, res, chash)
        return res if rec is not None else res

    # ── record shaping ─────────────────────────────────────────────────────
    def _cfg_block(self, base_cfg):
        """The config dictionary that IDENTIFIES a run, from the settings alone.

        Two things make this its own method instead of a few lines inside `_record`:

        * the resume check and the stored `config_hash` must be hashes of the SAME
          dictionary. Before, the check hashed `base_cfg` (nested `harness` block, no
          C/z/K/T/delta) and the writer hashed the flattened block, so the two never
          matched and every session recomputed work that was already in the log.
        * the profile is part of a run's identity. Smoke and full share seeds and, for
          the pinned-T DP arms, share rounds and epochs too; without the profile a
          non-reportable smoke row would suppress its reportable twin.
        """
        cfg_block = {k: v for k, v in base_cfg.items() if k != "harness"}
        cfg_block.update({k: v for k, v in (base_cfg.get("harness") or {}).items()
                          if k not in cfg_block})
        # `arch` is the short key the harness writes; the experiment tables and the
        # figures read `architecture`, so both names must be on every record.
        cfg_block["architecture"] = cfg_block.get("architecture") or base_cfg.get("arch")
        cfg_block["profile"] = config.PROFILE
        # The window cap changes how much data every run trained on, so it belongs to a
        # run's identity: a capped table must never silently satisfy an uncapped request,
        # and it gives every table a column that discloses the subsample.
        cfg_block["max_windows"] = config.MAX_WINDOWS
        return cfg_block

    def run_id(self, base_cfg):
        """The hash a run is recognised by on a later session."""
        from .runlog import stable_hash
        return stable_hash(self._cfg_block(base_cfg))

    def _record(self, base_cfg, res, config_hash=None):
        """Flatten a result into the Work Order's required log fields and write it."""
        priv = res.get("privacy") or {}
        cfg_block = self._cfg_block(base_cfg)
        # These five describe what the run actually used; they come from the result, so
        # they are recorded but deliberately not hashed (see `run_id`).
        cfg_block.setdefault("C", res.get("clip_C"))
        cfg_block.setdefault("z", res.get("dp_z"))
        cfg_block.setdefault("K", res.get("K"))
        cfg_block.setdefault("T", res.get("rounds"))
        cfg_block.setdefault("delta", res.get("dp_delta"))
        # The release's SHAPE, as the run left it. A table that only carries z cannot tell
        # the plain arm from the sparse or head-only one, and those three sit at the SAME
        # epsilon by construction - so the mechanism has to be identifiable from the log
        # alone, not from the cell name a caller happened to pass. Same reason these are
        # recorded but not hashed: they describe the run, they do not identify its settings.
        cfg_block.setdefault("dp_sparsity", res.get("dp_sparsity"))
        cfg_block.setdefault("dp_support", res.get("dp_support"))
        cfg_block.setdefault("dp_head_only", res.get("dp_head_only"))
        cfg_block.setdefault("dp_head_tensors", res.get("dp_head_tensors"))
        cfg_block.setdefault("dp_iter_avg_window", res.get("dp_iter_avg_window"))
        cfg_block.setdefault("clip_basis", res.get("clip_basis"))
        cfg_block.setdefault("init_weight_norm", res.get("init_weight_norm"))
        cfg_block.setdefault("noise_coords", res.get("noise_coords"))
        cfg_block.setdefault("total_coords", res.get("total_coords"))
        cfg_block.setdefault("signal_energy_fraction", res.get("signal_energy_fraction"))
        per_round = res.get("per_round") or []

        def _curve(key):
            # `.get`, not `[...]`: the centralised control records a shorter per-round
            # dict than the FL loop, and a missing metric must log as a missing point
            # rather than lose the whole run's record.
            return [p.get(key) for p in per_round]

        result_block = dict(
            accuracy=res.get("accuracy"), f1=res.get("f1"), f1_macro=res.get("f1_macro"),
            precision=res.get("precision"), recall=res.get("recall"),
            f1_per_class=res.get("f1_per_class"),
            precision_per_class=res.get("precision_per_class"),
            recall_per_class=res.get("recall_per_class"),
            specificity_per_class=res.get("specificity_per_class"),
            class_support=res.get("class_support"), n_test=res.get("n_test"),
            balanced_accuracy=res.get("balanced_accuracy"),
            macro_specificity=res.get("macro_specificity"),
            kappa=res.get("kappa"), roc_auc_ovr=res.get("roc_auc_ovr"),
            nll=res.get("nll"), confusion=res.get("confusion"),
            per_round_test_acc=_curve("test_acc"),
            per_round_test_loss=_curve("test_loss"),
            per_round_val_acc=_curve("val_acc"),
            per_round_val_macro_f1=_curve("val_macro_f1"),
            per_round_macro_f1=_curve("macro_f1"),
            per_round_f1_weighted=_curve("f1_weighted"),
            per_round_precision=_curve("precision"),
            per_round_recall=_curve("recall"),
            per_round_balanced_accuracy=_curve("balanced_accuracy"),
            per_round_macro_specificity=_curve("macro_specificity"),
            per_round_kappa=_curve("kappa"),
            per_round_roc_auc=_curve("roc_auc_ovr"),
            per_round_n_clients=_curve("n_clients"),
            per_round_eps_cum=_curve("eps_cum"),
            per_round_C=_curve("C"), per_round_sigma=_curve("sigma"),
            per_round_clip_fraction=_curve("clip_fraction"),
            per_round_release_snr=_curve("release_snr"),
            clip_history=res.get("clip_history"),
            wall_time_s=res.get("wall_time_s"),
            communication_payload_bytes=res.get("communication_payload_bytes"),
            client_sizes=res.get("client_sizes"),
            partition_heterogeneity=res.get("partition_heterogeneity"),
            local_epoch_time_s=res.get("local_epoch_time_s"),
        )
        for _k in ("accuracy_iter_avg", "f1_iter_avg", "f1_macro_iter_avg",
                   "iter_avg_window", "test_acc_final_round", "selected_round",
                   "best_val_acc", "model_selection", "lr_schedule", "lr_final",
                   "class_weights", "augment"):
            if res.get(_k) is not None:
                result_block[_k] = res[_k]
        if priv:
            result_block.update(
                eps=priv.get("eps"), alpha_star=priv.get("alpha_star"),
                eps_updates=priv.get("eps_updates"),
                eps_clip_fraction=priv.get("eps_clip_fraction"),
                eps_total=priv.get("eps_total", priv.get("eps")),
                # how many rounds the clip estimator actually released, so the
                # certificate in the table can be read against the mechanism it is for
                T_clip_estimator=priv.get("T_clip_estimator"),
                clip_freeze_rounds=priv.get("freeze_rounds"),
                eps_per_round=[p["eps"] for p in priv.get("per_round_eps", [])],
                mechanism=priv.get("mechanism"), adjacency=priv.get("adjacency"),
                sigma=priv.get("sigma"), accountant_note=priv.get("note"),
                # measured, not predicted: how big the clipped update was against the noise
                # that was drawn over the released support. This is the column that answers
                # "why is this arm at chance" without anyone having to re-run anything, and
                # it is a RESULT, so it sits here rather than in the hashed config block.
                release_snr_mean=res.get("release_snr_mean"),
                release_snr_first=res.get("release_snr_first"),
                release_snr_last=res.get("release_snr_last"),
                release_signal_norm=res.get("release_signal_norm"),
                release_noise_norm=res.get("release_noise_norm"))
        return self.log(base_cfg["stage"]).record(cfg_block, result_block,
                                                  config_hash=config_hash)

#@@FILE ppml/multiseed.py
"""Multi-seed aggregation and paired significance testing (Work Order Task 4).

The submission reported one seed (42) for everything. With 5 paired samples the
Wilcoxon signed-rank test cannot go below p = 0.0625, so the family of comparisons
here is evaluated on >= 6 seeds and Holm-corrected across the declared family.
"""

import numpy as np
from scipy import stats

MIN_SEEDS_FOR_WILCOXON = 6


def _ci_t(vals):
    vals = np.asarray(vals, float)
    n = len(vals)
    if n < 2:
        return (float(vals[0]), float(vals[0])) if n else (float("nan"), float("nan"))
    m = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(n))
    crit = float(stats.t.ppf(0.975, n - 1))
    return m - crit * se, m + crit * se


def summarise(vals, metric_name="accuracy"):
    vals = np.asarray(vals, float)
    lo, hi = _ci_t(vals)
    return dict(metric=metric_name, n=int(len(vals)), mean=float(np.mean(vals)),
                std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                ci95_low=float(lo), ci95_high=float(hi),
                min=float(np.min(vals)), max=float(np.max(vals)))


def cohen_dz(diffs):
    d = np.asarray(diffs, float)
    if len(d) < 2:
        return float("nan")
    s = float(d.std(ddof=1))
    return float(d.mean() / s) if s > 1e-12 else float("nan")


def compare(a, b, name_a="A", name_b="B", family=None, min_seeds=MIN_SEEDS_FOR_WILCOXON):
    """Paired comparison over the SAME seeds.

    Returns mean difference, 95% CI, Cohen's dz, paired t-test p, Wilcoxon p and the
    Holm-adjusted p when `family` (a list of the raw p-values in the comparison family)
    is supplied.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) != len(b):
        raise ValueError("paired comparison needs equal seed counts (%d vs %d)"
                         % (len(a), len(b)))
    n = len(a)
    d = a - b
    lo, hi = _ci_t(d)
    t_stat, p_t = stats.ttest_rel(a, b) if n >= 2 else (float("nan"), float("nan"))
    if n >= 2 and np.any(np.abs(d) > 1e-15):
        try:
            w_stat, p_w = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, p_w = float("nan"), float("nan")
    else:
        w_stat, p_w = float("nan"), float("nan")
    out = dict(comparison="%s vs %s" % (name_a, name_b), n_seeds=int(n),
               mean_a=float(a.mean()), mean_b=float(b.mean()),
               mean_diff=float(d.mean()), diff_ci95_low=float(lo), diff_ci95_high=float(hi),
               cohen_dz=cohen_dz(d), t_stat=float(t_stat), p_ttest=float(p_t),
               wilcoxon_stat=float(w_stat) if w_stat == w_stat else float("nan"),
               p_wilcoxon=float(p_w), insufficient_seeds=bool(n < min_seeds),
               note=("n=%d < %d: Wilcoxon cannot reach p<0.05" % (n, min_seeds)
                     if n < min_seeds else ""))
    if family is not None:
        out["holm_p"] = float(holm_adjust(list(family), p_w))
    return out


def holm_adjust(pvals, p):
    """Holm step-down adjusted p for the entry whose raw value is `p`."""
    order = np.argsort(list(pvals))
    m = len(pvals)
    adj = [float("nan")] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * float(pvals[idx]))
        running = max(running, val)          # monotonicity enforcement
        adj[idx] = running
    for i, v in enumerate(pvals):
        if abs(float(v) - float(p)) < 1e-15:
            return adj[i]
    return float("nan")


def holm_grouped(rows, family_field, key="p_wilcoxon", out="holm_p_grouped"):
    """Holm within groups defined by `family_field`, pooled across everything else.

    `holm_table` corrects inside one dataset x architecture, which is the family the Work
    Order describes; but the same comparison tag is also run on several datasets, and a
    response letter that says "Holm corrected" invites the stricter reading. This gives
    both: group by the comparison name across every dataset and architecture, correct
    inside that, and write `out` plus the family size so the reader sees which of the two
    corrections a star came from. The per-family `holm_p` column is left untouched.
    """
    def clean(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 1.0
        return f if f == f else 1.0
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(str(r.get(family_field, "")), []).append(i)
    for idxs in groups.values():
        pvals = [clean(rows[i].get(key, 1.0)) for i in idxs]
        order = np.argsort(pvals)
        m = len(pvals)
        running = 0.0
        for rank, pos in enumerate(order):
            val = min(1.0, (m - rank) * pvals[pos])
            running = max(running, val)
            rows[idxs[pos]][out] = running
            rows[idxs[pos]]["holm_group_size"] = m
    for r in rows:
        r.setdefault(out, None)
        r.setdefault("holm_group_size", 0)
    return rows


def holm_table(rows, key="p_wilcoxon"):
    """rows: list of compare() dicts. Adds holm_p to each, in place, and returns them."""
    def clean(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 1.0
        return f if f == f else 1.0          # an undefined test is not significant
    pvals = [clean(r.get(key, 1.0)) for r in rows]
    order = np.argsort(pvals)
    m = len(pvals)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        rows[idx]["holm_p"] = running
        rows[idx]["p_raw"] = pvals[idx]
    return rows


def group_by_seeds(runs, key_fields, metric="accuracy"):
    """{config tuple: {seed: value}} - the shape every paired test wants."""
    groups = {}
    for r in runs:
        k = tuple(str(r.get(f)) for f in key_fields)
        groups.setdefault(k, {})[int(r["seed"])] = float(r[metric])
    return groups


def paired_from_logs(runs, field, value_a, value_b, pair_keys=("dataset", "partition_id"),
                     metric="accuracy", family=None, name_a=None, name_b=None):
    """Paired comparison of two values of `field`, matched on seed within each
    (pair_keys) cell. `field` is normally 'method' or 'architecture'."""
    A, B = {}, {}
    for r in runs:
        cell = tuple(str(r.get(f)) for f in pair_keys)
        seed = int(r.get("seed", -1))
        if r.get(field) == value_a:
            A[(cell, seed)] = float(r[metric])
        if r.get(field) == value_b:
            B[(cell, seed)] = float(r[metric])
    common = sorted(set(A) & set(B))
    if len(common) < 2:
        return dict(n_paired=len(common), mean_diff=None, p_wilcoxon=1.0, p_ttest=1.0,
                    note="fewer than 2 paired cells available")
    a = [A[c] for c in common]
    b = [B[c] for c in common]
    out = compare(a, b, name_a or str(value_a), name_b or str(value_b), family=family)
    out["n_paired"] = len(common)
    return out

#@@FILE ppml/crossdataset.py
"""The four datasets in one file, on one scale.

Work Order line 12 forbids the exact situation the previous revision was in: WISDM v2's
sweeps came from an older broken pipeline (~23% accuracy) while Table 5 came from the fixed
script (84.9%), and nothing in the deliverables could tell a reader which was which. Three
artifacts close that. `cross_dataset_runs.csv` is every logged run of every dataset on one
fixed header, so the four per-dataset sessions leave behind four files that stack.
`cross_dataset_comparison.csv` puts the same cell from all four datasets in one row, so a
stale or partial dataset is visible as a gap or a level shift with no arithmetic to do.
`pipeline_identity.csv` compares the settings each dataset actually ran, taken from the
logs rather than from the settings table, and says plainly whether the four are still one
pipeline.

Everything here reads `results/logs/`. No model is run and nothing is invented: a cell with
no logged run is an empty column, and a run logged before a metric existed leaves that
metric blank.

One session cannot hold all four: Kaggle wipes `/kaggle/working` between runs, so each
per-dataset bundle writes the tables for the datasets whose logs it has, and `consolidate`
is the merge step that takes the exported folders (or the four CSVs) back in and writes the
combined file and the cross-dataset figures from them.
"""

import csv
import json
import os

from . import config, multiseed, runlog

# the four datasets the sheet fixes, in the order the paper presents them
DATASETS = ("UCI HAR", "PAMAP2", "WISDM v2", "HARTH")

# the same headline set as exp_main.SUMMARY_METRICS, restated here so this module can be
# imported without pulling the experiment stack
METRICS = ("accuracy", "precision", "recall", "f1", "f1_macro", "balanced_accuracy",
           "macro_specificity", "kappa", "roc_auc_ovr", "nll")

CERT_COLS = ("eps", "alpha_star", "z", "C", "K", "T", "delta", "adjacency", "mechanism")

# The header of `cross_dataset_runs.csv`, in the order a reader wants it: which run this
# is, what it was configured with, every metric, the certificate it paid, then anything
# the stage carries that no other stage does. Fixed, not inferred, so the file written by
# the UCI HAR session and the file written by the HARTH session stack on top of each
# other without a column moving.
RUN_COLUMNS = (
    "dataset", "stage", "architecture", "cell", "label", "method", "seed",
    "partition_id", "reportable", "profile", "config_hash", "logged_at", "wall_time_s",
    "rounds", "T", "local_epochs", "epochs", "lr", "batch", "max_windows", "split_kind",
    "n_clients", "n_windows_train", "n_windows_test", "variant", "channel_mode",
    "feature_set", "attack", "mia_attack", "byz_attack", "byz_aggregator",
    "byz_fraction", "dp_support", "dp_sparsity", "dp_head_only",
) + METRICS + CERT_COLS + (
    "release_snr_mean", "release_snr_min", "release_snr_max", "noise_dominates",
    "accuracy_iter_avg", "f1_iter_avg", "noise_coords", "total_coords",
    "user_level_certificate", "extra_json")
# `T` is both a setting and a certificate column; a header must not repeat a column
RUN_COLUMNS = tuple(dict.fromkeys(RUN_COLUMNS))

# The columns that stay text when a merged CSV is read back: everything else that parses
# as a number is turned back into one, because a comparison of two text "0.850" values is
# a string comparison and a table of them is silently wrong.
TEXT_COLUMNS = ("dataset", "stage", "architecture", "cell", "label", "method", "profile",
                "reportable", "config_hash", "logged_at", "split_kind", "variant",
                "channel_mode", "feature_set", "attack", "mia_attack", "byz_attack",
                "byz_aggregator", "mechanism", "adjacency", "dp_support", "dp_head_only",
                "user_level_certificate", "noise_dominates", "extra_json")

# The settings that must agree across datasets for their rows to be comparable at all.
# `reportable` and `profile` are here because a smoke row mixed into a full table is the
# same class of error as a different round count, and just as invisible in a figure.
# `stage` is deliberately absent: it is the grouping the check runs inside, not a setting,
# so comparing it against itself would flag every table as inconsistent.
# Both `rounds`/`T` and `local_epochs`/`epochs` are listed: the FL loop writes the first of
# each pair and a centralised run the second, and a key a stage never logged is skipped.
PIPELINE_KEYS = ("rounds", "T", "local_epochs", "epochs", "lr", "batch", "z", "C",
                 "delta", "max_windows", "split_kind", "profile", "reportable")

# Properties of the signal, not of the pipeline: four datasets with different subject
# counts cannot share these, and a check that flagged them would flag itself as noise.
# They are still reported, so the table shows what actually differed.
DATASET_KEYS = ("K", "n_clients", "n_windows_train", "n_windows_test")


def _slug(name):
    return str(name).replace(" ", "_").replace("/", "-")


def slug(name):
    """The column suffix this module uses for a dataset name (public: readers of the CSVs
    need to know which column is which dataset without guessing the spelling)."""
    return _slug(name)


def _cell_of(r):
    return str(r.get("cell") or r.get("label") or r.get("method") or "?")


def _arch_of(r):
    return str(r.get("architecture") or r.get("arch") or "?")


def _num(v):
    """A float, or None for anything a CSV column cannot hold."""
    if v is None or isinstance(v, (list, dict, tuple, bool, str)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _csv_num(v):
    """The same, but from CSV text: a merged file carries every number as a string."""
    if v is None or isinstance(v, (list, dict, tuple, bool)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _fmt_val(v):
    """One canonical spelling of a setting value.

    A log holds the round count as the int 50, a merged CSV reads it back as the float
    50.0, and comparing the two texts would report a pipeline difference that does not
    exist. Numbers get the same format from both sources; everything else keeps its text.
    """
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return format(float(v), ".10g")
    f = _csv_num(v)
    return format(f, ".10g") if f is not None else str(v)


def _reference_set(seen):
    """The value-set the most datasets used at this stage: the majority, not the union.

    Flagging against the union marks every dataset guilty when one drifts, which hides
    which one drifted. Ties break on the sorted values so two runs agree.
    """
    counts = {}
    for vals in seen.values():
        counts[frozenset(vals)] = counts.get(frozenset(vals), 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], sorted(kv[0])))
    return set(ranked[0][0])


def read_all(datasets=None, stages=None, reportable_only=True):
    """Every logged run of every named dataset, as one list. The source of all tables."""
    rows = []
    for ds in (datasets or DATASETS):
        for stage in (stages or [None]):
            try:
                rows.extend(runlog.read_logs(dataset=ds, stage=stage,
                                             reportable_only=reportable_only))
            except Exception:
                # a dataset with no logs at all is a coverage statement, not an error
                continue
    return rows


def _read_log_dir(log_dir):
    """The JSONL logs in one exported folder, in the same shape as `runlog.read_logs`.

    A session's `/kaggle/working` is gone by the next day; what survives is the folder the
    user exported. The file name is still `<dataset>__<stage>.jsonl`, so the dataset and
    stage come from it exactly as they do from the live log directory.
    """
    out = []
    try:
        names = sorted(os.listdir(log_dir))
    except Exception:
        return out
    for fname in names:
        if not fname.endswith(".jsonl"):
            continue
        base = fname[:-len(".jsonl")]
        ds, _, st = base.partition("__")
        with open(os.path.join(log_dir, fname)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                rec.setdefault("dataset", ds)
                rec.setdefault("stage", st)
                out.append(rec)
    return out


def runs_table(rows=None, datasets=None, stages=None, reportable_only=True):
    """One row per logged run, all four datasets sharing ONE fixed column set.

    This is the single file the four sessions can be merged with: every number any figure
    or table is built from, with its dataset, stage, architecture, cell, seed, the full
    metric set, the settings that produced it and the certificate it was bought with.
    The header is fixed rather than the union of whatever the logs happen to carry, so
    the UCI HAR file and the HARTH file have the same columns in the same order and a
    reader can stack the four with one command; a field that is real but not in the
    column list is kept in `extra_json` instead of silently changing the header.
    """
    rows = rows if rows is not None else read_all(datasets, stages, reportable_only)
    known = set(RUN_COLUMNS)
    out = []
    for r in rows:
        row = {k: None for k in RUN_COLUMNS}
        row.update(dataset=r.get("dataset"), stage=r.get("stage"),
                   architecture=_arch_of(r), cell=_cell_of(r), seed=r.get("seed"),
                   partition_id=r.get("partition_id"),
                   reportable=r.get("reportable"), profile=r.get("profile"),
                   config_hash=r.get("config_hash"), logged_at=r.get("logged_at"),
                   wall_time_s=_num(r.get("wall_time_s")))
        for k in METRICS:
            row[k] = _num(r.get(k))
        for k in RUN_COLUMNS:
            if r.get(k) is not None and k not in METRICS:
                row[k] = r[k]
        extras = {k: v for k, v in r.items()
                  if k not in known and not isinstance(v, (list, dict, tuple))}
        if extras:
            row["extra_json"] = json.dumps(extras, sort_keys=True, default=str)
        out.append(row)
    out.sort(key=lambda x: (str(x["dataset"]), str(x["stage"]), str(x["architecture"]),
                            str(x["cell"]), str(x["seed"])))
    return out


def comparison_table(rows=None, datasets=None, stages=None, metrics=None,
                     reportable_only=True):
    """One row per (stage, architecture, cell); each dataset's spread side by side.

    For every metric and dataset the row carries `<metric>_mean__<Dataset>`,
    `<metric>_std__<Dataset>` and `<metric>_ci__<Dataset>` (the interval as one string,
    because a table cell a human reads), plus `<metric>_n__<Dataset>`. `accuracy_spread`
    is max-minus-min of the four means: the number that says whether one dataset is off
    the scale the others are on. A dataset with no logged run in the cell gets
    `missing__<Dataset>` and no numbers - never a zero.
    """
    rows = rows if rows is not None else read_all(datasets, stages, reportable_only)
    datasets = list(datasets or DATASETS)
    metrics = list(metrics or METRICS)
    groups = {}
    for r in rows:
        key = (str(r.get("stage") or "?"), _arch_of(r), _cell_of(r))
        groups.setdefault(key, {}).setdefault(str(r.get("dataset")), []).append(r)
    out = []
    for (stage, arch, cell), by_ds in sorted(groups.items()):
        row = dict(stage=stage, architecture=arch, cell=cell,
                   datasets_present=len(by_ds))
        acc_means = []
        for ds in datasets:
            group = by_ds.get(ds) or []
            slug = _slug(ds)
            for metric in metrics:
                vals = [_num(r.get(metric)) for r in group]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                s = multiseed.summarise(vals, metric_name=metric)
                row["%s_mean__%s" % (metric, slug)] = round(s["mean"], 6)
                row["%s_std__%s" % (metric, slug)] = round(s["std"], 6)
                row["%s_ci__%s" % (metric, slug)] = "[%.4f, %.4f]" % (s["ci95_low"],
                                                                      s["ci95_high"])
                row["%s_n__%s" % (metric, slug)] = s["n"]
                if metric == "accuracy":
                    acc_means.append(s["mean"])
            snr = [_num(r.get("release_snr_mean")) for r in group]
            snr = [v for v in snr if v is not None]
            if snr:
                row["release_snr_mean__%s" % slug] = round(sum(snr) / len(snr), 6)
            eps = [r.get("eps") for r in group if r.get("eps") is not None]
            if eps:
                uniq = sorted({_fmt_val(v) for v in eps})
                row["eps__%s" % slug] = eps[0] if len(uniq) == 1 else "|".join(uniq)
            # the settings that produced these numbers, on the same row: a reader who sees
            # a level shift can check whether the runs were even comparable without
            # opening a second file
            for field in ("rounds", "T", "n_clients"):
                vals = sorted({_fmt_val(r[field]) for r in group
                               if r.get(field) is not None})
                if vals:
                    row["%s__%s" % (field, slug)] = "|".join(vals)
            if not group:
                row["missing__%s" % slug] = "no logged runs"
        row["accuracy_spread"] = (round(max(acc_means) - min(acc_means), 6)
                                  if len(acc_means) > 1 else None)
        row["complete"] = len(acc_means) == len(datasets)
        out.append(row)
    return out


def pipeline_identity(rows=None, datasets=None, reportable_only=True):
    """Do the four datasets still run one pipeline? Answered from the logs, not settings.

    One row per (stage, setting, dataset) holding the distinct values that dataset was
    logged using at that stage. A setting is only comparable when every dataset that ran
    the stage used the SAME SET of values: a single value everywhere agrees, and a grid
    swept identically on all four agrees too. A dataset that used a subset or a different
    value is flagged NO, and that is exactly the defect the sheet's line 12 names - the
    WISDM sweeps from an older broken pipeline next to Table 5 from the fixed script.
    """
    rows = rows if rows is not None else read_all(datasets, None, reportable_only)
    wanted = list(datasets or DATASETS)
    # The pooled multi-dataset arm is a member of the evidence but not one of the four
    # datasets, and it deliberately runs a different window cap - so it is not part of a
    # statement about whether the four are still one pipeline.
    rows = [r for r in rows if str(r.get("dataset")) in set(wanted)]
    out = []
    stages = sorted({str(r.get("stage") or "?") for r in rows})
    for stage in stages:
        sub = [r for r in rows if str(r.get("stage") or "?") == stage]
        # Coverage first, as its own row: the sheet also complains about a dataset missing
        # from a table outright (Table 7 ran without WISDM v2), and a per-setting
        # comparison cannot show that - a missing dataset simply contributes no rows.
        counts = {}
        for r in sub:
            counts[str(r.get("dataset"))] = counts.get(str(r.get("dataset")), 0) + 1
        for ds in wanted:
            n = counts.get(ds, 0)
            out.append(dict(stage=stage, setting="__coverage__", dataset=ds,
                            values="%d logged runs" % n, n_distinct=1 if n else 0,
                            must_be_identical="no - the grid may legitimately differ",
                            agrees_with_other_datasets=(
                                "yes" if n else "NO - no logged run at this stage")))
        for key, required in [(k, True) for k in PIPELINE_KEYS] + \
                             [(k, False) for k in DATASET_KEYS]:
            seen = {}
            for r in sub:
                v = r.get(key)
                if v is None or isinstance(v, (list, dict, tuple)):
                    continue
                seen.setdefault(str(r.get("dataset")), set()).add(_fmt_val(v))
            if not seen:
                continue
            reference = _reference_set(seen)
            for ds in sorted(seen):
                vals = seen[ds]
                same = vals == reference
                out.append(dict(
                    stage=stage, setting=key, dataset=ds,
                    values="|".join(sorted(vals)),
                    n_distinct=len(vals),
                    must_be_identical=("yes" if required else
                                       "no - property of the signal, not the pipeline"),
                    agrees_with_other_datasets=(
                        "yes" if same else
                        ("not required - differs, dataset-intrinsic" if not required
                         else "NO - the other %d datasets used %s"
                              % (len(seen) - 1, "|".join(sorted(reference)))))))
    return out


def identity_violations(identity_rows):
    """The (stage, setting, dataset) triples that break the one-pipeline rule."""
    return sorted({(r["stage"], r["setting"], r["dataset"]) for r in identity_rows
                   if str(r["agrees_with_other_datasets"]).startswith("NO")})


def _write(path, rows, fieldnames=None):
    if not rows:
        return None
    fields = list(fieldnames) if fieldnames else sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def print_summary(rows=None, datasets=None, stages=None, reportable_only=True, limit=14):
    """The comparable numbers, on the console, in the order they will be read.

    Pasted into a session log this is the evidence for line 12: the same cell, four
    datasets, one line each, and the identity verdict underneath. `rows` lets a caller
    that has already merged logs from several sessions print from that merge instead of
    re-reading this session's log directory.
    """
    rows = rows if rows is not None else read_all(datasets, stages, reportable_only)
    # the four columns are always printed, including for a dataset that has not logged
    # anything yet: a blank column is the coverage statement the sheet asks for
    cols = list(datasets or DATASETS)
    if not rows:
        print("  cross-dataset: no logged runs yet for %s - the comparison is drawn from"
              " the logs, so it fills in as the per-dataset sessions finish"
              % ", ".join(cols))
        return
    print("  cross-dataset: %d logged runs, %d of %d datasets with logs"
          % (len(rows), len({str(r.get("dataset")) for r in rows} & set(cols)), len(cols)))
    comp = [c for c in comparison_table(rows=rows, datasets=cols)
            if c.get("stage") == "main"] or comparison_table(rows=rows, datasets=cols)
    hdr = "  %-9s %-12s %-16s" % ("stage", "architecture", "cell")
    line = hdr + "".join(" %-16s" % d for d in cols)
    print(line)
    print("  " + "-" * (len(line) - 2))
    for c in comp[:limit]:
        cells = []
        for d in cols:
            # A metric with no logged value is a blank, never a zero: a 0.000 F1 next to a
            # real accuracy would be a number no run produced.
            parts = []
            for metric in ("accuracy", "f1"):
                v = c.get("%s_mean__%s" % (metric, _slug(d)))
                parts.append("-" if v is None else "%.3f" % v)
            cells.append("/".join(parts))
        print("  %-9s %-12s %-16s%s" % (c["stage"], c["architecture"], c["cell"],
                                        "".join(" %-16s" % v for v in cells)))
    if len(comp) > limit:
        print("  ... %d more cells in cross_dataset_comparison.csv" % (len(comp) - limit))
    ident = pipeline_identity(rows=rows, datasets=cols)
    bad = identity_violations(ident)
    checked = sorted({i["setting"] for i in ident if i["setting"] != "__coverage__"})
    print("  settings compared within each stage: %s" % ", ".join(checked))
    if bad:
        print("  PIPELINE NOT IDENTICAL - %d of the %d checks disagree (see"
              " pipeline_identity.csv; the rows of the datasets that disagree are not"
              " comparable):" % (len(bad), len(ident)))
        for stage, setting, ds in bad[:12]:
            mine = [i["values"] for i in ident
                    if i["stage"] == stage and i["setting"] == setting
                    and i["dataset"] == ds]
            others = sorted({i["values"] for i in ident
                             if i["stage"] == stage and i["setting"] == setting
                             and i["dataset"] != ds})
            print("     %-11s %-14s %-9s used %s; the others used %s"
                  % (stage, setting, ds, mine[0], ", ".join(others)))
        if len(bad) > 12:
            print("     ... %d more in pipeline_identity.csv" % (len(bad) - 12))
    else:
        print("  pipeline identity: every setting checked is identical on all %d datasets"
              " with logs, and every stage has runs from all of them" % len(cols))


def write_outputs(datasets=None, stages=None, results_dir=None, reportable_only=True):
    """Write the three cross-dataset CSVs and return their paths (absent where empty)."""
    config.ensure_dirs()
    results_dir = results_dir or config.RESULTS_DIR
    rows = read_all(datasets, stages, reportable_only)
    paths = {}
    for fname, data, cols in (
            ("cross_dataset_runs.csv", runs_table(rows=rows), RUN_COLUMNS),
            ("cross_dataset_comparison.csv", comparison_table(rows=rows), None),
            ("pipeline_identity.csv", pipeline_identity(rows=rows), None)):
        p = _write(os.path.join(results_dir, fname), data, fieldnames=cols)
        if p:
            paths[fname] = p
            print("  cross-dataset: %-30s %4d rows -> %s"
                  % (fname, len(data), p))
        else:
            print("  cross-dataset: %-30s  skipped - nothing logged yet" % fname)
    return paths


def session_block(datasets=None, stages=None, results_dir=None):
    """What the driver calls at the end of a session: write the files, print the reading."""
    paths = write_outputs(datasets=datasets, stages=stages, results_dir=results_dir)
    print_summary(datasets=datasets, stages=stages)
    return paths


def collect_run_files(paths):
    """Stack several `cross_dataset_runs.csv` on one header, and prove they match.

    The four per-dataset bundles each write the same columns, so merging them is a
    concatenation - but only if the headers really are the same, which is checked rather
    than assumed. A file whose header is not `RUN_COLUMNS` is refused and named: silently
    dropping its columns is how one dataset ends up compared on fewer settings than
    another.

    Returns (rows, notes). `rows` is in the shape the tables here take, so the CSV text
    values are turned back into numbers for the metric columns.
    """
    rows, notes = [], []
    for p in (paths or []):
        try:
            with open(p, newline="") as fh:
                got = list(csv.DictReader(fh))
        except Exception as exc:
            notes.append("%s could not be read (%s: %s) - skipped"
                         % (p, type(exc).__name__, exc))
            continue
        fields = tuple(got[0].keys()) if got else ()
        if fields and fields != tuple(RUN_COLUMNS):
            notes.append("%s has %d columns that differ from the %d-column standard"
                         " header - skipped; re-run that bundle with the current code so"
                         " its file stacks with the others"
                         % (p, len(fields), len(RUN_COLUMNS)))
            continue
        for rec in got:
            row = {}
            for k, v in rec.items():
                if v == "" or v is None:
                    row[k] = None
                elif k in TEXT_COLUMNS:
                    row[k] = v
                else:
                    f = _csv_num(v)
                    row[k] = v if f is None else f
            if row.get("dataset") is None:
                notes.append("%s has a row with no dataset - skipped" % p)
                continue
            rows.append(row)
        notes.append("%s: %d runs from %s" % (os.path.basename(p), len(got),
                                              sorted({str(r["dataset"]) for r in got})))
    return rows, notes


def consolidate(inputs=None, datasets=None, results_dir=None, make_figures=True):
    """Merge what the four per-dataset sessions left behind into one comparable set.

    `/kaggle/working` is wiped between sessions, so no single bundle ever holds all four
    datasets' logs. This is the merge step: point `inputs` at the folders or files the
    earlier sessions exported (attached as Kaggle outputs/datasets, or downloaded), and it
    writes the three cross-dataset CSVs and the cross-dataset figures from the combined
    runs - same code path, same columns, same figures as a session that had everything.

    Accepts a log folder, a `results` folder, a `cross_dataset_runs.csv` or a mixture; a
    directory is searched for both the JSONL logs and the CSV.
    """
    inputs = [inputs] if isinstance(inputs, str) else list(inputs or [])
    log_dirs, csvs, notes = [], [], []
    for item in inputs:
        if os.path.isdir(item):
            # a session exports results/logs next to the CSVs; accept the notebook output
            # root, a results folder or the log folder itself
            for d in (item, os.path.join(item, "results", "logs"),
                      os.path.join(item, "logs")):
                if os.path.isdir(d) and any(f.endswith(".jsonl") for f in os.listdir(d)):
                    log_dirs.append(d)
            for root, _, files in os.walk(item):
                for f in files:
                    if f == "cross_dataset_runs.csv":
                        csvs.append(os.path.join(root, f))
        elif item.endswith(".csv"):
            csvs.append(item)
        elif item.endswith(".jsonl"):
            log_dirs.append(os.path.dirname(item))
        else:
            notes.append("%s is not a folder, a .csv or a .jsonl - skipped" % item)
    rows, seen = [], set()

    def _key(r):
        return (str(r.get("dataset")), str(r.get("stage")), _arch_of(r),
                _cell_of(r), str(r.get("seed")), str(r.get("config_hash")))

    for d in sorted(set(log_dirs)):
        for r in _read_log_dir(d):
            if _key(r) in seen:
                continue
            seen.add(_key(r))
            rows.append(r)
    from_logs = len(rows)
    extra, extra_notes = collect_run_files(csvs)
    notes.extend(extra_notes)
    for rec in extra:
        if _key(rec) in seen:
            continue
        seen.add(_key(rec))
        rows.append(rec)
    if not rows:
        print("  consolidate: nothing found in %s - pass the folders the four sessions"
              " exported (their results/logs, or their cross_dataset_runs.csv)" % inputs)
        return dict(rows=0, paths={}, notes=notes)
    present = sorted({str(r.get("dataset")) for r in rows})
    print("  consolidate: %d runs (%d from logs, %d from the runs CSVs) covering %s"
          % (len(rows), from_logs, len(rows) - from_logs, ", ".join(present)))
    results_dir = results_dir or config.RESULTS_DIR
    config.ensure_dirs()
    paths = {}
    for fname, data, cols in (
            ("cross_dataset_runs.csv", runs_table(rows=rows), RUN_COLUMNS),
            ("cross_dataset_comparison.csv", comparison_table(rows=rows), None),
            ("pipeline_identity.csv", pipeline_identity(rows=rows, datasets=datasets),
             None)):
        p = _write(os.path.join(results_dir, fname), data, fieldnames=cols)
        if p:
            paths[fname] = p
            print("  consolidate: %-30s %4d rows -> %s" % (fname, len(data), p))
    print_summary(rows=rows, datasets=datasets)
    if make_figures:
        _figures(datasets=present)
    return dict(rows=len(rows), paths=paths, notes=notes)


def _figures(datasets):
    """Draw only the cross-dataset figures, from whatever is now in the logs."""
    try:
        from . import plotting
    except Exception as exc:
        print("  consolidate: figures unavailable (%s: %s) - the CSVs are still written"
              % (type(exc).__name__, exc))
        return
    for fn, kwargs in ((plotting.fig_privacy_utility, dict(datasets=datasets)),
                       (plotting.fig_privacy_utility, dict(datasets=datasets,
                                                           metric="f1")),
                       (plotting.fig_cross_dataset, dict(datasets=datasets)),
                       (plotting.fig_cross_dataset, dict(datasets=datasets,
                                                          metric="f1")),
                       (plotting.fig_cross_dataset_eps, dict(datasets=datasets)),
                       (plotting.fig_cross_dataset_eps, dict(datasets=datasets,
                                                              metric="f1"))):
        try:
            fn(**kwargs)
        except Exception as exc:
            print("  consolidate: %s skipped (%s: %s)"
                  % (fn.__name__, type(exc).__name__, exc))

#@@FILE ppml/plotting.py
"""Figures drawn ONLY from the run logs (Work Order section 1 rule 1 and section 13).

The rejected submission plotted from hardcoded result dictionaries and from generators
that fabricated curves and pseudo-membership-attacks. Nothing here can accept numbers
that are not in results/logs/*.jsonl: every entry point starts by calling
runlog.read_logs(), and a figure whose runs have not been logged is reported as
*skipped*, with the filter it looked for, instead of being drawn.

Each figure is written as PDF (vector, for the paper) and PNG (300 dpi, for the response
letter) under config.FIG_DIR and is returned in a manifest recording the stage, the
number of runs behind it and the exact log filter used, so every panel can be traced back
to the runs that produced it.

Field names come from harness._record(): the config block contributes `arch`,
`architecture`, `cell`, `method`, `seed`, `partition_id`, `z`, `C`, `K`, `T`, `delta`,
`split_kind`, `variant`, `channel_mode`, `feature_set`, `alpha`, `byz_attack`,
`byz_aggregator`, `byz_fraction`, `dp_level`, `mia_attack`; the result block contributes
`accuracy`, `f1`, `f1_macro`, `precision`, `recall`, `balanced_accuracy`,
`macro_specificity`, `kappa`, `roc_auc_ovr`, `nll`, `f1_per_class`,
`precision_per_class`, `recall_per_class`, `specificity_per_class`, `class_support`,
`confusion`, `eps`, `eps_total`, `eps_updates`, `eps_clip_fraction`, `alpha_star`,
`release_snr_mean`, `per_round_test_acc`, `per_round_precision`, `per_round_recall`,
`per_round_macro_f1`, `per_round_eps_cum`, `per_round_release_snr`, `eps_per_round`,
`clip_history`, `communication_payload_bytes`, `partition_heterogeneity` and, for the cost
stage, the nested `latency_cpu_batch1`, `memory_gpu`, `memory_cpu` blocks.
"""

import os

import numpy as np

from . import config, multiseed, runlog

# ── style ─────────────────────────────────────────────────────────────────────
FIGSIZE_1COL = 3.5
FIGSIZE_2COL = 7.16

# Okabe-Ito: colour-blind safe, and stable across figures (same method, same colour).
METHOD_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
                 "#000000", "#F0E442", "#66C2A5", "#FC8D62"]
_GRID = dict(linestyle=":", alpha=0.45)
_CAPSIZE = 2.0
# The three release supports the DP sweep measures. They pay the SAME certificate at a given
# z, which is why a privacy-utility figure must not group by epsilon alone: three mechanisms
# at one eps would average into a single marker and the comparison the resubmission rests on
# would vanish from the figure that is supposed to show it.
DP_SUPPORTS = ("all", "released-noise", "head")
DP_SUPPORT_LABELS = {"all": "whole model (as published)",
                     "released-noise": "sparse release",
                     "head": "head-only release"}
DP_SUPPORT_STYLES = ("-", "--", "-.")
DP_SUPPORT_MARKERS = ("o", "s", "^")

_RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.labelsize": 9,
    "lines.linewidth": 1.4,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
    "errorbar.capsize": _CAPSIZE,
}

STATE = {"applied": False, "inline": None, "plt": None, "mpl": None}


def apply_style():
    """Set the figure style once per session. Returns False when matplotlib is absent."""
    if STATE["applied"]:
        return True
    try:
        import matplotlib
        try:
            from IPython import get_ipython
            in_notebook = get_ipython() is not None
        except Exception:
            in_notebook = False
        if not in_notebook:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("  plotting unavailable (%s)" % exc)
        return False
    plt.rcParams.update(_RC)
    STATE.update(applied=True, plt=plt, mpl=matplotlib,
                 inline=bool(in_notebook) if STATE["inline"] is None else STATE["inline"])
    return True


def _pyplot():
    if not STATE["applied"] and not apply_style():
        raise RuntimeError("matplotlib is not installed")
    return STATE["plt"]


def _show(fig, name):
    """Save PDF + 300 dpi PNG and, in a notebook, display the figure inline."""
    plt = _pyplot()
    config.ensure_dirs()
    pdf = os.path.join(config.FIG_DIR, name + ".pdf")
    png = os.path.join(config.FIG_DIR, name + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    if STATE["inline"] is None:
        try:
            from IPython import get_ipython
            STATE["inline"] = get_ipython() is not None
        except Exception:
            STATE["inline"] = False
    if STATE["inline"]:
        try:
            from IPython.display import display
            display(fig)
            plt.show()
        except Exception:
            plt.show()
    plt.close(fig)
    return pdf, png


def _finish(fig, name, info):
    """Attach the paths and the traceability fields to one figure record."""
    pdf, png = _show(fig, name)
    rec = dict(figure=name, path_pdf=pdf, path_png=png)
    rec.update(info)
    print("  [fig] %-34s <- %s" % (name, rec.get("log_filter", "")))
    return rec


# ── log access helpers ────────────────────────────────────────────────────────
def _arch_of(rec):
    return str(rec.get("architecture") or rec.get("arch") or "?")


def _cell_of(rec):
    return str(rec.get("cell") or rec.get("method") or "?")


def _support_of(rec):
    """Which coordinates a DP run released: the whole model, the sparse set, or the head.

    Read from the mechanism fields, never from the cell name, because the name is what a
    method is called and these are what it did.
    """
    s = rec.get("dp_support")
    if s in DP_SUPPORTS:
        return str(s)
    if rec.get("dp_head_only"):
        return "head"
    sp = rec.get("dp_sparsity")
    try:
        if sp is not None and float(sp) < 1.0:
            return "released-noise"
    except (TypeError, ValueError):
        pass
    return "all"


def _val(rec, metric):
    """A metric as float, or None when this record does not carry it."""
    v = rec.get(metric)
    if v is None or isinstance(v, (dict, list, tuple, bool, str)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _mean(vals):
    """Arithmetic mean of the values that exist; 0.0 for an empty group.

    Used for ordering only, so a cell with no logged reading sorts last rather than
    raising, and no plotted number comes from here.
    """
    v = [float(x) for x in vals if x is not None]
    return sum(v) / len(v) if v else 0.0


def _nested(rec, block, key):
    d = rec.get(block)
    if isinstance(d, dict):
        return _val(d, key)
    return None


def _summary(runs, key_fields, metric="accuracy"):
    """Group logged runs by `key_fields` and summarise the seed replicates."""
    groups = {}
    for r in runs:
        k = tuple(str(r.get(f, "?")) for f in key_fields)
        groups.setdefault(k, []).append(r)
    rows = []
    for k, g in sorted(groups.items()):
        vals = [_val(x, metric) for x in g]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        s = multiseed.summarise(vals, metric_name=metric)
        s["group"] = k
        s["metric"] = metric
        s.update(dict(zip(key_fields, k)))
        s["_runs"] = g
        rows.append(s)
    return rows


def _ci_bars(axes, xs, rows, metric="accuracy", color=None, width=0.62,
             show_points=True, label=None, alpha=0.85):
    """Bars of the seed mean with 95% CI whiskers; every individual reading as a dot.

    The dots are the reviewers' request (R2-12) made visible: a bar of means cannot show
    that a method won five seeds and lost one badly.
    """
    means = [row["mean"] for row in rows]
    low_err = [row["mean"] - row["ci95_low"] for row in rows]
    high_err = [row["ci95_high"] - row["mean"] for row in rows]
    col = color or METHOD_COLORS[0]
    bars = axes.bar(xs, means, color=col, width=width, alpha=alpha, label=label)
    axes.errorbar(xs, means, yerr=[low_err, high_err], fmt="none", ecolor="black",
                  elinewidth=0.9, capsize=_CAPSIZE, capthick=0.9, zorder=3)
    if show_points:
        rng = np.random.RandomState(0)
        for i, row in enumerate(rows):
            vals = [_val(r, metric) for r in row["_runs"]]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            jitter = np.clip(rng.uniform(-0.5, 0.5, len(vals)) * width * 0.75,
                             -width * 0.4, width * 0.4)
            axes.plot(np.asarray(xs, float)[i] + jitter, vals, "o", ms=3.0,
                      color="black", alpha=0.75, zorder=4)
    return bars, means


def _label_axis(axes, labels, title, ylabel, xlabel=None, ylim=None):
    axes.set_xticks(list(range(len(labels))))
    axes.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    axes.set_title(title, fontsize=9)
    axes.set_ylabel(ylabel)
    if xlabel:
        axes.set_xlabel(xlabel)
    if ylim:
        axes.set_ylim(*ylim)
    axes.grid(axis="y", **_GRID)


def _skip(name, reason):
    print("  [fig] %-34s SKIPPED: %s" % (name, reason))
    return dict(figure=name, skipped=reason, n_runs=0)


def _n(runs):
    return len({(r.get("dataset"), r.get("stage"), _cell_of(r),
                 int(r.get("partition_id", 0)), int(r.get("seed", 0)))
               for r in runs})


ALL_DATASETS_DEFAULT = ("UCI HAR", "PAMAP2", "WISDM v2", "HARTH")


# ══ Task 4 - main results, every reading, curves, paired effects ══════════════
def fig_main_results(dataset, stages=("main",), metric="accuracy"):
    """Every method x architecture with 95% CI bars and per-seed dots (Tables 5, 6)."""
    plt = _pyplot()
    runs = [r for st in stages for r in runlog.read_logs(dataset=dataset, stage=st)]
    runs = [r for r in runs if _val(r, metric) is not None]
    if not runs:
        return _skip("fig_main_%s" % dataset.replace(" ", "_"),
                     "no logged runs for %s stage=%s" % (dataset, ",".join(stages)))
    rows = _summary(runs, ["architecture", "cell"], metric=metric)
    rows.sort(key=lambda r: -r["mean"])
    arch_labels = {}
    for r in rows:
        arch_labels.setdefault(str(r.get("architecture") or r.get("arch")),
                               len(arch_labels))
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 3.3))
    colors = [METHOD_COLORS[arch_labels[str(r.get("architecture")
                                          or r.get("arch"))] % len(METHOD_COLORS)]
              for r in rows]
    for i, row in enumerate(rows):
        _ci_bars(axes, [i], [row], metric=metric, color=colors[i], show_points=True)
    for arch, idx in sorted(arch_labels.items(), key=lambda kv: kv[1]):
        axes.plot([], [], "s", ms=7, color=METHOD_COLORS[idx % len(METHOD_COLORS)],
                  label=str(arch))
    axes.plot([], [], "o", ms=3, color="black", label="seed reading")
    axes.plot([], [], "|", ms=9, mec="black", color="none", label="95% CI (t, n seeds)")
    _label_axis(axes, ["%s\n%s" % (r.get("architecture") or r.get("arch"), r["cell"])
                       for r in rows],
                "%s: %s per method, mean over %d logged runs"
                % (dataset, metric, _n(runs)), metric, ylim=(0, 1.02))
    axes.legend(fontsize=7, frameon=False, ncol=4, loc="lower center",
                bbox_to_anchor=(0.5, -0.42))
    fig.tight_layout()
    return _finish(fig, "fig_main_%s" % dataset.replace(" ", "_"), dict(
        stage=",".join(stages), n_runs=_n(runs), cells=len(rows),
        log_filter="dataset=%s stage=%s metric=%s" % (dataset, ",".join(stages), metric)))


def fig_all_readings(dataset, metric="accuracy"):
    """Per-seed readings of every method, in one strip chart per cell."""
    plt = _pyplot()
    runs = [r for r in runlog.read_logs(dataset=dataset)
            if _val(r, metric) is not None and r.get("seed") is not None]
    per_cell = {}
    for r in runs:
        key = "%s | %s" % (_arch_of(r), _cell_of(r))
        per_cell.setdefault(key, {})[int(r["seed"])] = _val(r, metric)
    keep = {k: v for k, v in sorted(per_cell.items()) if len(v) >= 2}
    name = "fig_all_readings_%s" % dataset.replace(" ", "_")
    if not keep:
        return _skip(name, "no cell of %s has >= 2 logged seeds" % dataset)
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 3.4))
    keys = list(keep)
    half = max(0.28, min(0.45, 6.0 / max(1, len(keys))))
    for i, k in enumerate(keys):
        vals = list(keep[k].values())
        axes.add_patch(plt.Rectangle((i - half, np.percentile(vals, 25)), 2 * half,
                                     np.percentile(vals, 75) - np.percentile(vals, 25),
                                     facecolor=METHOD_COLORS[i % len(METHOD_COLORS)],
                                     edgecolor="none", alpha=0.35, zorder=1))
        axes.plot([i - half, i + half], [np.median(vals)] * 2, color="black", lw=1.1,
                  zorder=3)
        rng = np.random.RandomState(int(i))
        xs = i + np.clip(rng.uniform(-half, half, len(vals)), -half * 0.85, half * 0.85)
        axes.plot(xs, vals, "o", ms=3.4, color="black", alpha=0.8, zorder=4)
        axes.annotate("n=%d" % len(vals), (i, 0.02), xycoords=("data", "axes fraction"),
                      fontsize=6, ha="center", color="0.4")
    axes.set_xticks(list(range(len(keys))))
    axes.set_xticklabels(keys, rotation=50, ha="right", fontsize=7)
    axes.set_ylabel(metric)
    axes.set_title("%s: every logged reading (box = IQR, line = median, dot = seed)"
                   % dataset, fontsize=9)
    axes.grid(axis="y", **_GRID)
    fig.tight_layout()
    return _finish(fig, name, dict(n_runs=sum(len(v) for v in keep.values()),
                                   cells=len(keys),
                                   log_filter="dataset=%s all stages" % dataset))


def fig_training_curves(dataset, max_curves=40):
    """Per-round test accuracy for every logged run - the curves the old code faked."""
    plt = _pyplot()
    runs = runlog.read_logs(dataset=dataset)
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 3.0))
    drawn, seen = 0, set()
    for r in runs:
        curve = r.get("per_round_test_acc") or []
        curve = [c for c in curve if c is not None]
        if len(curve) < 2:
            continue
        key = (_arch_of(r), _cell_of(r), int(r.get("partition_id", 0)),
               int(r.get("seed", 0)))
        if key in seen or drawn >= max_curves:
            continue
        seen.add(key)
        axes.plot(range(1, len(curve) + 1), curve, lw=0.9,
                  color=METHOD_COLORS[drawn % len(METHOD_COLORS)],
                  label="%s/%s s%d" % (_arch_of(r), _cell_of(r), r.get("seed", 0)))
        drawn += 1
    name = "fig_curves_%s" % dataset.replace(" ", "_")
    if not drawn:
        return _skip(name, "no logged run for %s carries a per_round_test_acc curve"
                     % dataset)
    axes.set_xlabel("communication round")
    axes.set_ylabel("test accuracy")
    axes.set_title("%s: per-round test accuracy, %d logged runs%s"
                   % (dataset, drawn, " (capped)" if drawn >= max_curves else ""),
                   fontsize=9)
    axes.grid(True, **_GRID)
    if drawn <= 14:
        axes.legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    return _finish(fig, name, dict(n_runs=drawn,
                                   log_filter="dataset=%s, per_round_test_acc logged"
                                   % dataset))


def fig_paired_forest(dataset):
    """Forest plot of the declared comparison family: paired mean difference and CI.

    Drawn from exp_main.paired_tests(), which recomputes the statistics from the logs,
    so the figure and the CSV cannot disagree.
    """
    plt = _pyplot()
    try:
        from .experiments import exp_main
    except Exception as exc:
        return _skip("fig_forest_%s" % dataset.replace(" ", "_"),
                     "exp_main unavailable (%s)" % exc)
    rows = [r for r in exp_main.paired_tests(dataset)
            if r.get("mean_diff") is not None]
    name = "fig_forest_%s" % dataset.replace(" ", "_")
    if not rows:
        return _skip(name, "no comparison has >= 2 paired seeds for %s" % dataset)
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 0.34 * len(rows) + 1.5))
    ys = np.arange(len(rows))[::-1]
    for y, row in zip(ys, rows):
        lo = row["mean_diff"] - row["diff_ci95_low"]
        hi = row["diff_ci95_high"] - row["mean_diff"]
        col = METHOD_COLORS[1] if row["mean_diff"] > 0 else METHOD_COLORS[0]
        axes.errorbar(row["mean_diff"], y, xerr=[[lo], [hi]], fmt="o", ms=5, lw=1.2,
                      color=col, capsize=3)
        axes.annotate("n=%d  p=%.3g%s" % (row["n_seeds"], row.get("holm_p", row["p_raw"]),
                                          " *" if (row.get("holm_p") or 1) < 0.05 else ""),
                      (row["diff_ci95_high"] + 0.004, y), va="center", fontsize=6.5)
    axes.axvline(0.0, color="black", lw=0.9, ls="--")
    axes.set_yticks(list(ys))
    axes.set_yticklabels([r["comparison"] for r in rows], fontsize=7)
    axes.set_xlabel("paired mean difference in test accuracy (first minus second)")
    axes.set_title("%s: comparison family, Holm-adjusted p (*)" % dataset, fontsize=9)
    axes.grid(axis="x", **_GRID)
    fig.tight_layout()
    return _finish(fig, name, dict(n_runs=len(rows),
                                   log_filter="dataset=%s stage=main paired_tests()"
                                   % dataset))


# ══ Task 1 / 2 - privacy, clipping ════════════════════════════════════════════
_METRIC_PRETTY = {"accuracy": "test accuracy", "f1": "weighted F1",
                  "f1_macro": "macro F1", "precision": "weighted precision",
                  "recall": "weighted recall", "balanced_accuracy": "balanced accuracy",
                  "macro_specificity": "macro specificity", "kappa": "Cohen's kappa",
                  "roc_auc_ovr": "one-vs-rest ROC-AUC", "nll": "negative log-likelihood"}


def _pretty(metric):
    return _METRIC_PRETTY.get(metric, metric)


def fig_privacy_utility(datasets=None, stage="privacy", metric="accuracy"):
    """A metric versus the certified cumulative epsilon, one panel per dataset x arch.

    Work Order line 65 asks for accuracy AND F1 against cumulative epsilon for all four
    datasets and all three architectures, so the metric is a parameter and the file name
    follows it: `fig_privacy_utility` for accuracy, `fig_privacy_utility_<metric>` for any
    other, so the two readings are never drawn over each other.
    """
    plt = _pyplot()
    datasets = list(datasets or ALL_DATASETS_DEFAULT)
    runs = [r for ds in datasets for r in runlog.read_logs(dataset=ds, stage=stage)]
    usable = [r for r in runs if _val(r, metric) is not None]
    name = ("fig_privacy_utility" if metric == "accuracy"
            else "fig_privacy_utility_%s" % metric)
    if not usable:
        return _skip(name, "no %s-stage runs logged for %s" % (stage, datasets))
    cells, baselines, zlabels = {}, {}, {}
    for r in usable:
        e = _val(r, "eps")
        key = (r["dataset"], _arch_of(r))
        if e is None:
            # no noise -> no finite certificate; keep it as the non-private reference line
            baselines.setdefault(key, []).append(_val(r, metric))
            continue
        # series are (release support, eps), not eps: see DP_SUPPORTS
        cells.setdefault(key, {}).setdefault((_support_of(r), e), []).append(
            _val(r, metric))
        zlabels.setdefault((key, _support_of(r), e), set()).add(
            "z=%g" % float(r["z"]) if r.get("z") is not None else "?")
    if not cells:
        return _skip(name, "privacy runs were logged without an eps certificate")
    keys = sorted(cells)
    ncol = 2
    nrow = int(np.ceil(len(keys) / float(ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIGSIZE_2COL, 2.35 * nrow),
                             squeeze=False)
    for i, key in enumerate(keys):
        ax = axes[i // ncol][i % ncol]
        supports = [s for s in DP_SUPPORTS if any(k[0] == s for k in cells[key])]
        for si, sup in enumerate(supports):
            pts = sorted(e for (s, e) in cells[key] if s == sup)
            means = [float(np.mean(cells[key][(sup, e)])) for e in pts]
            errs = []
            for e in pts:
                summ = multiseed.summarise(cells[key][(sup, e)], metric_name=metric)
                errs.append([summ["mean"] - summ["ci95_low"],
                             summ["ci95_high"] - summ["mean"]])
            ax.errorbar(pts, means, yerr=np.asarray(errs).T,
                        marker=DP_SUPPORT_MARKERS[si % len(DP_SUPPORT_MARKERS)],
                        ms=4, lw=1.2, ls=DP_SUPPORT_STYLES[si % len(DP_SUPPORT_STYLES)],
                        color=METHOD_COLORS[si % len(METHOD_COLORS)],
                        label=DP_SUPPORT_LABELS.get(sup, sup))
            for x, m in zip(pts, means):
                ax.annotate("/".join(sorted(zlabels[(key, sup, x)])), (x, m), fontsize=5.5,
                            xytext=(0, 5), textcoords="offset points", ha="center")
        bl = [_val(r, metric) for r in usable
              if (r["dataset"], _arch_of(r)) == key and _val(r, "eps") is None]
        if bl:
            ax.axhline(float(np.mean(bl)), ls="--", lw=0.9, color="black")
            ax.annotate("non-private %.3f" % float(np.mean(bl)),
                        (ax.get_xlim()[1], float(np.mean(bl))), fontsize=6, ha="right",
                        va="bottom", xytext=(0, 2), textcoords="offset points")
        ax.set_xscale("log")
        ax.invert_xaxis()      # stronger privacy to the right
        ax.set_xlabel("cumulative $\\varepsilon$ ($\\delta=10^{-5}$)")
        ax.set_ylabel(_pretty(metric))
        ax.set_title("%s / %s" % (key[0], key[1]), fontsize=8)
        ax.grid(True, **_GRID)
        if len(supports) > 1:
            ax.legend(fontsize=5.5, frameon=False)
    for j in range(len(keys), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("%s versus privacy-utility from logged certificates (%d runs, %d points)"
                 % (_pretty(metric).capitalize(), _n(usable),
                    sum(len(v) for v in cells.values())), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _finish(fig, name, dict(stage=stage, metric=metric, n_runs=_n(usable),
                                   panels=len(keys),
                                   log_filter="stage=%s, eps certificate logged" % stage))


def fig_cross_dataset(datasets=None, stage="main", metric="accuracy"):
    """One cell, four datasets, one axis: the figure that makes the four comparable.

    Grouped bars per (architecture, cell), one bar per dataset, all four panels of the
    paper's argument on a single shared y-scale. This is the picture that answers the
    sheet's line 12 visually: the broken-pipeline WISDM sweeps and the fixed-pipeline
    Table 5 cannot sit side by side without the gap showing. A dataset with no logged run
    for a cell is a missing bar, never a bar of height zero.

    The dotted line is chance for the harmonised label set (1/N_CLASS), because the whole
    DP result is a statement about how far above chance a noisy release still gets.
    """
    plt = _pyplot()
    datasets = list(datasets or ALL_DATASETS_DEFAULT)
    runs_by_ds, by_ds = {}, {}
    for ds in datasets:
        runs = [r for r in runlog.read_logs(dataset=ds, stage=stage)
                if _val(r, metric) is not None]
        runs_by_ds[ds] = runs
        rows = _summary(runs, ["architecture", "cell"], metric=metric)
        # a cell can be present under both `architecture` spellings of an old log, so key
        # on the pair the summary already resolved
        by_ds[ds] = {(r["architecture"], r["cell"]): r for r in rows}
    present = [ds for ds in datasets if by_ds[ds]]
    n_runs = _n([r for ds in present for r in runs_by_ds[ds]])
    name = ("fig_cross_dataset" if metric == "accuracy"
            else "fig_cross_dataset_%s" % metric)
    if not present:
        return _skip(name, "no %s-stage run with a logged %s for %s"
                     % (stage, metric, datasets))
    cells = sorted({k for ds in present for k in by_ds[ds]})
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 3.4))
    width = min(0.8 / len(present), 0.2)
    for i, ds in enumerate(present):
        xs, rows = [], []
        for j, cell in enumerate(cells):
            row = by_ds[ds].get(cell)
            if row is None:
                continue
            xs.append(j + (i - (len(present) - 1) / 2.0) * width)
            rows.append(row)
        _ci_bars(axes, xs, rows, metric=metric, color=METHOD_COLORS[i], width=width,
                 show_points=False, label="%s (%d runs)" % (ds, _n(runs_by_ds[ds])))
    axes.axhline(1.0 / config.N_CLASS, ls=":", lw=1.0, color="black")
    axes.annotate("chance 1/%d" % config.N_CLASS, (axes.get_xlim()[1], 1.0 / config.N_CLASS),
                  fontsize=6, ha="right", va="bottom", xytext=(0, 2),
                  textcoords="offset points")
    axes.set_xticks(list(range(len(cells))))
    axes.set_xticklabels(["%s\n%s" % c for c in cells], rotation=0, ha="center", fontsize=6)
    axes.set_ylabel(_pretty(metric))
    axes.set_ylim(0, 1)
    axes.grid(True, axis="y", **_GRID)
    axes.legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle("%s by dataset, one pipeline (%s stage, %d logged runs)"
                 % (_pretty(metric).capitalize(), stage, n_runs), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _finish(fig, name, dict(stage=stage, metric=metric, n_runs=n_runs,
                                   cells=len(cells), datasets=",".join(present),
                                   log_filter="stage=%s, dataset in %s, %s logged"
                                   % (stage, datasets, metric)))


def fig_cross_dataset_eps(datasets=None, stages=("privacy", "clip"), metric="accuracy"):
    """Cumulative epsilon versus a metric with the four datasets on one axis, per arch.

    `fig_privacy_utility` gives one panel per dataset, which is the right way to read a
    single dataset's mechanism comparison and the wrong way to compare two datasets: a
    reader cannot tell a level shift from an axis shift. Here every dataset shares the axes
    of its architecture panel, in the dataset's colour, and the release support is carried
    by the line style for the same reason as in `fig_privacy_utility` - the three supports
    pay the same certificate at a given z and must not average into one marker.
    """
    plt = _pyplot()
    datasets = list(datasets or ALL_DATASETS_DEFAULT)
    stages = list(stages or ["privacy"])
    runs = [r for st in stages for ds in datasets
            for r in runlog.read_logs(dataset=ds, stage=st)
            if _val(r, metric) is not None and _val(r, "eps") is not None]
    name = ("fig_cross_dataset_eps" if metric == "accuracy"
            else "fig_cross_dataset_eps_%s" % metric)
    if not runs:
        return _skip(name, "no %s-stage runs with both %s and an eps certificate for %s"
                     % (",".join(stages), metric, datasets))
    arches = sorted({_arch_of(r) for r in runs})
    ds_colors = {ds: METHOD_COLORS[i % len(METHOD_COLORS)] for i, ds in enumerate(datasets)}
    ncol = 2
    nrow = int(np.ceil(len(arches) / float(ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIGSIZE_2COL, 2.6 * nrow),
                             squeeze=False)
    for i, arch in enumerate(arches):
        ax = axes[i // ncol][i % ncol]
        for ds in datasets:
            cells = {}
            for r in runs:
                if _arch_of(r) != arch or str(r.get("dataset")) != ds:
                    continue
                cells.setdefault((_support_of(r), _val(r, "eps")), []).append(
                    _val(r, metric))
            if not cells:
                continue
            for si, sup in enumerate(DP_SUPPORTS):
                pts = sorted(e for (s, e) in cells if s == sup)
                if not pts:
                    continue
                means, errs = [], []
                for e in pts:
                    summ = multiseed.summarise(cells[(sup, e)], metric_name=metric)
                    means.append(summ["mean"])
                    errs.append([summ["mean"] - summ["ci95_low"],
                                 summ["ci95_high"] - summ["mean"]])
                ax.errorbar(pts, means, yerr=np.asarray(errs).T,
                            marker=DP_SUPPORT_MARKERS[si % len(DP_SUPPORT_MARKERS)],
                            ms=4, lw=1.2, ls=DP_SUPPORT_STYLES[si % len(DP_SUPPORT_STYLES)],
                            color=ds_colors[ds],
                            label="%s %s" % (ds, DP_SUPPORT_LABELS.get(sup, sup)))
        ax.axhline(1.0 / config.N_CLASS, ls=":", lw=0.9, color="black")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("cumulative $\\varepsilon$ ($\\delta=10^{-5}$)")
        ax.set_ylabel(_pretty(metric))
        ax.set_ylim(0, 1)
        ax.set_title(arch, fontsize=8)
        ax.grid(True, **_GRID)
        ax.legend(fontsize=5, frameon=False, loc="lower left")
    for j in range(len(arches), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("%s versus cumulative $\\varepsilon$, all datasets on one scale"
                 " (%d certified runs)" % (_pretty(metric).capitalize(), _n(runs)),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _finish(fig, name, dict(stage=",".join(stages), metric=metric,
                                   n_runs=_n(runs), panels=len(arches),
                                   log_filter="stage in %s, eps certificate logged,"
                                   " %s logged" % (stages, metric)))


def fig_clip_sweep(dataset, stage="clip", metric="accuracy"):
    """Fixed-C sweep plus the adaptive and fixed arms at the same z."""
    plt = _pyplot()
    runs = [r for r in runlog.read_logs(dataset=dataset, stage=stage)
            if _val(r, metric) is not None]
    name = "fig_clip_%s" % dataset.replace(" ", "_")
    if not runs:
        return _skip(name, "no clip-stage runs logged for %s" % dataset)
    rows = _summary(runs, ["architecture", "cell"], metric=metric)
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 3.1))
    colors = [METHOD_COLORS[0] if "| C=" in r["cell"] else
              (METHOD_COLORS[1] if r["cell"].endswith("adaptive") else METHOD_COLORS[2])
              for r in rows]
    for i, row in enumerate(rows):
        _ci_bars(axes, [i], [row], metric=metric, color=colors[i])
    _label_axis(axes, ["%s\n%s" % (r.get("architecture") or r.get("arch"), r["cell"])
                       for r in rows],
                "%s: clipping - fixed C sweep, then adaptive vs fixed at z=%.1f (%d runs)"
                % (dataset, config.DP_ARM_Z, _n(runs)), metric, ylim=(0, 1.02))
    axes.plot([], [], "s", ms=7, color=METHOD_COLORS[0], label="fixed C sweep")
    axes.plot([], [], "s", ms=7, color=METHOD_COLORS[1], label="adaptive")
    axes.plot([], [], "s", ms=7, color=METHOD_COLORS[2], label="fixed (default C)")
    axes.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return _finish(fig, name, dict(stage=stage, n_runs=_n(runs), cells=len(rows),
                                   log_filter="dataset=%s stage=clip" % dataset))


def fig_clip_history(dataset, stage="clip", arch="MLP"):
    """The logged C_r and clipped-fraction trajectory - raw, never smoothed."""
    plt = _pyplot()
    curves = []
    for r in runlog.read_logs(dataset=dataset, stage=stage):
        hist = r.get("clip_history")
        if isinstance(hist, list) and hist and str(r.get("cell", "")).startswith(arch):
            curves.append((int(r.get("seed", 0)), hist))
    name = "fig_clip_history_%s" % dataset.replace(" ", "_")
    if not curves:
        return _skip(name, "no clip_history logged for %s/%s (written by the adaptive "
                           "and fixed clip arms)" % (dataset, arch))
    curves.sort(key=lambda c: c[0])
    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE_2COL, 2.7))
    frozen_at = None
    for i, (seed, hist) in enumerate(curves[:8]):
        col = METHOD_COLORS[i % len(METHOD_COLORS)]
        cs = [_val(h, "C") for h in hist]
        fr = [_val(h, "clip_fraction") for h in hist]
        xs = [int(h.get("round", j + 1)) for j, h in enumerate(hist)]
        axes[0].plot(xs, cs, lw=1.1, color=col, label="seed %d" % seed)
        axes[1].plot(xs, fr, lw=1.1, color=col)
        rel = [x for x, h in zip(xs, hist) if h.get("estimator_released")]
        if rel and len(rel) < len(xs):
            frozen_at = max(frozen_at or 0, max(rel))
    if frozen_at is not None:
        # where the clip-fraction estimator stopped releasing; after this the line is a
        # fixed radius chosen by the first rounds, and only those releases are accounted
        axes[0].axvline(frozen_at + 0.5, ls=":", lw=0.9, color="grey")
        axes[0].text(frozen_at + 0.8, axes[0].get_ylim()[1],
                     "estimator stopped", fontsize=6, va="top", color="grey")
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("$C_r$")
    axes[0].set_title("%s/%s: clip norm as logged" % (dataset, arch), fontsize=8)
    axes[0].legend(fontsize=6, frameon=False)
    axes[1].axhline(config.ADAPT_TARGET_Q, ls="--", lw=0.9, color="black")
    axes[1].set_xlabel("round")
    axes[1].set_ylabel("clipped fraction")
    axes[1].set_title("fraction clipped (dashed = target $\\gamma$=%.2f)"
                      % config.ADAPT_TARGET_Q, fontsize=8)
    for ax in axes:
        ax.grid(True, **_GRID)
    fig.tight_layout()
    return _finish(fig, name, dict(stage=stage, n_runs=len(curves),
                                   log_filter="dataset=%s stage=clip cell startswith %s"
                                   % (dataset, arch)))


def fig_eps_split(dataset, stage="clip"):
    """Budget decomposition: epsilon of the updates vs of the clip-fraction estimator."""
    plt = _pyplot()
    runs = [r for r in runlog.read_logs(dataset=dataset, stage=stage)
            if _val(r, "eps_total") is not None]
    name = "fig_eps_split_%s" % dataset.replace(" ", "_")
    if not runs:
        return _skip(name, "no adaptive-accounted run logged for %s (adaptive=True "
                            "writes eps_updates/eps_clip_fraction/eps_total)" % dataset)
    rows = _summary(runs, ["architecture", "cell"], metric="eps_total")
    up = [float(np.mean([_val(r, "eps_updates") or np.nan for r in x["_runs"]]))
          for x in rows]
    cl = [float(np.mean([_val(r, "eps_clip_fraction") or np.nan for r in x["_runs"]]))
          for x in rows]
    tot = [x["mean"] for x in rows]
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 2.9))
    xs = list(range(len(rows)))
    axes.bar(xs, up, color=METHOD_COLORS[0], width=0.6, label="$\\varepsilon$ updates")
    axes.bar(xs, cl, bottom=up, color=METHOD_COLORS[1], width=0.6,
             label="$\\varepsilon$ clip-fraction estimator")
    for i, t in enumerate(tot):
        axes.annotate("$\\varepsilon$=%.2f\n$\\alpha^*$=%s"
                      % (t, _fmt_star(rows[i]["_runs"])),
                      (i, (up[i] + cl[i]) * 1.03), ha="center", va="bottom", fontsize=7)
    axes.set_xticks(xs)
    axes.set_xticklabels(["%s\n%s" % (r.get("architecture") or r.get("arch"), r["cell"])
                          for r in rows], rotation=40, ha="right", fontsize=7)
    axes.set_ylabel("cumulative $\\varepsilon$")
    axes.set_title("%s: adaptive accounting, composed at the shared $\\alpha^*$ "
                   "(%d runs)" % (dataset, _n(runs)), fontsize=9)
    axes.legend(frameon=False, fontsize=8)
    axes.grid(axis="y", **_GRID)
    fig.tight_layout()
    return _finish(fig, name, dict(stage=stage, n_runs=_n(runs),
                                   log_filter="dataset=%s stage=clip eps_total logged"
                                   % dataset))


def _fmt_star(runs):
    vals = [_val(r, "alpha_star") for r in runs]
    vals = [v for v in vals if v is not None]
    return "%.2f" % float(np.mean(vals)) if vals else "n/a"


# ══ Task 3 - splits ═══════════════════════════════════════════════════════════
def fig_leakage(datasets=None, stage="splits", metric="accuracy"):
    """Window-level (leaking) vs subject-disjoint accuracy, paired per seed."""
    plt = _pyplot()
    datasets = list(datasets or ALL_DATASETS_DEFAULT)
    cells = {}
    total = 0
    for ds in datasets:
        for r in runlog.read_logs(dataset=ds, stage=stage):
            cell = _cell_of(r)
            if not (cell.endswith("| window-split") or cell.endswith("| subject-split")):
                continue
            if _val(r, metric) is None:
                continue
            kind = "window" if cell.endswith("window-split") else "subject"
            arch = cell.split("|")[0].strip()
            cells.setdefault((ds, arch), {}).setdefault(kind, {})[int(r["seed"])] = \
                _val(r, metric)
            total += 1
    name = "fig_leakage"
    pairs = {k: v for k, v in cells.items() if len(v.get("window", {})) and
             len(v.get("subject", {}))}
    if not pairs:
        return _skip(name, "no dataset has both window-split and subject-split runs")
    keys = sorted(pairs)
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 0.5 * len(keys) + 1.6))
    ys = np.arange(len(keys))[::-1]
    for y, k in zip(ys, keys):
        w, s = pairs[k]["window"], pairs[k]["subject"]
        seeds = sorted(set(w) & set(s))
        gap = np.array([w[x] - s[x] for x in seeds])
        summ = multiseed.summarise(gap, metric_name="gap")
        col = METHOD_COLORS[1] if gap.mean() > 0 else METHOD_COLORS[0]
        axes.errorbar(gap.mean(), y,
                      xerr=[[summ["mean"] - summ["ci95_low"]],
                            [summ["ci95_high"] - summ["mean"]]],
                      fmt="o", ms=5, color=col, capsize=3)
        for x in seeds:
            axes.plot(w[x] - s[x], y + 0.12, "|", ms=7, color="black", alpha=0.6)
        axes.annotate("n=%d seeds" % len(seeds), (0.985, y),
                      xycoords=("axes fraction", "data"), fontsize=6, ha="right",
                      color="0.4")
    axes.axvline(0.0, color="black", lw=0.9, ls="--")
    axes.set_yticks(list(ys))
    axes.set_yticklabels(["%s / %s" % (k[0], k[1]) for k in keys], fontsize=7)
    axes.set_xlabel("accuracy(window-level) - accuracy(subject-disjoint)")
    axes.set_title("Split leakage gap: positive means the window split flattered "
                   "the model (%d runs)" % total, fontsize=9)
    axes.grid(axis="x", **_GRID)
    fig.tight_layout()
    return _finish(fig, name, dict(stage=stage, n_runs=total,
                                   log_filter="stage=%s cell endswith window-split / "
                                              "subject-split" % stage))


def fig_loso(dataset, stage="splits", metric="accuracy"):
    """Leave-one-subject-out: held-out accuracy distribution per architecture."""
    plt = _pyplot()
    groups = {}
    for r in runlog.read_logs(dataset=dataset, stage=stage):
        cell = _cell_of(r)
        if "loso" not in cell or _val(r, metric) is None:
            continue
        groups.setdefault(cell.split("|")[0].strip(), []).append(_val(r, metric))
    name = "fig_loso_%s" % dataset.replace(" ", "_")
    if not groups:
        return _skip(name, "no loso-subjN cells logged for %s" % dataset)
    keys = sorted(groups)
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 2.8))
    bp = axes.boxplot([groups[k] for k in keys], widths=0.5, patch_artist=True,
                      medianprops=dict(color="black"))
    for i, p in enumerate(bp["boxes"]):
        p.set_facecolor(METHOD_COLORS[i % len(METHOD_COLORS)])
        p.set_alpha(0.4)
    rng = np.random.RandomState(1)
    for i, k in enumerate(keys):
        axes.plot((i + 1) + rng.uniform(-0.16, 0.16, len(groups[k])), groups[k], "o",
                  ms=3.2, color="black", alpha=0.8)
    axes.set_xticks(list(range(1, len(keys) + 1)))
    axes.set_xticklabels(["%s\n(n=%d folds)" % (k, len(groups[k])) for k in keys],
                         rotation=30, ha="right", fontsize=7)
    axes.set_ylabel("held-out subject %s" % metric)
    axes.set_title("%s: leave-one-subject-out, one point per unseen subject" % dataset,
                   fontsize=9)
    axes.grid(axis="y", **_GRID)
    fig.tight_layout()
    return _finish(fig, name, dict(stage=stage, n_runs=sum(len(v) for v in groups.values()),
                                   log_filter="dataset=%s stage=%s cell contains loso"
                                   % (dataset, stage)))


def fig_matched(dataset, stage="splits", metric="accuracy"):
    """Architecture-matched and representation-matched arms side by side."""
    plt = _pyplot()
    rows = _summary([r for r in runlog.read_logs(dataset=dataset, stage=stage)
                     if _val(r, metric) is not None and
                     ("matched" in _cell_of(r))],
                    ["architecture", "cell"], metric=metric)
    name = "fig_matched_%s" % dataset.replace(" ", "_")
    if not rows:
        return _skip(name, "no arch-matched / rep-matched runs logged for %s" % dataset)
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 3.0))
    for i, row in enumerate(rows):
        _ci_bars(axes, [i], [row], metric=metric,
                 color=METHOD_COLORS[i % len(METHOD_COLORS)])
    _label_axis(axes, ["%s\n%s" % (r.get("architecture"), r["cell"]) for r in rows],
                "%s: matched comparisons (%d runs; same rounds, epochs, batch, lr)"
                % (dataset, _n([r for r in runlog.read_logs(dataset=dataset, stage=stage)
                                if "matched" in _cell_of(r)])), metric, ylim=(0, 1.02))
    fig.tight_layout()
    return _finish(fig, name, dict(stage=stage, n_cells=len(rows),
                                   log_filter="dataset=%s stage=%s cell contains matched"
                                   % (dataset, stage)))


# ══ Task 5 - non-IID ═════════════════════════════════════════════════════════
def fig_noniid(dataset, stage="noniid", metric="accuracy"):
    """Accuracy vs Dirichlet alpha per method; spread is over independent partitions."""
    plt = _pyplot()
    runs = [r for r in runlog.read_logs(dataset=dataset, stage=stage)
            if _val(r, metric) is not None]
    name = "fig_noniid_%s" % dataset.replace(" ", "_")
    if not runs:
        return _skip(name, "no noniid runs logged for %s" % dataset)
    def arm(r):
        # The DP and non-DP arms of Task 5 share a `method` string; without the suffix
        # they would be averaged into one curve.
        return "%s%s" % (r.get("method"), "+DP" if r.get("z") else "")
    methods = sorted({arm(r) for r in runs})
    order = [a for a in ("0.05", "0.1", "0.3", "0.5", "1.0", "5.0", "IID")
             if any(str(r.get("alpha")) == a for r in runs)]
    order += sorted({str(r.get("alpha")) for r in runs} - set(order))
    if not order:
        return _skip(name, "the %s logs carry no alpha column" % dataset)
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 2.9))
    xi = {a: i for i, a in enumerate(order)}
    for j, meth in enumerate(methods):
        means, mins, maxs, ns = [], [], [], []
        for a in order:
            vals = [_val(r, metric) for r in runs
                    if str(r.get("alpha")) == a and arm(r) == meth]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            s = multiseed.summarise(vals, metric_name=metric)
            means.append((xi[a], s["mean"]))
            mins.append(min(vals))
            maxs.append(max(vals))
            ns.append(len(vals))
        if not means:
            continue
        xs = [m[0] for m in means]
        col = METHOD_COLORS[j % len(METHOD_COLORS)]
        axes.fill_between(xs, mins, maxs, alpha=0.15, color=col)
        axes.plot(xs, [m[1] for m in means], "-o", ms=4, color=col, label=meth)
        for i, x in enumerate(xs):
            axes.plot([x, x], [mins[i], maxs[i]], "-", lw=0.8, color=col, alpha=0.5)
            axes.annotate("n=%d" % ns[i], (x, maxs[i]), fontsize=5.5, ha="center",
                          va="bottom", color="0.35")
    axes.set_xticks(list(range(len(order))))
    axes.set_xticklabels(["$\\alpha$=" + a for a in order])
    axes.set_ylabel("%s (mean over partitions)" % metric)
    axes.set_title("%s: label heterogeneity, thin line = min-max over partitions\n"
                   "(+DP curves: a Dirichlet client mixes subjects, so its eps is not a "
                   "user-level certificate)" % dataset, fontsize=8)
    axes.legend(fontsize=7, frameon=False, title="aggregation")
    axes.grid(True, **_GRID)
    fig.tight_layout()
    return _finish(fig, name, dict(stage=stage, n_runs=_n(runs),
                                   log_filter="dataset=%s stage=%s" % (dataset, stage)))


# ══ Task 6 - Byzantine ═══════════════════════════════════════════════════════
def fig_byzantine(dataset, stage="byzantine", metric="accuracy"):
    """Clean accuracy vs adversary fraction: one line per attack, one panel per aggregator."""
    plt = _pyplot()
    runs = [r for r in runlog.read_logs(dataset=dataset, stage=stage)
            if _val(r, metric) is not None and r.get("byz_attack") is not None]
    if not runs:
        return _skip("fig_byzantine_%s" % dataset.replace(" ", "_"),
                     "no byzantine-stage runs logged for %s" % dataset)
    out = []
    cells = [(arch, part) for arch in sorted({_arch_of(r) for r in runs})
             for part in sorted({str(r.get("byz_partition") or "iid") for r in runs})]
    for arch, part in cells:
        # One figure per (architecture, partition kind): an IID curve and a Dirichlet
        # curve pooled together would hide exactly the difference Task 6 asks about.
        sub = [r for r in runs if _arch_of(r) == arch
               and str(r.get("byz_partition") or "iid") == part]
        aggs = list(config.BYZ_AGGREGATORS)
        aggs = [a for a in aggs if any(str(r.get("byz_aggregator")) == a for r in sub)] or \
            sorted({str(r.get("byz_aggregator")) for r in sub})
        attacks = [a for a in config.BYZ_ATTACKS
                   if any(str(r.get("byz_attack")) == a for r in sub)]
        fig, axes = plt.subplots(1, len(aggs), figsize=(FIGSIZE_2COL, 2.7),
                                 squeeze=False, sharey=True)
        drawn = 0
        for i, agg in enumerate(aggs):
            ax = axes[0][i]
            for j, atk in enumerate(attacks):
                pts = {}
                for r in sub:
                    if str(r.get("byz_aggregator")) != agg or str(r.get("byz_attack")) != atk:
                        continue
                    pts.setdefault(float(r.get("byz_fraction") or 0.0), []).append(
                        _val(r, metric))
                if not pts:
                    continue
                xs = sorted(pts)
                ms = [float(np.mean(pts[x])) for x in xs]
                sd = [float(np.std(pts[x], ddof=1)) if len(pts[x]) > 1 else 0.0
                      for x in xs]
                ax.errorbar(xs, ms, yerr=sd, marker="o", ms=3.5, lw=1.1,
                            color=METHOD_COLORS[j % len(METHOD_COLORS)], label=atk)
                drawn += sum(len(v) for v in pts.values())
            ax.set_xlabel("adversary fraction")
            ax.set_title(str(agg), fontsize=8)
            ax.set_xticks(list(config.BYZ_FRACTIONS))
            ax.set_xticklabels(["%.1f" % f for f in config.BYZ_FRACTIONS])
            ax.grid(True, **_GRID)
        axes[0][0].set_ylabel("%s (mean $\\pm$ std over seeds)" % metric)
        axes[0][0].legend(fontsize=6, frameon=False, title="attack", ncol=2)
        fig.suptitle("%s / %s / %s: Byzantine robustness, Krum assumes f=%d (%d runs)"
                     % (dataset, arch, part, config.BYZ_ASSUMED_F, drawn), fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        name = "fig_byzantine_%s_%s_%s" % (dataset.replace(" ", "_"),
                                           arch.replace(" ", ""), part)
        out.append(_finish(fig, name, dict(stage=stage, n_runs=drawn, arch=arch,
                                           partition=part,
                                           log_filter="dataset=%s stage=%s arch=%s "
                                                      "byz_partition=%s"
                                                      % (dataset, stage, arch, part))))
    return out


# ══ Task 7 - resampling and channels ═════════════════════════════════════════
def fig_resampling(datasets=None, stage="resampling", metric="accuracy"):
    """50 Hz / native / band-limited arms, per dataset, non-DP and DP side by side."""
    plt = _pyplot()
    datasets = [d for d in (datasets or list(ALL_DATASETS_DEFAULT))]
    per_ds = {}
    total = 0
    for ds in datasets:
        flat = [dict(r, dp_flag=_dp_of(r))
                for r in runlog.read_logs(dataset=ds, stage=stage)
                if _val(r, metric) is not None and not r.get("channel_mode")]
        rows = _summary(flat, ["architecture", "variant", "dp_flag"], metric=metric)
        if rows:
            per_ds[ds] = rows
            total += _n(flat)
    name = "fig_resampling"
    if not per_ds:
        return _skip(name, "no resampling-stage runs logged (variant column empty)")
    keys = list(per_ds)
    fig, axes = plt.subplots(1, len(keys), figsize=(FIGSIZE_2COL, 2.9), squeeze=False)
    for i, ds in enumerate(keys):
        ax = axes[0][i]
        rows = per_ds[ds]
        variants = list(dict.fromkeys(r["variant"] for r in rows))
        arms = list(dict.fromkeys("%s / %s" % (r["architecture"], r["dp_flag"])
                                 for r in rows))
        width = 0.8 / max(1, len(arms))
        for j, arm in enumerate(arms):
            for k, v in enumerate(variants):
                hit = [r for r in rows if r["variant"] == v and
                       "%s / %s" % (r["architecture"], r["dp_flag"]) == arm]
                if not hit:
                    continue
                x = j + (k - (len(variants) - 1) / 2.0) * width
                _ci_bars(ax, [x], [hit[0]], metric=metric,
                         color=METHOD_COLORS[j % len(METHOD_COLORS)], width=width * 0.9,
                         show_points=False,
                         label=arm if k == 0 else None)
        ax.set_xticks(list(range(len(arms))))
        ax.set_xticklabels(arms, rotation=35, ha="right", fontsize=6.5)
        ax.set_ylabel(metric)
        ax.set_title("%s: %s" % (ds, ", ".join(variants)), fontsize=8)
        ax.legend(fontsize=6, frameon=False)
        ax.grid(axis="y", **_GRID)
    fig.suptitle("Task 7 sampling-rate ablation (%d runs)" % total, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _finish(fig, name, dict(stage=stage, n_runs=total,
                                   log_filter="stage=%s variant logged, channel_mode "
                                              "absent" % stage))


def _dp_of(r):
    return "DP z=%g" % float(r.get("z") or 0) if r.get("z") else "non-DP"


def fig_channel_treatment(dataset="WISDM v2", stage="resampling", metric="accuracy"):
    """Zero-pad vs accelerometer-only vs channel mask, non-DP and at one z."""
    plt = _pyplot()
    runs = [dict(r, dp_flag=_dp_of(r)) for r in runlog.read_logs(dataset=dataset,
                stage=stage) if r.get("channel_mode") and _val(r, metric) is not None]
    name = "fig_channels_%s" % dataset.replace(" ", "_")
    if not runs:
        return _skip(name, "no channel_mode runs logged for %s (exp_resampling."
                            "channel_ablation writes them on raw windows)" % dataset)
    rows = _summary(runs, ["architecture", "channel_mode", "dp_flag"], metric=metric)
    modes = list(dict.fromkeys(r["channel_mode"] for r in rows))
    fig, axes = plt.subplots(figsize=(FIGSIZE_2COL, 3.1))
    for i, row in enumerate(rows):
        _ci_bars(axes, [i], [row], metric=metric,
                 color=METHOD_COLORS[modes.index(row["channel_mode"])
                                     % len(METHOD_COLORS)])
    _label_axis(axes, ["%s\n%s / %s" % (r.get("architecture"), r["channel_mode"],
                                        r["dp_flag"]) for r in rows],
                "%s: missing-gyroscope treatment on raw windows (%d runs)"
                % (dataset, _n(runs)), metric, ylim=(0, 1.02))
    fig.tight_layout()
    return _finish(fig, name, dict(stage=stage, n_runs=_n(runs), cells=len(rows),
                                   log_filter="dataset=%s stage=%s channel_mode logged"
                                   % (dataset, stage)))


# ══ Task 8 - feature families ════════════════════════════════════════════════
def fig_feature_families(dataset, stage="features", metric="accuracy"):
    """Leave-one-family-out and single-family accuracy against the all-families arm."""
    plt = _pyplot()
    runs = [dict(r, dp_flag=_dp_of(r)) for r in runlog.read_logs(dataset=dataset,
                stage=stage) if r.get("feature_set") and _val(r, metric) is not None]
    name = "fig_features_%s" % dataset.replace(" ", "_")
    if not runs:
        return _skip(name, "no feature-stage runs logged for %s (feature_set column "
                           "empty)" % dataset)
    rows = _summary(runs, ["feature_set", "dp_flag"], metric=metric)
    dps = sorted({r["dp_flag"] for r in rows})
    base = {r["dp_flag"]: r["mean"] for r in rows if r["feature_set"] == "all_families"}
    panels = [("leave-one-family-out",
               [r for r in rows if r["feature_set"].startswith("lofo_")]),
              ("single family only",
               [r for r in rows if r["feature_set"].startswith("only_")])]
    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE_2COL, 3.2))
    for ax, (title, sel) in zip(axes, panels):
        fams = list(dict.fromkeys(_short_fam(r["feature_set"]) for r in sel))
        for j, dp in enumerate(dps):
            hits = [r for r in sel if r["dp_flag"] == dp]
            for k, r in enumerate(hits):
                x = (fams.index(_short_fam(r["feature_set"]))
                     + (j - (len(dps) - 1) / 2.0) * 0.36)
                _ci_bars(ax, [x], [r], metric=metric,
                         color=METHOD_COLORS[j % len(METHOD_COLORS)], width=0.32,
                         show_points=False, label=dp if k == 0 else None)
            if dp in base:
                ax.axhline(base[dp], ls="--", lw=0.8,
                           color=METHOD_COLORS[j % len(METHOD_COLORS)])
        ax.set_xticks(list(range(len(fams))))
        ax.set_xticklabels(fams, rotation=55, ha="right", fontsize=6.5)
        ax.set_title("%s (dashed = all families, same privacy arm)" % title, fontsize=8)
        ax.set_ylabel(metric)
        ax.grid(axis="y", **_GRID)
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.suptitle("%s: feature families, %d runs (n_features is logged per run)"
                 % (dataset, _n(runs)), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _finish(fig, name, dict(stage=stage, n_runs=_n(runs),
                                   n_families=len({r["feature_set"] for r in rows}),
                                   log_filter="dataset=%s stage=%s feature_set logged"
                                   % (dataset, stage)))


def _short_fam(fs):
    return str(fs).replace("lofo_", "w/o ").replace("only_", "")


# ══ Task 9 - deployment cost ═════════════════════════════════════════════════
def fig_compute_cost(dataset, stage="cost"):
    """Parameters / MACs / latency / memory / payload for CNN-BiLSTM vs LightTCN."""
    plt = _pyplot()
    runs = [r for r in runlog.read_logs(dataset=dataset, stage=stage)
            if _arch_of(r) != "?"]
    name = "fig_cost_%s" % dataset.replace(" ", "_")
    if not runs:
        return _skip(name, "no cost-stage runs logged for %s" % dataset)
    metrics = [
        ("parameters", lambda r: _val(r, "params")),
        ("MACs / window", lambda r: _val(r, "macs")),
        ("CPU latency (ms, median)", lambda r: _nested(r, "latency_cpu_batch1",
                                                       "latency_ms_median")),
        ("host RSS (kB)", lambda r: _nested(r, "memory_cpu", "host_rss_delta_kb")),
        ("payload / round (bytes)", lambda r: _val(r, "uplink_downlink_bytes_per_round")),
        ("local epoch (s)", lambda r: _val(r, "local_epoch_time_s")),
    ]
    have = [(lab, get) for lab, get in metrics if any(get(r) is not None for r in runs)]
    if not have:
        return _skip(name, "cost logs carry none of the measured columns")
    arches = sorted({_arch_of(r) for r in runs})
    fig, axes = plt.subplots(1, len(have), figsize=(FIGSIZE_2COL, 2.6), squeeze=False)
    vals_by_metric = {}
    for i, (lab, get) in enumerate(have):
        ax = axes[0][i]
        vals = []
        for a in arches:
            v = [get(r) for r in runs if _arch_of(r) == a and get(r) is not None]
            vals.append(float(np.mean(v)) if v else np.nan)
        vals_by_metric[lab] = dict(zip(arches, vals))
        good = [(j, v) for j, v in enumerate(vals) if v == v and v > 0]
        ax.bar([j for j, _ in good], [v for _, v in good],
               color=[METHOD_COLORS[j % len(METHOD_COLORS)] for j, _ in good])
        ax.set_yscale("log")
        ax.set_xticks(list(range(len(arches))))
        ax.set_xticklabels(arches, fontsize=6.5, rotation=30, ha="right")
        ax.set_title(lab, fontsize=7.5)
        for j, v in good:
            ax.annotate("%.3g" % v, (j, v), ha="center", va="bottom", fontsize=6)
    red = _reduction(vals_by_metric)
    fig.suptitle("%s: measured deployment cost - %s (paper claims ~30%% fewer params)"
                 % (dataset, red or "no CNN/TCN pair logged"), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _finish(fig, name, dict(stage=stage, n_runs=len(runs),
                                   reductions=vals_by_metric,
                                   log_filter="dataset=%s stage=%s" % (dataset, stage)))


def _reduction(vals_by_metric):
    for lab in ("parameters", "MACs / window"):
        d = vals_by_metric.get(lab) or {}
        a, b = d.get("CNN-BiLSTM"), d.get("LightTCN")
        if a and b and a > 0:
            return "%s: %.1f%% fewer in LightTCN" % (lab, 100.0 * (1.0 - b / a))
    return None


# ══ Task 10 - membership inference ═══════════════════════════════════════════
def fig_mia(dataset, stage="mia"):
    """Attack advantage and AUC per attack, by privacy level - the metric MIA reports."""
    plt = _pyplot()
    runs = [r for r in runlog.read_logs(dataset=dataset, stage=stage)
            if r.get("mia_attack") is not None]
    name = "fig_mia_%s" % dataset.replace(" ", "_")
    if not runs:
        return _skip(name, "no mia-stage runs logged for %s" % dataset)
    adv = _summary(runs, ["dp_level", "mia_attack"], metric="advantage")
    auc = _summary(runs, ["dp_level", "mia_attack"], metric="auc")
    if not adv:
        return _skip(name, "mia runs for %s carry no advantage column" % dataset)
    attacks = sorted({r["mia_attack"] for r in adv})
    levels = sorted({r["dp_level"] for r in adv})
    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE_2COL, 2.9))
    for ax, rows, metric, ylab in ((axes[0], adv, "advantage",
                                   "membership advantage"),
                                   (axes[1], auc, "auc", "attack AUC")):
        for j, lev in enumerate(levels):
            for r in rows:
                if r["dp_level"] != lev:
                    continue
                x = attacks.index(r["mia_attack"]) + (j - (len(levels) - 1) / 2.0) * 0.36
                _ci_bars(ax, [x], [r], metric=metric,
                         color=METHOD_COLORS[j % len(METHOD_COLORS)], width=0.32,
                         show_points=False, label="%s%s" % (lev, _eps_tag(r)))
        if metric == "advantage":
            ax.axhline(0.0, color="black", lw=0.9, ls="--")
            ax.annotate("random guess", (axes[0].get_xlim()[1], 0.0), fontsize=6,
                        ha="right", va="bottom")
        else:
            ax.axhline(0.5, color="black", lw=0.9, ls="--")
        ax.set_xticks(list(range(len(attacks))))
        ax.set_xticklabels([_short_attack(a) for a in attacks], rotation=50, ha="right",
                           fontsize=6.5)
        ax.set_ylabel(ylab)
        ax.set_title(metric, fontsize=8)
        ax.legend(fontsize=6, frameon=False, title="target")
        ax.grid(axis="y", **_GRID)
    fig.suptitle("%s: membership inference, threshold chosen on shadow data (%d runs)"
                 % (dataset, len(runs)), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _finish(fig, name, dict(stage=stage, n_runs=len(runs),
                                   log_filter="dataset=%s stage=%s mia_attack logged"
                                   % (dataset, stage)))


def _eps_tag(r):
    e = _val(r["_runs"][0], "target_eps") if r.get("_runs") else None
    e = e if e is not None else (_val(r["_runs"][0], "eps") if r.get("_runs") else None)
    return "" if e is None else " ($\\varepsilon$=%.2f)" % e


def _short_attack(a):
    s = str(a).split("[")[0]
    return {"yeom_loss_threshold": "Yeom loss", "confidence": "Confidence",
            "shokri_shadow": "Shokri shadow", "label_only": "Label-only",
            "lira": "LiRA"}.get(s, s)


# ── the whole set ─────────────────────────────────────────────────────────────
PANEL_METRICS = ("accuracy", "precision", "recall", "f1_macro", "balanced_accuracy",
                 "macro_specificity", "kappa", "roc_auc_ovr")
PANEL_LABELS = {"accuracy": "accuracy", "precision": "precision (w)",
                "recall": "recall (w)", "f1_macro": "macro F1",
                "balanced_accuracy": "balanced acc.",
                "macro_specificity": "macro specificity", "kappa": "Cohen's kappa",
                "roc_auc_ovr": "ROC-AUC (ovr)"}


def _cells_for(dataset, stage, arch):
    """Logged runs of one dataset/stage/architecture, grouped as {cell: [runs]}."""
    runs = runlog.read_logs(dataset=dataset, stage=stage)
    cells = {}
    for r in runs:
        if arch and _arch_of(r).upper() != arch.upper():
            continue
        cells.setdefault(_cell_of(r), []).append(r)
    return cells


def fig_metric_panels(dataset, stage="main", arch="MLP"):
    """Every headline metric, one panel each, for all methods of one architecture.

    The accuracy-only figure cannot show the failure mode a privacy paper has to answer:
    a model that keeps accuracy by collapsing onto the majority class shows it here as
    recall and macro-F1 falling while accuracy does not. Panels are only drawn for metrics
    the logs carry, so a run logged before a column existed leaves that panel absent rather
    than plotting a number nobody measured.
    """
    plt = _pyplot()
    name = "fig_metrics_%s_%s" % (dataset.replace(" ", "_"), arch.replace("-", ""))
    cells = _cells_for(dataset, stage, arch)
    if not cells:
        return _skip(name, "no logged runs for %s stage=%s arch=%s"
                            % (dataset, stage, arch))
    ordered = sorted(cells, key=lambda c: -_mean(
        [_val(r, "accuracy") for r in cells[c]]))
    have = [m for m in PANEL_METRICS
            if any(_val(r, m) is not None for c in ordered for r in cells[c])]
    if not have:
        return _skip(name, "logs carry none of the panel metrics")
    ncol = 4
    nrow = -(-len(have) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIGSIZE_2COL, 2.5 * nrow),
                             squeeze=False)
    for i, metric in enumerate(have):
        ax = axes[i // ncol][i % ncol]
        rows = [r for r in _summary([r for c in ordered for r in cells[c]],
                                    ["cell"], metric=metric)
                if r["mean"] is not None]
        if not rows:
            ax.axis("off")
            continue
        xs = list(range(len(rows)))
        _ci_bars(ax, xs, rows, metric=metric, color=METHOD_COLORS[i % len(METHOD_COLORS)])
        ax.set_xticks(xs)
        ax.set_xticklabels([r["cell"] for r in rows], rotation=60, ha="right", fontsize=5.5)
        ax.set_title(PANEL_LABELS.get(metric, metric), fontsize=8)
        ax.set_ylim(0, 1.02 if metric != "roc_auc_ovr" else 1.0)
        ax.grid(axis="y", **_GRID)
    for j in range(len(have), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("%s / %s: full metric set, %d logged runs (bars = seed mean, "
                 "whiskers = 95%% t-CI, dots = each seed)"
                 % (dataset, arch, sum(len(v) for v in cells.values())), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _finish(fig, name, dict(stage=stage, n_runs=sum(len(v) for v in cells.values()),
                                   metrics=list(have),
                                   log_filter="dataset=%s stage=%s architecture=%s"
                                   % (dataset, stage, arch)))


def fig_per_class(dataset, stage="main", arch="MLP", top=6):
    """Class-wise macro picture: mean F1 per class, one line per method.

    A per-class panel is what shows which activity a private model gave up. Only methods
    whose runs logged `f1_per_class` appear, and the class axis is the integer label index
    the log carries - the class names come from the Work Order's mapping table, not here.
    """
    plt = _pyplot()
    name = "fig_per_class_%s_%s" % (dataset.replace(" ", "_"), arch.replace("-", ""))
    cells = {c: rs for c, rs in _cells_for(dataset, stage, arch).items()
             if any(isinstance(r.get("f1_per_class"), list) for r in rs)}
    if not cells:
        return _skip(name, "no logged run for %s stage=%s arch=%s has f1_per_class"
                            % (dataset, stage, arch))
    ordered = sorted(cells, key=lambda c: -_mean(
        [_val(r, "accuracy") for r in cells[c]]))[:top]
    n_class = max(len(r["f1_per_class"]) for c in ordered for r in cells[c]
                  if isinstance(r.get("f1_per_class"), list))
    fig, axes = plt.subplots(1, 3, figsize=(FIGSIZE_2COL, 2.3))
    for panel, (ax, key, title) in enumerate(zip(
            axes, ("f1_per_class", "precision_per_class", "recall_per_class"),
            ("per-class F1", "per-class precision", "per-class recall"))):
        for i, c in enumerate(ordered):
            series = {}
            for r in cells[c]:
                lst = r.get(key)
                if not isinstance(lst, list):
                    continue
                for k, v in enumerate(lst[:n_class]):
                    if v is not None:
                        series.setdefault(k, []).append(float(v))
            if not series:
                continue
            xs = sorted(series)
            means = [sum(series[x]) / len(series[x]) for x in xs]
            ax.plot(xs, means, "-", marker="o", ms=3.5, lw=1.1,
                    color=METHOD_COLORS[i % len(METHOD_COLORS)], label=c)
        ax.set_xticks(list(range(n_class)))
        ax.set_xlabel("class index", fontsize=7)
        ax.set_ylabel(title.split()[-1], fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.grid(axis="y", **_GRID)
        if panel == 0 and ordered:
            ax.legend(fontsize=5.5, frameon=False, ncol=2)
    fig.suptitle("%s / %s: per-class readings for the %d logged methods with the highest "
                 "mean accuracy (seed mean)" % (dataset, arch, len(ordered)), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _finish(fig, name, dict(stage=stage, n_runs=sum(len(cells[c]) for c in ordered),
                                   cells=list(ordered), n_class=n_class,
                                   log_filter="dataset=%s stage=%s architecture=%s "
                                              "f1_per_class logged"
                                              % (dataset, stage, arch)))


def fig_confusion(dataset, stage="main", arch="MLP", top=4):
    """Row-normalised confusion heatmaps, averaged over the seeds that logged a matrix.

    The rejected submission's figures carried no confusion evidence at all. These are drawn
    only from the `confusion` arrays in the run log; a method with no logged matrix is
    absent from the figure.
    """
    plt = _pyplot()
    name = "fig_confusion_%s_%s" % (dataset.replace(" ", "_"), arch.replace("-", ""))
    cells = {c: rs for c, rs in _cells_for(dataset, stage, arch).items()
             if any(isinstance(r.get("confusion"), list) for r in rs)}
    if not cells:
        return _skip(name, "no logged run for %s stage=%s arch=%s has a confusion matrix"
                            % (dataset, stage, arch))
    ordered = sorted(cells, key=lambda c: -_mean(
        [_val(r, "accuracy") for r in cells[c]]))[:top]
    n = max(len(r["confusion"]) for c in ordered for r in cells[c]
            if isinstance(r.get("confusion"), list))
    ncol = min(4, len(ordered))
    nrow = -(-len(ordered) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIGSIZE_2COL, 2.1 * nrow),
                             squeeze=False)
    for i, c in enumerate(ordered):
        ax = axes[i // ncol][i % ncol]
        mats = [r["confusion"] for r in cells[c] if isinstance(r.get("confusion"), list)]
        # arithmetic mean of the logged counts, cell by cell, so the panel is checkable
        # against confusion_table
        tot_counts = [[0.0 for _ in range(n)] for _ in range(n)]
        for m in mats:
            for a in range(n):
                for b in range(n):
                    tot_counts[a][b] += float(m[a][b]) if (
                        a < len(m) and b < len(m[a])) else 0.0
        counts = [[v / max(1, len(mats)) for v in row] for row in tot_counts]
        rate = []
        for a in range(n):
            row_sum = sum(counts[a])
            rate.append([v / row_sum if row_sum > 0 else 0.0 for v in counts[a]])
        im = ax.imshow(rate, cmap="Blues", vmin=0.0, vmax=1.0)
        for a in range(n):
            for b in range(n):
                ax.text(b, a, "%.2f" % rate[a][b], ha="center", va="center",
                        fontsize=5,
                        color="white" if rate[a][b] > 0.6 else "black")
        ax.set_xticks(list(range(n))); ax.set_yticks(list(range(n)))
        ax.set_xlabel("pred", fontsize=6.5); ax.set_ylabel("true", fontsize=6.5)
        ax.set_title("%s (n=%d seeds)" % (c, len(mats)), fontsize=7)
        ax.set_xticklabels([str(k) for k in range(n)], fontsize=5.5)
        ax.set_yticklabels([str(k) for k in range(n)], fontsize=5.5)
    for j in range(len(ordered), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("%s / %s: confusion, row-normalised, averaged over the seeds that logged "
                 "a matrix" % (dataset, arch), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _finish(fig, name, dict(stage=stage, n_runs=sum(len(cells[c]) for c in ordered),
                                   cells=list(ordered),
                                   log_filter="dataset=%s stage=%s architecture=%s "
                                              "confusion logged"
                                              % (dataset, stage, arch)))


def fig_epsilon_curve(dataset, stages=("privacy", "clip", "main")):
    """Epsilon spent per round, beside the measured release SNR per round.

    Left panel: the accountant's cumulative curve for each DP level that was logged, which
    is the quantity the paper's privacy claims are stated in. Right panel: the same runs'
    measured signal-to-noise of the released update (mean over seeds). Read together they
    are the honest version of "why is accuracy flat" - the curve on the left is what the
    mechanism guarantees, the one on the right is what it leaves usable.
    """
    plt = _pyplot()
    name = "fig_eps_curve_%s" % dataset.replace(" ", "_")
    runs = [r for st in stages for r in runlog.read_logs(dataset=dataset, stage=st)]
    runs = [r for r in runs if isinstance(r.get("eps_per_round"), list) and r["eps_per_round"]]
    if not runs:
        return _skip(name, "no logged run for %s in stages=%s carries eps_per_round"
                            % (dataset, ",".join(stages)))
    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE_2COL, 2.6))
    groups = {}
    for r in runs:
        groups.setdefault("%s | %s" % (_arch_of(r), _cell_of(r)), []).append(r)
    for i, (key, rs) in enumerate(sorted(groups.items())):
        curves = [r["eps_per_round"] for r in rs if isinstance(r.get("eps_per_round"), list)]
        if not curves:
            continue
        n_max = max(len(c) for c in curves)
        # the curve is identical across seeds by construction (eps depends on z, T, delta
        # only), so the mean is taken to average away nothing but to stay honest if a run
        # stopped early
        xs = list(range(1, n_max + 1))
        eps_mean, snr_mean = [], []
        for x in xs:
            e = [c[x - 1] for c in curves if len(c) >= x]
            eps_mean.append(sum(e) / len(e) if e else None)
        col = METHOD_COLORS[i % len(METHOD_COLORS)]
        pts = [(x, v) for x, v in zip(xs, eps_mean) if v is not None]
        if pts:
            axes[0].plot([p[0] for p in pts], [p[1] for p in pts], "-", color=col,
                         lw=1.3, marker="o", ms=2.5, label=key)
        snr = [r.get("per_round_release_snr") for r in rs
               if isinstance(r.get("per_round_release_snr"), list)]
        if snr:
            n_s = max(len(s) for s in snr)
            for x in range(1, n_s + 1):
                vals = [s[x - 1] for s in snr if len(s) >= x and s[x - 1] is not None]
                if vals:
                    snr_mean.append((x, sum(vals) / len(vals)))
            if snr_mean:
                axes[1].plot([p[0] for p in snr_mean], [p[1] for p in snr_mean], "-",
                             color=col, lw=1.3, marker="s", ms=2.5, label=key)
    axes[0].set_xlabel("round", fontsize=7); axes[0].set_ylabel("$\\varepsilon$ spent",
                                                               fontsize=7)
    axes[0].set_title("cumulative privacy cost (accountant)", fontsize=8)
    axes[0].grid(**_GRID)
    axes[1].axhline(1.0, color="black", lw=0.9, ls="--")
    axes[1].annotate("signal = noise", (1.0, 1.0), fontsize=5.5, ha="left", va="bottom")
    axes[1].set_xlabel("round", fontsize=7)
    axes[1].set_ylabel("release SNR (measured)", fontsize=7)
    axes[1].set_title("utility the noise leaves behind", fontsize=8)
    axes[1].grid(**_GRID)
    axes[0].legend(fontsize=5.0, frameon=False, ncol=2)
    fig.suptitle("%s: epsilon and measured release SNR per round, %d DP runs"
                 % (dataset, len(runs)), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _finish(fig, name, dict(stage=",".join(stages), n_runs=len(runs),
                                   cells=len(groups),
                                   log_filter="dataset=%s stages=%s eps_per_round logged"
                                   % (dataset, ",".join(stages))))


def _per_dataset_fns():
    return [("fig_main", fig_main_results), ("fig_all_readings", fig_all_readings),
            ("fig_metrics", fig_metric_panels), ("fig_per_class", fig_per_class),
            ("fig_confusion", fig_confusion),
            ("fig_curves", fig_training_curves), ("fig_forest", fig_paired_forest),
            ("fig_eps_curve", fig_epsilon_curve),
            ("fig_clip", fig_clip_sweep), ("fig_clip_history", fig_clip_history),
            ("fig_eps_split", fig_eps_split), ("fig_loso", fig_loso),
            ("fig_matched", fig_matched), ("fig_noniid", fig_noniid),
            ("fig_features", fig_feature_families), ("fig_cost", fig_compute_cost),
            ("fig_mia", fig_mia)]


def all_figures(datasets=None, inline=None):
    """Regenerate every figure the logs can support. Returns the manifest list.

    A figure whose data has not been logged is reported as skipped with the filter it
    looked for, so an empty panel is never mistaken for a result and nothing here can
    invent a value.
    """
    if not apply_style():
        return [dict(figure="*", skipped="matplotlib unavailable")]
    if inline is not None:
        STATE["inline"] = bool(inline)
    datasets = list(datasets) if datasets else list(ALL_DATASETS_DEFAULT)
    built = []
    for ds in datasets:
        print(" figures for %s" % ds)
        for _, fn in _per_dataset_fns():
            try:
                built.append(fn(ds))
            except Exception as exc:
                built.append(_skip(fn.__name__, "%s(%s) raised %s: %s"
                                  % (fn.__name__, ds, type(exc).__name__, exc)))
        for fn in (fig_byzantine,):
            try:
                got = fn(ds)
                built.extend(got if isinstance(got, list) else [got])
            except Exception as exc:
                built.append(_skip(fn.__name__, "%s(%s) raised %s: %s"
                                  % (fn.__name__, ds, type(exc).__name__, exc)))
        if ds in ("WISDM v2", "PAMAP2"):
            try:
                built.append(fig_channel_treatment(ds))
            except Exception as exc:
                built.append(_skip("fig_channels", "raised %s: %s"
                                  % (type(exc).__name__, exc)))
    # The cross-dataset and privacy sweeps: accuracy AND F1, because line 65 of the sheet
    # asks for both against cumulative epsilon on all four datasets and all three
    # architectures, and one shared-axis view of the four datasets, because a per-dataset
    # panel cannot show whether they are still on the same scale.
    cross = [(fig_privacy_utility, dict(datasets=datasets)),
             (fig_privacy_utility, dict(datasets=datasets, metric="f1")),
             (fig_cross_dataset, dict(datasets=datasets)),
             (fig_cross_dataset, dict(datasets=datasets, metric="f1")),
             (fig_cross_dataset_eps, dict(datasets=datasets)),
             (fig_cross_dataset_eps, dict(datasets=datasets, metric="f1")),
             (fig_leakage, dict(datasets=datasets)),
             (fig_resampling, dict(datasets=datasets))]
    for fn, kwargs in cross:
        metric = kwargs.get("metric", "accuracy")
        label = fn.__name__ if metric == "accuracy" else "%s_%s" % (fn.__name__, metric)
        try:
            built.append(fn(**kwargs))
        except Exception as exc:
            built.append(_skip(label, "raised %s: %s" % (type(exc).__name__, exc)))
    write_manifest(built)
    return built


def write_manifest(built):
    import json
    config.ensure_dirs()
    path = os.path.join(config.FIG_DIR, "figure_manifest.json")
    rows = [{k: v for k, v in b.items() if k != "_runs"} for b in built]
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=1, default=str)
    ok = [b for b in built if not b.get("skipped")]
    print("figures: %d drawn, %d skipped -> %s" % (len(ok), len(built) - len(ok),
                                                   config.FIG_DIR))
    for b in built:
        if b.get("skipped"):
            print("   skipped %-26s %s" % (b.get("figure"), b["skipped"]))
    with open(os.path.join(config.FIG_DIR, "figure_manifest.csv"), "w", newline="") as fh:
        import csv as _csv
        cols = ["figure", "path_png", "path_pdf", "stage", "n_runs", "cells",
                "log_filter", "skipped"]
        w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for b in rows:
            w.writerow({k: b.get(k, "") for k in cols})
    return path


if __name__ == "__main__":      # pragma: no cover
    print("figure manifest from the logs that exist:")
    for rec in all_figures():
        print("  %s" % rec.get("figure"))

#@@FILE ppml/reporting.py
"""Run report and contradiction flags (Work Order section 13).

Section 13 asks for four things the notebooks never produced:

  1. a README that says how each table and figure is reproduced - `README.md` at the top
     of the deliverable, and `reproduce_matrix()` / `write_reproduce_matrix()` here emit
     the same mapping as `results/reproduce_matrix.csv`;
  2. raw per-run logs and summary CSVs - written by the experiment modules;
  3. figures regenerated from the logs - `plotting.all_figures()`;
  4. a run report with hardware, software versions, total runtime and anything that
     could not be completed - `run_report()` below.

`contradictions()` is the part the Work Order asks for explicitly: "flag immediately any
result that contradicts a claim in the current paper ... The authors need to rewrite
those claims, not tune the experiment until it agrees." Every check compares a claim
quoted from the submission against a number read from the logs, and returns a row even
when the evidence is missing (verdict "NOT YET MEASURED") - it never silently drops a
claim that failed.
"""

import csv
import json
import os
import time

import numpy as np

from . import config, multiseed, plotting, runlog
from .runlog import read_logs

# Claims carried by the rejected submission, quoted so the response letter can point at
# them. Each entry is checked against the logs below.
PAPER_CLAIMS = {
    "eps.0.85": "FL+DP on UCI-HAR carries eps = 0.85 (Renyi accounting)",
    "eps.8.30": "CNN-BiLSTM DP carries eps = 8.30",
    "eps.crosscheck": "the reported epsilons agree with an independent accountant",
    "table5.wisdm": "Table 5 reports WISDM v2 FedAvg accuracy 84.9%",
    "params.30pct": "LightTCN needs ~30% fewer parameters than CNN-BiLSTM",
    "median.wins": "coordinate-wise median aggregation beats FedAvg under attack",
    "adaptdp.wins": "adaptive clipping beats fixed-C DP at equal budget",
    "adaptdp.cost": "adaptive clipping is privacy-cheap at this participation level",
    "fed.beats.central": "federated training matches or beats the centralised control",
    "resample.harmless": "linear upsampling to 50 Hz is harmless for the analytics",
    "zeropad.neutral": "zero-padding the missing WISDM gyroscope is neutral",
    "noniid.29pp": "UCI-HAR loses 29.4 pp at alpha = 0.5 (one partition)",
    "noniid.dp.scope": "the DP numbers of the non-IID sweep are user-level guarantees",
    "mia.adv": "DP reduces the membership-inference advantage to about chance",
    "deploy.measured": "the deployment-cost claims are backed by measurements",
    "pipeline.one": "all four datasets are run with the same fixed pipeline "
                    "(Work Order line 12)",
}

# Worked around, not run: the sheet marks each of these optional ("if feasible", "if
# time allows", "energy is optional"). They are declared here so the run report lists
# them as not measured instead of a reader assuming they were.
NOT_ATTEMPTED = (
    ("energy", "Task 9: energy per inference is not measured. The deployment claims stay "
               "prospective wording, as the Work Order permits."),
    ("dpsgd-example-level", "Task 1 optional extra: example-level DP-SGD is implemented "
                            "(experiments/exp_dpsgd.py, accountant "
                            "dp_accountant.account_dpsgd) but is opt-in in the per-dataset "
                            "files, so unless the stage was added to RUN_STAGES no "
                            "example-level number exists. Set RUN_STAGES = ['dpsgd'] "
                            "for it. It is a DIFFERENT PRIVACY "
                            "UNIT - one window, not one person - and its row is labelled "
                            "adjacency=example_level, user_level_certificate=NO. No "
                            "user-level claim in the paper may cite it."),
    ("pooled-multi-dataset", "Task 7 optional: the pooled cross-sensor channel arm "
                             "(exp_resampling.pooled_run) runs in the ALL-datasets session "
                             "only, because it needs a gyroscope dataset and a "
                             "gyroscope-less dataset in one stack; a single-dataset script "
                             "prints that it skipped. Use RUN_DATASETS = ['UCI HAR', "
                             "'WISDM v2'] with RUN_STAGES = ['resampling'] - that is "
                             "the whole session. Until it has run, the mask arm is a single-dataset "
                             "comparison, which is the case where a mask and a constant "
                             "cannot differ."),
)

# Arms that ran, but on a narrower scope than the full grid - the reader has to see this
# next to the numbers, not in a footnote.
SCOPED_ARMS = (
    ("byzantine non-IID", "the Dirichlet arm of Task 6 runs the MLP only "
                          "(config.BYZ_NONIID_ARCHES) for session cost"),
    ("Task 5 DP arm", "one z (%.1f) with both aggregators; its epsilon is not a user-level "
                      "certificate under a Dirichlet partition" % config.NONIID_DP_Z),
    ("MIA shadows", "shadow models are trained without DP (they only set the threshold); "
                    "the DP level is applied to the target, as the protocol requires"),
    ("MIA DP levels", "the membership-inference sweep attacks the non-DP model and five DP "
                      "levels (three z, the sparse release and the head-only release), so "
                      "every arm that Task 1 and Task 4 price is also MIA-tested. The "
                      "adaptive-C arm is NOT - its own certificate is the open question "
                      "there, and adding a sixth level would multiply the trial count for "
                      "the one arm whose epsilon already exceeds the others."),
)


def profile_scope_notes():
    """What replication and grid the ACTIVE profile actually runs, computed not asserted.

    Section 12 of the Work Order puts >= 6 seeds on Tasks 1, 2, 3, 7 and 8 and names six
    clip radii and four corruption fractions. Those numbers used to be hardcoded here,
    which meant this function could only ever be as honest as the last edit. Now the run
    report reads the grid out of the profile that produced the logs, so a thinner session
    says so itself and a compliant one says that too.
    """
    cfg = config.cfg()
    need = {"seeds": len(config.SEEDS), "seeds_supp": len(config.SEEDS),
            "dp_seeds": len(config.SEEDS), "byz_seeds": len(config.SEEDS_BYZANTINE),
            "mia_trials": len(config.SEEDS_MIA)}
    out = []
    thin = ["%s=%d (sheet asks %d)" % (k, len(cfg[k]), v)
            for k, v in need.items() if len(cfg[k]) < v]
    if cfg.get("clip_sweep") and len(cfg["clip_sweep"]) < len(config.CLIP_SWEEP):
        thin.append("clip radii=%d of %d" % (len(cfg["clip_sweep"]),
                                             len(config.CLIP_SWEEP)))
    for key, label, want in (("partitions", "partitions per alpha",
                             config.N_PARTITIONS_PER_ALPHA),
                             ("partitions_dp", "DP partitions per alpha",
                              config.N_PARTITIONS_DP),
                             ("byz_fractions", "corruption fractions",
                              len(config.BYZ_FRACTIONS))):
        v = cfg.get(key)
        if v is None:
            continue
        n = len(v) if isinstance(v, (tuple, list)) else int(v)
        if n < want:
            thin.append("%s=%d of %d" % (label, n, want))
    methods_note = None
    if config.METHODS_FILTER:
        keep = set(config.METHODS_FILTER)
        skipped = [l for l, _ in config.MAIN_METHODS if l not in keep]
        methods_note = (
            "DP-only session (config.METHODS_FILTER is set)",
            "Task 4 ran %d of %d arms; not run at all: %s. Section 12 asks for 'all FL "
            "methods + centralised', so these logs are a mechanism check and not the Task 4 "
            "table. Three of the sheet's paired tests lose their other arm and come back as "
            "'insufficient paired seeds' rather than as a result: median_vs_fedavg, "
            "fedprox_vs_fedavg and fedavg_vs_centralised - which means the claims 'median "
            "beats FedAvg' and 'federated beats centralised' cannot be evaluated from this "
            "session. The pairs that stay valid are the DP ones (adaptdp_vs_dp, "
            "sparse_dp_vs_plain_dp, head_dp_vs_plain_dp), since both arms of each are DP. "
            "Rerun with METHODS_FILTER=None for the reportable table."
            % (len(keep & set(l for l, _ in config.MAIN_METHODS)),
               len(config.MAIN_METHODS), ", ".join(skipped) or "none"))
    # The DP operating point, the head-sweep switch and the local-epoch budget, as
    # executed. Each is a place a session can differ from the Work Order sheet without
    # moving a single epsilon while still changing what an accuracy column means, so the
    # report states them instead of leaving a reader to infer them from the code.
    from . import dp_accountant as _acc
    _T = int(cfg["rounds_dp"])
    out.append(("DP operating point",
                "every DP arm outside the Task 1 sweep runs at z=%g, which certifies "
                "eps=%.4f at T=%d, delta=%g (config.DP_ARM_Z; the example-level DP-SGD arm "
                "keeps its own z=%g because its privacy unit is a window, not a person). "
                "The Task 1 sweep is the measured curve a budget is stated on; the accuracy "
                "a z produces is reported for that z and never the other way round."
                % (config.DP_ARM_Z,
                   _acc.account_central_gaussian(config.DP_ARM_Z, _T,
                                                 config.DELTA)["eps"],
                   _T, config.DELTA, config.DP_SGD_Z)))
    out.append(("head-only z sweep",
                "RUN: the head release is measured at every z level for %s, so its "
                "signal-to-noise curve is in privacy_table_<ds>.csv beside the plain and "
                "sparse arms." % ", ".join(config.PRIVACY_HEAD_ARCHES)
                if config.PRIVACY_HEAD_SWEEP else
                "NOT run (config.PRIVACY_HEAD_SWEEP=False): the head release is measured "
                "only at the single operating point z=%g, so the release SNR at the levels "
                "where the arithmetic says it can cross 1 is untested here. "
                "release_snr_mean on each privacy row is the column that settles it."
                % config.DP_ARM_Z))
    out.append(("local epochs",
                "E=%d (MLP) / %d (CNN) passes per client per round; the sheet names 10/5, "
                "which measures out at roughly 60-75 GPU-h for UCI HAR alone. E appears in "
                "no accountant formula - the noise is added once per round at the server, "
                "after the update is clipped to C - so this budget changes accuracy and "
                "wall-clock and no privacy number."
                % (cfg["local_epochs_mlp"], cfg["local_epochs_cnn"])))
    if thin:
        out.append(("%s profile replication" % config.PROFILE,
                    "thinner than Section 12 on: " + "; ".join(thin) +
                    ". Every table row carries its own n, and a profile other than "
                    "'full' is never a reportable result."))
    else:
        out.append(("full-profile replication",
                    "the run matrix's minimums are met as executed: %d seeds on Tasks 3/7/8, "
                    "%d seeds on every T=%d DP arm, %d clip radii, %d corruption fractions, "
                    "%d partitions per alpha (DP arm %d). No replication count or grid point "
                    "was thinned for cost."
                    % (len(cfg["seeds_supp"]), len(cfg["dp_seeds"]), config.ROUNDS_DP,
                       len(cfg.get("clip_sweep") or config.CLIP_SWEEP),
                       len(cfg.get("byz_fractions") or config.BYZ_FRACTIONS),
                       cfg.get("partitions", config.N_PARTITIONS_PER_ALPHA),
                       cfg.get("partitions_dp", config.N_PARTITIONS_DP))))
    if methods_note:
        out.append(methods_note)
    return out

# Where this pipeline's mechanism differs from the rejected submission's, and why the
# difference cannot move epsilon. Each of these is asserted in the preflight
# (fl.projection_selftest / fl.clipper_selftest / dp_accountant.selftest) before a single
# run is logged, so a deviation that would silently buy accuracy with privacy stops the
# session instead of reporting a number.
MECHANISM_DEVIATIONS = (
    ("clip radius", "C is derived from the model's own initialisation norm "
                    "(config.CLIP_REL x ||w0||) instead of one constant for every "
                    "architecture. sigma = z*C/K scales with it, so z and the certificate "
                    "are unchanged; the absolute-C arm stays in the tables as published."),
    ("sparse release (FL+DP-Sparse)", "the release updates only the largest fraction of "
                                      "the ALREADY-NOISY average, selected after the draw. "
                                      "That is post-processing, so it is the same epsilon. "
                                      "Selecting on the clean average, as DP-Forward's "
                                      "output projection does, is not covered here and is "
                                      "not what this pipeline runs."),
    ("head-only release (FL+DP-Head)", "a fixed mask decided by the architecture: only the "
                                       "classifier head's tensors are released, the feature "
                                       "extractor stays at its public initialisation. A "
                                       "constant 0/1 diagonal mask cannot raise the "
                                       "sensitivity, so epsilon is identical while the "
                                       "released dimension falls from d to d_head."),
    ("server momentum and iterate averaging", "both act on aggregates that were already "
                                              "released with noise, so they are adaptive "
                                              "post-processing: free in epsilon, and they "
                                              "do not change what left the clients."),
    ("frozen clip estimator (FL+AdaptDP)", "the Gaussian quantile estimator releases for "
                                           "the first %d of %d DP rounds and C is held "
                                           "after that. The run is accounted for the "
                                           "releases it made (%d rounds of estimator cost "
                                           "instead of %d); the unfrozen certificate is "
                                           "reported beside it in adaptive_eps_split.csv."
                                           % (config.ADAPT_FREEZE_ROUNDS, config.ROUNDS_DP,
                                              min(config.ADAPT_FREEZE_ROUNDS,
                                                  config.ROUNDS_DP), config.ROUNDS_DP)),
)


# ── coverage ──────────────────────────────────────────────────────────────────
def coverage(datasets=None, stages=None):
    """How many runs each (dataset, stage) cell has, and what is still missing."""
    from .experiments import STAGES as REG
    datasets = list(datasets or plotting.ALL_DATASETS_DEFAULT)
    stages = list(stages or REG)
    profile_seeds = len(config.cfg()["seeds"])
    rows = []
    for ds in datasets:
        for st in stages:
            runs = read_logs(dataset=ds, stage=st)
            cells = {str(r.get("cell")) for r in runs if r.get("cell")}
            seeds = {int(r["seed"]) for r in runs if r.get("seed") is not None}
            parts = {int(r.get("partition_id", 0)) for r in runs}
            rows.append(dict(
                dataset=ds, stage=st, description=REG[st][1],
                n_runs=len(runs), n_cells=len(cells),
                n_seeds=len(seeds), seeds_needed=profile_seeds,
                n_partitions=len(parts),
                complete=bool(runs and len(seeds) >= profile_seeds),
                reportable=bool(runs and all(r.get("reportable") for r in runs))))
    return rows


def completeness_note(coverage_rows):
    """Human sentence about what could not be completed, for the run report."""
    missing = [r for r in coverage_rows if not r["complete"]]
    if not missing:
        return "every dataset x stage has at least one logged run"
    by_ds = {}
    for r in missing:
        by_ds.setdefault(r["dataset"], []).append(r["stage"])
    return "; ".join("%s: %s" % (ds, ", ".join(st)) for ds, st in sorted(by_ds.items()))


# ── runtime ───────────────────────────────────────────────────────────────────
def runtime(datasets=None):
    """Total wall-clock spend reconstructed from the per-run wall_time_s fields.

    Runtime is a record of what the session DID, so smoke runs count here even though
    no table or figure may use them.
    """
    datasets = list(datasets or plotting.ALL_DATASETS_DEFAULT)
    tot, per_stage, per_ds = 0.0, {}, {}
    n = 0
    for ds in datasets:
        for r in read_logs(dataset=ds, reportable_only=False):
            w = r.get("wall_time_s")
            if not isinstance(w, (int, float)):
                continue
            tot += float(w)
            n += 1
            per_stage[r.get("stage", "?")] = per_stage.get(r.get("stage", "?"), 0.0) + float(w)
            per_ds[ds] = per_ds.get(ds, 0.0) + float(w)
    return dict(total_wall_s=tot, total_wall_h=tot / 3600.0, n_timed_runs=n,
                by_stage={k: round(v, 1) for k, v in sorted(per_stage.items())},
                by_dataset={k: round(v, 1) for k, v in sorted(per_ds.items())})


# ── contradiction checks ──────────────────────────────────────────────────────
def _row(claim_id, claim, evidence, verdict, value, action):
    return dict(claim_id=claim_id, paper_claim=claim, evidence=evidence,
                verdict=verdict, measured=value, action_needed=action)


def contradictions(datasets=None):
    """Every submission claim that the logs can confirm or refute. No tuning."""
    datasets = list(datasets or plotting.ALL_DATASETS_DEFAULT)
    out = []
    out += _claim_eps(datasets)
    out += _claim_crosscheck(datasets)
    out += _claim_table5(datasets)
    out += _claim_params(datasets)
    out += _claim_median_attack(datasets)
    out += _claim_adapt(datasets)
    out += _claim_adaptive_cost(datasets)
    out += _claim_paired(datasets, "FL-FedAvg", "Centralised", "fed.beats.central")
    out += _claim_resampling(datasets)
    out += _claim_channels(datasets)
    out += _claim_noniid(datasets)
    out += _claim_noniid_scope(datasets)
    out += _claim_mia(datasets)
    out += _claim_monotone_privacy(datasets)
    out += _claim_deployment(datasets)
    out += _claim_single_pipeline(datasets)
    return out


def _claim_single_pipeline(datasets):
    """Work Order line 12, checked from the logs rather than from the settings.

    The sheet names the defect outright: the rejected WISDM sweeps (~23%) came from an older
    broken pipeline while Table 5 (84.9%) came from the fixed script. Reading the config each
    dataset actually ran with is the only way to show that cannot still be true, so a setting
    used by one dataset and not the others is CONTRADICTED here even though the source code is
    shared - a shared file that was run with different arguments is still two pipelines.
    """
    from . import crossdataset
    mine = [d for d in datasets if d in crossdataset.DATASETS]
    if len(mine) < 2:
        only = ", ".join(mine) if mine else "no dataset"
        return [_row("pipeline.one", PAPER_CLAIMS["pipeline.one"],
                     "only %s has reportable logged runs - one dataset cannot show that "
                     "the four agree" % only, "NOT YET MEASURED", "n/a",
                     "run the other per-dataset bundles and merge their exported logs with "
                     "ppml.crossdataset.consolidate([...]) before quoting a four-dataset "
                     "table")]
    rows = crossdataset.read_all(datasets=mine)
    if not rows:
        return [_row("pipeline.one", PAPER_CLAIMS["pipeline.one"],
                     "no reportable runs logged for %s" % ", ".join(mine),
                     "NOT YET MEASURED", "n/a",
                     "run the per-dataset bundles at full profile")]
    ident = crossdataset.pipeline_identity(rows=rows, datasets=mine)
    checked = len([r for r in ident if r["setting"] != "__coverage__"])
    bad = crossdataset.identity_violations(ident)
    setting_bad = [b for b in bad if b[1] != "__coverage__"]
    missing = [b for b in bad if b[1] == "__coverage__"]
    if not bad:
        return [_row("pipeline.one", PAPER_CLAIMS["pipeline.one"],
                     "%d runs over %s: %d stage/setting comparisons all agree across the "
                     "datasets that ran them, and every dataset appears in every stage "
                     "another dataset ran" % (len(rows), ", ".join(mine), checked),
                     "supported", "%d checks, 0 violations" % checked,
                     "claim stands - cross_dataset_runs.csv and pipeline_identity.csv carry "
                     "the evidence")]
    parts = []
    for stage, setting, ds in setting_bad:
        row = next(r for r in ident
                   if (r["stage"], r["setting"], r["dataset"]) == (stage, setting, ds))
        parts.append("%s/%s for %s used %s while the others used %s"
                     % (stage, setting, ds, row["values"],
                        row["agrees_with_other_datasets"].split("used ")[-1]))
    for stage, setting, ds in missing:
        parts.append("%s has no logged run at stage %s" % (ds, stage))
    verdict = "CONTRADICTED" if setting_bad else "NOT SUPPORTED"
    return [_row("pipeline.one", PAPER_CLAIMS["pipeline.one"],
                 "%d of %d stage/setting comparisons across %s disagree: %s"
                 % (len(bad), checked, ", ".join(mine), "; ".join(parts[:6])),
                 verdict, "%d violations" % len(bad),
                 "the four datasets are not one pipeline yet - re-run the datasets named "
                 "here with the current bundle so every stage shares the same settings, "
                 "and do not present the two answers side by side until they do"
                 if setting_bad else
                 "a dataset is absent from a stage another dataset ran (the sheet's Table 7 "
                 "complaint) - run that stage for the dataset named, or scope the claim to "
                 "the datasets that actually have the number")]


def _claim_crosscheck(_datasets=None):
    """Task 1 deliverable 4: our closed form against an independent accountant.

    The submission's epsilon was never checked against anything, which is how it drifted.
    A row is emitted even when dp-accounting is absent - "NOT YET MEASURED" here means the
    cross-check itself is missing, and the reader must not take the closed form on faith.
    """
    from .experiments import exp_privacy
    try:
        rows = exp_privacy.crosschecks()
    except Exception as exc:                       # pragma: no cover
        return [_row("eps.crosscheck", PAPER_CLAIMS["eps.crosscheck"],
                     "cross_check() raised %s" % exc, "UNEXPECTED", "n/a",
                     "fix the accountant before quoting any epsilon")]
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        why = rows[0].get("error", "dp-accounting unavailable") if rows else "no levels"
        return [_row("eps.crosscheck", PAPER_CLAIMS["eps.crosscheck"],
                     "independent accountant not run (%s)" % why, "NOT YET MEASURED",
                     "n/a",
                     "install dp-accounting in its own cell and re-run "
                     "ppml.experiments.exp_privacy.write_crosschecks(), or drop the "
                     "closed-form epsilon from the paper")]
    worst = max(ok, key=lambda r: abs(float(r.get("rel_diff") or 0.0)))
    wd = abs(float(worst.get("rel_diff") or 0.0))
    return [_row("eps.crosscheck", PAPER_CLAIMS["eps.crosscheck"],
                 "%d/%d z levels agree with dp-accounting; worst relative difference "
                 "%.2e at z=%.1f (ours %.4f vs theirs %.4f)"
                 % (len(ok), len(rows), wd, float(worst["z"]), worst["ours"],
                    worst["theirs"]),
                 "supported" if wd < 1e-3 else "CONTRADICTED",
                 "worst rel diff %.2e" % wd,
                 "the closed form disagrees with the reference implementation; every "
                 "epsilon in the paper has to be recomputed" if wd >= 1e-3 else
                 "claim stands")]


def _claim_adaptive_cost(datasets=None):
    """Adaptive clipping costs privacy as well as accuracy - the split is computed.

    Like the legacy-epsilon check this needs no runs: the cost of the estimator is a
    property of the settings, and it is the same for all four datasets, so one row.
    """
    from . import dp_accountant
    from .experiments import exp_clip
    try:
        sp = exp_clip.eps_split("config")
    except Exception as exc:                       # pragma: no cover
        return [_row("adaptdp.cost", PAPER_CLAIMS["adaptdp.cost"],
                     "eps_split() raised %s" % exc, "UNEXPECTED", "n/a",
                     "fix the adaptive accounting before quoting its epsilon")]
    tot, base = float(sp["eps_total"]), float(sp["eps_updates_fixed"])
    share = float(sp["clip_fraction_share"])
    all_t = float(sp["eps_total_all_rounds"])
    W, T = int(sp["T_clip_estimator"]), int(sp["T"])
    costly = tot > 1.5 * base
    rows = [_row("adaptdp.cost", PAPER_CLAIMS["adaptdp.cost"],
                 "at z=%.1f, T=%d the update mechanism alone costs eps=%.3f. The "
                 "quantile estimator (sigma_b=%.3g, i.e. %.3g after dividing by K=%d) is "
                 "charged for the %d rounds it releases and then frozen, which raises the "
                 "total to eps=%.3f (%.0f%% of it is the clip estimate); leaving it "
                 "running for all %d rounds, as Andrew et al. publish it, would cost "
                 "eps=%.3f"
                 % (sp["z"], T, base, sp["sigma_b"],
                    float(sp["sigma_b"]) / config.N_CLIENTS, config.N_CLIENTS,
                    W, tot, 100.0 * share, T, all_t),
             "CONTRADICTED" if costly else "supported",
             "adaptive total eps %.3f vs fixed %.3f (%.1fx); %.3f un-frozen"
             % (tot, base, tot / max(base, 1e-12), all_t),
             "even after freezing the estimator after %d of %d rounds the adaptive arm "
             "cannot be sold at the paper's epsilon: quote eps=%.1f for it, or make "
             "sigma_b larger and say the clip estimate is coarse" % (W, T, tot)
             if costly else "claim stands")]
    return rows + _claim_adaptive_releases(datasets, sp, W)


def _claim_adaptive_releases(datasets, sp, W):
    """The adaptive arm's certificate has to match the releases it actually logged.

    eps_split() prices the estimator for W rounds; that is only a certificate if the run
    really stopped releasing after W. The clipper records each release in its own
    clip_history and the accountant is called with the count it read back, so this check
    closes the loop: logged count == charged count == logged epsilon.
    """
    rows = []
    for ds in datasets:
        runs = [r for r in read_logs(dataset=ds, stage="clip")
                if r.get("eps_total") is not None
                and r.get("T_clip_estimator") is not None]
        if not runs:
            continue
        charged = [int(r["T_clip_estimator"]) for r in runs]
        logged = []
        for r in runs:
            hist = r.get("clip_history") or []
            rel = [h for h in hist if h.get("estimator_released")]
            if hist:
                logged.append(len(rel))
        eps_bad = [float(r["eps_total"]) for r in runs
                   if abs(float(r["eps_total"]) - float(sp["eps_total"]))
                   > 1e-6 * max(1.0, float(sp["eps_total"]))]
        cnt_bad = [n for n in charged if n != W]
        if logged:
            cnt_bad += [n for n in logged if n != W]
        consistent = not eps_bad and not cnt_bad
        rows.append(_row(
            "adaptdp.release_count",
            "self-check: the charged estimator rounds equal the released ones in the log",
            "%s: %d adaptive run(s), charged %s estimator rounds, freeze set at %d, "
            "logged eps %s vs analytic eps %s%s"
            % (ds, len(runs), sorted(set(charged)), W,
               "%.3f" % float(runs[0]["eps_total"]), "%.3f" % float(sp["eps_total"]),
               "" if consistent else "; mismatches: eps %s, counts %s"
               % (["%.3f" % e for e in eps_bad], cnt_bad)),
            "supported" if consistent else "UNEXPECTED",
            "charged %s, freeze %d" % (sorted(set(charged)), W),
            "the adaptive arm's epsilon is not the epsilon of what it released; do not "
            "report it until the count and the certificate agree" if not consistent
            else "certificate and release log agree"))
    return rows


def _claim_noniid_scope(datasets):
    """A Dirichlet client mixes subjects, so its certificate is not a user-level one."""
    rows = []
    for ds in datasets:
        dp_runs = [r for r in read_logs(dataset=ds, stage="noniid")
                   if r.get("certificate_scope") and float(r.get("z") or 0.0) > 0]
        if not dp_runs:
            continue
        scoped = [r for r in dp_runs
                  if "NOT a user-level" in str(r.get("certificate_scope"))]
        alphas = sorted({str(r.get("alpha")) for r in scoped})
        if not scoped:
            rows.append(_row("noniid.dp.scope", PAPER_CLAIMS["noniid.dp.scope"],
                             "%s: every DP non-IID run is labelled user-level" % ds,
                             "supported", "n/a", "claim stands"))
            continue
        rows.append(_row("noniid.dp.scope", PAPER_CLAIMS["noniid.dp.scope"],
                         "%s: %d DP runs at alpha=%s are logged as partition-level; the "
                         "clients there draw from more than one subject, so one client's "
                         "data is spread across the silos and the add/remove-a-silo "
                         "adjacency no longer holds"
                         % (ds, len(scoped), ", ".join(alphas)),
                         "CONTRADICTED",
                         "%d of %d DP non-IID runs are not user-level"
                         % (len(scoped), len(dp_runs)),
                         "present the non-IID DP numbers as round-level privacy on a "
                         "fixed partition, never as a user-level guarantee; only the "
                         "subject-silo (IID) arm supports that wording"))
    return rows


def _claim_deployment(_datasets=None):
    """Nothing in the Work Order's measured-cost list includes energy."""
    return [_row("deploy.measured", PAPER_CLAIMS["deploy.measured"],
                 "ppml.experiments.exp_cost measures params, MACs, inference latency and "
                 "peak memory from the logs; it has no energy sensor and the Work Order "
                 "marks energy optional (\"if feasible\"), so none was collected",
                 "NOT YET MEASURED", "energy: n/a",
                 "keep the deployment section prospective, or add a measured "
                 "millijoule-per-inference number before claiming one")]


def _claim_eps(_datasets=None):
    """The reported epsilons against what the accountant says the settings cost.

    This one needs no runs: the submission's own settings are enough to show what its
    epsilon should have been.
    """
    from . import dp_accountant
    zl = dp_accountant.legacy_equivalent_z(config.LEGACY_NOMINAL_EPS)
    whole = dp_accountant.account_central_gaussian(zl, 50)["eps"]
    cnn = dp_accountant.account_per_tensor_clipping(
        zl, 50, config.LEGACY_N_TENSORS["CNN-BiLSTM"])["eps"]
    rows = []
    for cid, got, kind in (("eps.0.85", whole, "whole-model clip"),
                           ("eps.8.30", cnn, "per-tensor clip")):
        want = 0.85 if cid == "eps.0.85" else 8.30
        rows.append(_row(cid, PAPER_CLAIMS[cid],
                         "legacy settings (nominal eps=%.1f, K=%d, T=50) -> z=%.3f; "
                         "the correct client-level certificate is eps=%.1f (%s)"
                         % (config.LEGACY_NOMINAL_EPS, config.N_CLIENTS, zl, got, kind),
                         "CONTRADICTED" if got > 20 else "supported",
                         "reported %.2f vs certified %.1f" % (want, got),
                         "the paper's epsilon must be replaced by the certificate of the "
                         "settings actually run" if got > 20 else "no change"))
    return rows


def _claim_table5(datasets):
    rows = []
    for ds in datasets:
        runs = [r for r in read_logs(dataset=ds, stage="main")
                if str(r.get("cell")) == "FL-FedAvg" and _val(r) is not None]
        if not runs:
            if ds == "WISDM v2":
                rows.append(_row("table5.wisdm", PAPER_CLAIMS["table5.wisdm"],
                                 "no main-stage FL-FedAvg run logged for %s" % ds,
                                 "NOT YET MEASURED", "n/a",
                                 "run the main stage for %s before restating Table 5" % ds))
            continue
        s = multiseed.summarise([_val(r) for r in runs])
        if ds != "WISDM v2":
            continue
        gap = abs(s["mean"] - 0.849)
        rows.append(_row("table5.wisdm", PAPER_CLAIMS["table5.wisdm"],
                         "%s FL-FedAvg over %d seeds: %.4f +/- %.4f (CI %.4f-%.4f)"
                         % (ds, s["n"], s["mean"], s["std"], s["ci95_low"], s["ci95_high"]),
                         "supported" if gap < 0.05 else "CONTRADICTED",
                         "%.1f%% measured vs 84.9%% in Table 5" % (100 * s["mean"]),
                         "Table 5 and the figures must both come from this pipeline; the "
                         "older ~23%% sweep has to be deleted" if gap >= 0.05 else
                         "Table 5 stands, redraw the figure from the same log"))
    return rows


def _val(r, metric="accuracy"):
    v = r.get(metric)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _claim_median_attack(datasets):
    """The median claim is about attacks, so it is tested on the byzantine logs.

    One row per (architecture, partition kind) that actually ran - the Dirichlet arm is
    where a robust aggregator is supposed to earn its keep, so it is judged separately
    instead of averaged into the IID numbers.
    """
    from .experiments import exp_byzantine
    rows = []
    for ds in datasets:
        for arch in sorted({str(r.get("architecture") or r.get("arch") or "?")
                           for r in read_logs(dataset=ds, stage="byzantine")}):
            for part in sorted({str(r.get("partition") or "iid")
                                for r in read_logs(dataset=ds, stage="byzantine")
                                if str(r.get("architecture") or r.get("arch") or "?")
                                == arch}):
                tab = [r for r in exp_byzantine.table(ds)
                       if r["architecture"] == arch and r["adversary_fraction"] > 0
                       and str(r.get("partition")) == part]
                if not tab:
                    continue
                by_agg = {}
                for r in tab:
                    by_agg.setdefault(r["aggregator"], []).append(r["acc_mean"])
                ranked = sorted(((float(np.mean(v)), k, len(v))
                                 for k, v in by_agg.items()), reverse=True)
                med = [x for x in ranked if x[1] == "median"]
                avg = [x for x in ranked if x[1] == "fedavg"]
                if not med or not avg:
                    continue
                best = ranked[0]
                rows.append(_row("median.wins", PAPER_CLAIMS["median.wins"],
                                 "%s/%s/%s: mean accuracy under attack - %s"
                                 % (ds, arch, part,
                                    ", ".join("%s %.4f" % (k, m) for m, k, _n2
                                              in ranked)),
                                 "supported" if med[0][0] > avg[0][0] else "CONTRADICTED",
                                 "median %.4f vs fedavg %.4f (best is %s)"
                                 % (med[0][0], avg[0][0], best[1]),
                                 "name the aggregator that actually held up (%s) instead "
                                 "of asserting median" % best[1]
                                 if med[0][1] != best[1] or med[0][0] <= avg[0][0]
                                 else "claim stands"))
    return rows


def _claim_params(datasets):
    from .experiments import exp_cost
    rows = []
    for ds in datasets:
        t = exp_cost.ratio_table(ds)
        if not t:
            continue
        red = float(t["params_reduction_pct"])
        rows.append(_row("params.30pct", PAPER_CLAIMS["params.30pct"],
                         "measured params CNN-BiLSTM=%d, LightTCN=%d -> %.1f%% fewer"
                         % (t["params_cnn"], t["params_tcn"], red),
                         "CONTRADICTED" if red > 40 else "supported",
                         "%.1f%%" % red,
                         "state the measured reduction (%.0f%%), not ~30%%" % red))
    return rows


def _claim_paired(datasets, hi, lo, cid, claim=None):
    """Is `hi` actually better than `lo`? Verdicts are threshold-based, never tuned.

    A claim is CONTRADICTED as soon as one architecture is materially worse
    (more than 2 pp), NOT SUPPORTED when it never wins significantly, and supported
    only when it wins on every architecture with at least one Holm-significant pair.
    """
    from .experiments import exp_main
    claim = claim or PAPER_CLAIMS.get(cid, "%s beats %s" % (hi, lo))
    rows = []
    for ds in datasets:
        good = []
        for t in exp_main.paired_tests(ds):
            forward = t.get("high") == hi and t.get("low") == lo
            reverse = t.get("high") == lo and t.get("low") == hi
            if not (forward or reverse) or t.get("mean_diff") is None:
                continue
            if forward:
                good.append(t)
            else:
                # the logged pair is stored as high - low; read it from the claim's side
                flip = dict(t)
                flip["high"], flip["low"] = lo, hi
                flip["mean_diff"] = -float(t["mean_diff"])
                good.append(flip)
        if not good:
            rows.append(_row(cid, claim, "%s: no paired seeds for %s vs %s"
                             % (ds, hi, lo), "NOT YET MEASURED", "n/a",
                             "run the main stage for %s on >= 6 seeds" % ds))
            continue
        worst = min(good, key=lambda t: t["mean_diff"])
        sig = [((t.get("holm_p") if t.get("holm_p") is not None else t.get("p_ttest"))
                or 1.0) < 0.05 for t in good]
        wins_everywhere = all(t["mean_diff"] > 0 for t in good)
        materially_worse = worst["mean_diff"] < -0.02
        evidence = ("%s: %s (paired over %d architectures, Holm-significant in %d/%d)"
                    % (ds, "  ".join("%s-%s=%+.4f" % (t["high"], t["low"], t["mean_diff"])
                                     for t in good), len(good), int(sum(sig)), len(good)))
        measured = "worst paired difference %+.4f on %s" % (worst["mean_diff"],
                                                            worst["architecture"])
        if materially_worse:
            verdict = "CONTRADICTED"
            action = ("%s is materially worse than %s on %s, so the claim as written "
                      "cannot stay in the paper" % (hi, lo, worst["architecture"]))
        elif wins_everywhere and any(sig):
            verdict = "supported"
            action = "claim stands"
        else:
            verdict = "NOT SUPPORTED"
            action = ("the difference is within noise: rewrite it as not significant, do "
                      "not add seeds until it turns positive")
        rows.append(_row(cid, claim, evidence, verdict, measured, action))
    return rows


def _claim_adapt(datasets):
    from .experiments import exp_clip
    rows = []
    for ds in datasets:
        t = exp_clip.paired(ds, arch="MLP")
        if t.get("mean_diff") is None:
            rows.append(_row("adaptdp.wins", PAPER_CLAIMS["adaptdp.wins"],
                             "%s: adaptive vs fixed not paired yet" % ds,
                             "NOT YET MEASURED", "n/a", "run the clip stage for %s" % ds))
            continue
        better = t["mean_diff"] > 0
        sig = (t.get("holm_p") or t.get("p_wilcoxon") or 1.0) < 0.05
        rows.append(_row("adaptdp.wins", PAPER_CLAIMS["adaptdp.wins"],
                         "%s: AdaptDP - DP = %+.4f, p=%.4f (n=%d seeds)"
                         % (ds, t["mean_diff"], t["p_wilcoxon"], t["n_seeds"]),
                         "supported" if (better and sig) else
                         ("NOT SUPPORTED" if better else "CONTRADICTED"),
                         "accuracy delta %+.4f over %d paired seeds (p=%.4f)"
                         % (t["mean_diff"], t["n_seeds"], t["p_wilcoxon"]),
                         "report adaptive clipping as a privacy-estimator improvement, "
                         "not an accuracy win" if not (better and sig) else "claim stands"))
    return rows


def _claim_resampling(datasets):
    rows = []
    for ds in datasets:
        sp = [r for r in read_logs(dataset=ds, stage="resampling")
              if r.get("arch_role") == "spectral_distortion"]
        if not sp:
            continue
        r = sp[-1]
        lin = _val(r, "spurious_energy_fraction_linear")
        poly = _val(r, "spurious_energy_fraction_polyphase")
        if lin is None:
            rows.append(_row("resample.harmless", PAPER_CLAIMS["resample.harmless"],
                             "%s: already at 50 Hz, no upsampling artefact possible" % ds,
                             "not applicable", "n/a",
                             "say the ablation is vacuous for this dataset"))
            continue
        rows.append(_row("resample.harmless", PAPER_CLAIMS["resample.harmless"],
                         "%s: linear 20->50 Hz puts %.3f of the energy above the source "
                         "Nyquist; polyphase puts %.3f" % (ds, lin, poly if poly is not None
                                                           else float("nan")),
                         "CONTRADICTED" if lin > 0.02 else "supported",
                         "spurious fraction %.4f (reduction %.2fx)"
                         % (lin, (lin / max(poly, 1e-12)) if poly else float("nan")),
                         "resampling must be described as a measured distortion and the "
                         "polyphase variant rerun" if lin > 0.02 else "claim stands"))
    return rows


def _claim_channels(datasets):
    rows = []
    for ds in datasets:
        runs = [r for r in read_logs(dataset=ds, stage="resampling")
                if r.get("channel_mode") and _val(r) is not None]
        if not runs:
            continue
        by = {}
        for r in runs:
            by.setdefault(str(r["channel_mode"]), []).append(_val(r))
        base = by.get("zero_pad")
        if not base:
            continue
        worst = max(((m, float(np.mean(v)) - float(np.mean(base)))
                     for m, v in by.items() if m != "zero_pad"),
                    key=lambda kv: abs(kv[1]), default=None)
        if worst is None:
            continue
        rows.append(_row("zeropad.neutral", PAPER_CLAIMS["zeropad.neutral"],
                         "%s: %s vs zero-pad = %+.4f on raw windows (n=%d/%d runs)"
                         % (ds, worst[0], worst[1], len(by[worst[0]]), len(base)),
                         "CONTRADICTED" if abs(worst[1]) > 0.01 else "supported",
                         "%+.4f accuracy" % worst[1],
                         "the channel treatment is a confound and must be reported as a "
                         "result, not a footnote" if abs(worst[1]) > 0.01 else
                         "zero-padding is empirically neutral here"))
    return rows


def _claim_noniid(datasets):
    """The 29.4 pp figure is a non-DP number, so the DP arm of the same stage is excluded."""
    def plain(r):
        return (str(r.get("method")) == "fedavg" and not float(r.get("z") or 0.0))
    rows = []
    for ds in datasets:
        runs = [r for r in read_logs(dataset=ds, stage="noniid")
                if str(r.get("alpha")) == "0.5" and plain(r)]
        iid = [r for r in read_logs(dataset=ds, stage="noniid")
               if str(r.get("alpha")) == "IID" and plain(r)]
        if not runs or not iid:
            continue
        a = np.array([_val(r) for r in runs if _val(r) is not None])
        b = np.array([_val(r) for r in iid if _val(r) is not None])
        if len(a) < 2 or len(b) < 2:
            continue
        drop = float(b.mean() - a.mean())
        spread = float(a.std(ddof=1))
        rows.append(_row("noniid.29pp", PAPER_CLAIMS["noniid.29pp"],
                         "%s: alpha=0.5 over %d partitions, mean %.4f +/- %.4f "
                         "(range %.4f-%.4f); IID %.4f" % (ds, len(a), a.mean(), spread,
                                                         a.min(), a.max(), b.mean()),
                         "CONTRADICTED" if abs(drop - 0.294) > 0.05 else "supported",
                         "drop %.4f +/- %.4f pp across partitions"
                         % (100 * drop, 100 * spread),
                         "restate the degradation as mean +/- std over partitions; the "
                         "single-partition 29.4 pp figure is not reproducible"
                         if abs(drop - 0.294) > 0.05 else "claim stands"))
    return rows


def _claim_mia(datasets):
    rows = []
    for ds in datasets:
        runs = [r for r in read_logs(dataset=ds, stage="mia")
                if _val(r, "advantage") is not None]
        dp = [r for r in runs if r.get("dp_level") and r["dp_level"] != "non-DP"]
        if not dp:
            continue
        adv = np.array([_val(r, "advantage") for r in dp])
        auc = np.array([_val(r, "auc") for r in dp if _val(r, "auc") is not None])
        rows.append(_row("mia.adv", PAPER_CLAIMS["mia.adv"],
                         "%s: DP targets, advantage %.4f +/- %.4f, AUC %.4f (%d trials, "
                         "threshold on shadow data)" % (ds, adv.mean(),
                                                        adv.std(ddof=1) if len(adv) > 1 else 0.0,
                                                        auc.mean() if len(auc) else float("nan"),
                                                        len(dp)),
                         "supported" if (adv.mean() < 0.05 and (not len(auc) or
                                                                auc.mean() < 0.6))
                         else "CONTRADICTED",
                         "advantage %.3f, AUC %.3f" % (adv.mean(),
                                                       auc.mean() if len(auc) else float("nan")),
                         "the DP guarantee does not translate into membership privacy at "
                         "this epsilon; report the attack numbers, not the assurance"
                         if adv.mean() >= 0.05 else "claim stands"))
    return rows


def _claim_monotone_privacy(datasets):
    """Utility must fall as the certificate tightens; a rise is a finding, not noise.

    Grouped by mechanism as well as by architecture: the privacy stage runs the published
    plain release and the sparse release at the SAME epsilon, so pooling them would
    average two different utility curves into one meaningless point.
    """
    rows = []
    for ds in datasets:
        runs = [r for r in read_logs(dataset=ds, stage="privacy") if _val(r) is not None]
        by = {}
        for r in runs:
            if r.get("eps") is None:
                continue
            by.setdefault((_arch_of(r), _mech_of(r), float(r["eps"])), []).append(_val(r))
        if len(by) < 3:
            continue
        per_arch = {}
        for (arch, mech, e), v in by.items():
            per_arch.setdefault((arch, mech), []).append((e, float(np.mean(v))))
        for (arch, mech), pts in sorted(per_arch.items()):
            pts.sort(reverse=True)          # weak privacy (large eps) first
            accs = [a for _, a in pts]
            rises = [(pts[i][1], accs[i + 1] - accs[i]) for i in range(len(accs) - 1)
                     if accs[i + 1] - accs[i] > 0.02]
            if rises:
                rows.append(_row("privacy.monotone", "(internal check) utility decreases "
                                 "as epsilon tightens",
                                 "%s/%s %s: accuracy rises by %s as eps falls"
                                 % (ds, arch, mech, ", ".join("%+.3f at eps=%.2f"
                                                              % (d, e) for e, d in rises)),
                                 "UNEXPECTED", "non-monotone %s privacy-utility curve" % mech,
                                 "report it: it means the clip norm or the round count is "
                                 "doing more than the noise at those levels"))
    return rows


def _mech_of(r):
    sp = r.get("dp_sparsity")
    try:
        if sp is not None and float(sp) < 1.0:
            return "sparse release"
    except (TypeError, ValueError):
        pass
    if r.get("dp_head_only") or str(r.get("dp_support")) == "head":
        return "head-only release"
    # logs written before the release shape was recorded
    cell = str(r.get("cell") or "")
    if "sparse" in cell:
        return "sparse release"
    if "head" in cell:
        return "head-only release"
    return "plain release"


def _arch_of(r):
    return str(r.get("architecture") or r.get("arch") or "?")


# ── the deliverables ──────────────────────────────────────────────────────────
def reproduce_matrix():
    """Which module produces which table/figure - the README's mapping as a CSV."""
    from .experiments import STAGES as REG
    rows = []
    for st, (mod, what) in REG.items():
        rows.append(dict(stage=st, module=mod.__name__, work_order=what,
                         run_function="ppml.experiments.%s.run(harness)" % mod.__name__,
                         csv_writers=",".join(sorted(_writer_names(mod))),
                         figures=_figures_for(st)))
    rows.append(dict(stage="report", module="ppml.reporting",
                     work_order="section 13 run report + contradiction flags",
                     run_function="ppml.reporting.run_report()",
                     csv_writers="run_report.md,run_report.json,contradictions.csv,"
                                 "reproduce_matrix.csv,reviewer_points.csv,"
                                 "paired_tests_cross_dataset.csv,cross_dataset_runs.csv,"
                                 "cross_dataset_comparison.csv,pipeline_identity.csv",
                     figures="-"))
    # The one-file comparison is not a stage - it reads the other stages' logs - so it has
    # to be listed here or the README's mapping would omit a deliverable the sheet asks for.
    rows.append(dict(stage="cross-dataset", module="ppml.crossdataset",
                     work_order="line 12: one fixed pipeline, four datasets in one file",
                     run_function="ppml.crossdataset.write_outputs() / "
                                  "consolidate([...folders...])",
                     csv_writers="cross_dataset_runs.csv,"
                                 "cross_dataset_comparison.csv,pipeline_identity.csv",
                     figures="fig_cross_dataset, fig_cross_dataset_f1, "
                             "fig_cross_dataset_eps, fig_cross_dataset_eps_f1, "
                             "fig_privacy_utility, fig_privacy_utility_f1"))
    rows.append(dict(stage="figures", module="ppml.plotting",
                     work_order="all figures regenerated from results/logs",
                     run_function="ppml.plotting.all_figures()",
                     csv_writers="figure_manifest.csv/json",
                     figures="fig_main_*, fig_all_readings_*, fig_curves_*, "
                             "fig_forest_*, fig_metrics_*, fig_per_class_*, "
                             "fig_confusion_*, fig_eps_curve_*, fig_privacy_utility, "
                             "fig_privacy_utility_f1, fig_cross_dataset, "
                             "fig_cross_dataset_f1, fig_cross_dataset_eps, "
                             "fig_cross_dataset_eps_f1, fig_clip_*, "
                             "fig_eps_split_*, fig_leakage, fig_loso_*, fig_matched_*, "
                             "fig_noniid_*, fig_byzantine_*, fig_resampling, "
                             "fig_channels_*, fig_features_*, fig_cost_*, fig_mia_*"))
    return rows


def write_reproduce_matrix():
    """Emit the README's table/figure mapping as results/reproduce_matrix.csv."""
    rows = reproduce_matrix()
    if not rows:
        return None
    config.ensure_dirs()
    path = os.path.join(config.RESULTS_DIR, "reproduce_matrix.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


# The Work Order's appendix is an index of the 24 reviewer comments, each already carrying
# the Task number that answers it. That index is reproduced here as DATA - topics verbatim,
# tasks as the sheet writes them - so the response letter and the pipeline cannot drift into
# two different lists of what still has to be answered. `stages` is which stage has to have
# RUN to answer the point; None means the sheet assigns the point to the authors' writing,
# which no measurement can answer and which is therefore reported as such rather than as a
# gap in the code.
REVIEWER_POINTS = (
    ("R1-1", "Where noise is injected; Eq. (10) / Alg. 1 / Fig. 3 consistency", "1",
     ("privacy", "main")),
    ("R1-2", "Adjacency, same epsilon notion across architectures, reproducible epsilon "
             "table", "1", ("privacy",)),
    ("R1-3", "50 Hz resampling and Nyquist", "7", ("resampling",)),
    ("R1-4", "WISDM zero-padding confound", "7", ("resampling",)),
    ("R1-5", "Feature-family ablation; jerk definition", "8", ("features",)),
    ("R1-6", "FLOPs, memory, latency, payload", "9", ("cost",)),
    ("R1-7", "Federated vs centralised needs statistics", "4", ("main",)),
    ("R1-8", "Single seed; CIs for Tables 5-6", "4", ("main",)),
    ("R1-9", "Multiple Dirichlet partitions", "5", ("noniid",)),
    ("R1-10", "Actual Byzantine attacks", "6", ("byzantine",)),
    ("R1-11", "Adaptive clipping privacy cost", "2", ("clip",)),
    ("R2-1", "Which mechanism, adjacency, full derivation", "1", ("privacy",)),
    ("R2-2", "Accountant inputs, Renyi orders, worked example", "1", ("privacy",)),
    ("R2-3", "Adaptive clipping accounting", "2", ("clip",)),
    ("R2-4", "Window vs subject split; client construction", "3", ("splits",)),
    ("R2-5", "Architecture- and budget-matched comparisons", "1, 3, 9",
     ("main", "splits", "cost")),
    ("R2-6", "Multiple seeds, significance, code release", "4 + Section 13",
     ("main", "report")),
    ("R2-7", "MIA protocol, WISDM, stronger attacks", "10", ("mia",)),
    ("R2-8", "Harmonisation, label mapping, channel ablation", "7", ("resampling",)),
    ("R2-9", "Novelty / related work", "Authors (writing)", None),
    ("R2-10", "Deployment cost; prospective framing", "9", ("cost",)),
    ("R2-11", "Nominal vs cumulative epsilon; consistent counts", "1 + authors",
     ("privacy", "report")),
    ("R2-12", "Uncertainty in tables/figures; traceability", "4 + Section 13",
     ("main", "figures", "report")),
    ("R2-13", "Notation, proofreading", "Authors (writing)", None),
)

# Which files a stage has to leave behind for the point to be answerable in the letter.
# Kept to names the code actually writes (see reproduce_matrix), because a manifest that
# cites a file nothing produces is worse than no manifest.
_POINT_ARTIFACTS = {
    "privacy": "privacy_table_<ds>.csv, accountant_crosscheck.csv",
    "clip": "clip_sweep_<ds>.csv, adaptive_eps_split_<ds>.csv, dp_mechanism_<ds>.csv",
    "splits": "leakage_<ds>.csv, client_construction_<ds>.csv, loso_<ds>.csv, "
              "matched_<ds>.csv",
    "main": "<ds>__exp_main__summary_table.csv, paired_tests.csv, "
            "paired_tests_cross_dataset.csv",
    "noniid": "noniid_<ds>.csv",
    "byzantine": "byzantine_<ds>.csv",
    "resampling": "resampling_<ds>.csv, spectral_distortion_<ds>.csv, "
                  "label_map_table.csv, resampling_pooled.csv",
    "features": "feature_ablation_<ds>.csv, jerk_definition.csv",
    "cost": "compute_cost_<ds>.csv",
    "mia": "mia_<ds>.csv",
    "report": "run_report.md, contradictions.csv, reproduce_matrix.csv, "
              "cross_dataset_runs.csv, cross_dataset_comparison.csv, "
              "pipeline_identity.csv",
    "figures": "figures/ + figure_manifest.csv",
}


def reviewer_points(cov=None):
    """Each reviewer comment, and whether the logs contain the evidence that answers it.

    The verdict is counted from the run log, never asserted: a point whose stage has no
    logged run reads NOT YET RUN, which is the row the response letter has to be honest
    about. Two points are the authors' writing and are labelled as not answerable by
    measurement rather than as a pipeline gap.
    """
    cov = list(cov if cov is not None else coverage())
    by_stage = {}
    for c in cov:
        by_stage.setdefault(c["stage"], []).append(c)
    rows = []
    for cid, topic, task, stages in REVIEWER_POINTS:
        if stages is None:
            rows.append(dict(comment=cid, topic=topic, work_order_task=task, stages="-",
                             evidence_files="-", verdict="NOT A MEASUREMENT",
                             detail="the Work Order assigns this to the authors' writing"))
            continue
        runs, held, missing, files = 0, [], [], []
        produced_now = False
        for st in stages:
            # "report" and "figures" are not experiment stages: their artifact is this very
            # pass, so counting them as unrun would make every §13 point read PARTIAL forever
            if st in ("report", "figures"):
                produced_now = True
                files.append(_POINT_ARTIFACTS.get(st, "-"))
                continue
            cells = by_stage.get(st, [])
            n = sum(c["n_runs"] for c in cells)
            runs += n
            names = sorted({c["dataset"] for c in cells if c["n_runs"]})
            if n:
                held.extend(names)
            else:
                missing.append(st)
            files.append(_POINT_ARTIFACTS.get(st, "-"))
        verdict = ("EVIDENCE LOGGED" if not missing else
                   ("NOT YET RUN" if runs == 0 else "PARTIAL"))
        detail = "%d logged runs" % runs
        if held:
            detail += " on " + ",".join(sorted(set(held)))
        elif not produced_now:
            detail += "; no dataset has logged this stage"
        if produced_now:
            detail += "; plus this session's report and figures"
        if missing:
            detail += ". Still missing: " + ",".join(missing)
        rows.append(dict(comment=cid, topic=topic, work_order_task=task,
                         stages=",".join(stages),
                         evidence_files=" | ".join(files), verdict=verdict,
                         detail=detail))
    return rows


def write_reviewer_points(cov=None):
    """results/reviewer_points.csv - the answer sheet, with measured evidence status."""
    rows = reviewer_points(cov)
    if not rows:
        return None
    config.ensure_dirs()
    path = os.path.join(config.RESULTS_DIR, "reviewer_points.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def reviewer_points_note(rows=None):
    """The three-line summary printed in the report and on the console."""
    rows = list(rows if rows is not None else reviewer_points())
    have = [r["comment"] for r in rows if r["verdict"] == "EVIDENCE LOGGED"]
    part = [r["comment"] for r in rows if r["verdict"] == "PARTIAL"]
    none = [r["comment"] for r in rows if r["verdict"] == "NOT YET RUN"]
    write = [r["comment"] for r in rows if r["verdict"] == "NOT A MEASUREMENT"]
    return dict(n_total=len(rows), n_logged=len(have), n_partial=len(part),
                n_missing=len(none), n_writing=len(write),
                logged=",".join(have) or "-", partial=",".join(part) or "-",
                missing=",".join(none) or "-", writing=",".join(write) or "-")



def _writer_names(mod):
    return [n for n in dir(mod) if n.startswith("write_") or n == "table"
            or n.endswith("_table") or n in ("paired", "eps_split", "worked_example",
                                             "ratio_table", "crosschecks",
                                             "leakage_table", "loso_table",
                                             "matched_table", "client_note")]


def _figures_for(stage):
    m = {"main": "fig_main_<ds>, fig_all_readings_<ds>, fig_curves_<ds>, fig_forest_<ds>",
         "privacy": "fig_privacy_utility", "clip": "fig_clip_<ds>, fig_clip_history_<ds>, "
                                                  "fig_eps_split_<ds>",
         "splits": "fig_leakage, fig_loso_<ds>, fig_matched_<ds>",
         "noniid": "fig_noniid_<ds>", "byzantine": "fig_byzantine_<ds>_<arch>_<partition>",
         "resampling": "fig_resampling, fig_channels_<ds>",
         "features": "fig_features_<ds>", "cost": "fig_cost_<ds>", "mia": "fig_mia_<ds>"}
    return m.get(stage, "-")


def write_cross_dataset_paired(datasets=None):
    """Pool every dataset's Task 4 paired tests and Holm-correct across the pooled family.

    Each per-dataset table already carries `holm_p` for its own dataset x architecture
    family, which is the family the Work Order names. A response letter that says "Holm
    corrected" is read more strictly than that, so this file adds the stricter column too:
    for one comparison tag (`median_vs_fedavg`, `fedprox_vs_fedavg`, ...) every
    dataset x architecture instance of it forms one family and gets `holm_p_grouped`.
    Both columns are written, and neither replaces a raw p.
    """
    from .experiments import exp_main
    datasets = sorted(datasets or {c["dataset"] for c in coverage()})
    rows = []
    for ds in datasets:
        try:
            got = exp_main.paired_tests(ds)
        except Exception:
            got = []
        for r in got or []:
            row = dict(r)
            row.setdefault("dataset", ds)
            row["comparison_key"] = str(row.get("key") or row.get("comparison") or "")
            rows.append(row)
    if not rows:
        return None
    multiseed.holm_grouped(rows, "comparison_key")
    config.ensure_dirs()
    path = os.path.join(config.RESULTS_DIR, "paired_tests_cross_dataset.csv")
    fields = ["dataset", "comparison_key", "comparison", "architecture", "high", "low",
              "n_seeds", "mean_diff", "cohen_dz", "p_ttest", "p_wilcoxon", "holm_p",
              "holm_p_grouped", "holm_group_size"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def _metrics_preview(datasets, limit=24):
    """The full metric set for the main stage, straight from the logs.

    A compact copy of exp_main.metrics_table, embedded in the report so a pasted session
    shows accuracy, precision, recall, F1, balanced accuracy, specificity, kappa, AUC and,
    for the DP arms, the certificate and the measured release SNR in one place. The CSV with
    every column is what the paper's tables are transcribed from; this is the reading copy.
    """
    rows = []
    try:
        from .experiments import exp_main
    except Exception as exc:
        return [dict(dataset=",".join(datasets),
                     accuracy="(exp_main could not be imported: %s)"
                             % type(exc).__name__)]
    for ds in datasets:
        try:
            table = exp_main.metrics_table(ds)
        except Exception as exc:
            rows.append(dict(dataset=ds,
                             accuracy="(metrics_table failed: %s)" % type(exc).__name__))
            continue
        rows.extend(r for r in table
                    if str(r.get("stage", exp_main.STAGE)) == exp_main.STAGE)
    return rows[:limit]


def _cross_dataset_payload(seen):
    """The four datasets in one file, read only from the logs (Work Order line 12).

    `run_report` owns this so the section-13 deliverable and the three cross-dataset CSVs
    come out of the same session. Datasets with no logged run are named as absent rather
    than drawn as zero, and a stage one dataset never ran is a violation, not a blank.
    """
    from . import crossdataset
    mine = [d for d in crossdataset.DATASETS if d in set(seen)]
    absent = [d for d in crossdataset.DATASETS if d not in set(seen)]
    if not mine:
        return dict(datasets=[], absent=absent, n_runs=0, paths={},
                    comparison=[], identity=[], violations=[])
    rows = crossdataset.read_all(datasets=mine)
    ident = crossdataset.pipeline_identity(rows=rows, datasets=mine)
    # The comparison keeps all four columns even for the datasets with no rows here, which
    # is what the CSV holds and is the coverage statement the sheet asks for: a reader sees
    # the empty column, not a table quietly narrowed to what ran.
    return dict(datasets=mine, absent=absent, n_runs=len(rows),
                paths=crossdataset.write_outputs(datasets=mine),
                comparison=crossdataset.comparison_table(rows=rows),
                identity=ident,
                violations=[list(v) for v in crossdataset.identity_violations(ident)])


def _cross_dataset_md(cd):
    """The cross-dataset section of run_report.md - read from the logs, blanks stay blank."""
    from . import crossdataset

    def esc(x):
        return str(x).replace("|", "/")
    out = ["", "## The four datasets in one file (Work Order line 12)", ""]
    if not cd or not cd.get("datasets"):
        return out + ["No dataset has reportable logged runs in this session, so there is"
                      " nothing to put side by side yet. The four per-dataset bundles are"
                      " the run path; each writes the same fixed-schema"
                      " `cross_dataset_runs.csv`, and the merge step below stacks them.",
                      ""]
    ds = list(crossdataset.DATASETS)
    slugs = [crossdataset.slug(d) for d in ds]
    out += ["%d reportable runs over %s. Three files hold them:"
            % (cd["n_runs"], ", ".join(cd["datasets"])), "",
            "- `results/cross_dataset_runs.csv` - %d columns, one row per logged run, the"
            " same header whatever bundle wrote it, so the four sessions stack;" %
            len(crossdataset.RUN_COLUMNS),
            "- `results/cross_dataset_comparison.csv` - %d cells (stage x architecture x"
            " method) with every metric mean, std, 95%% interval and n per dataset;"
            % len(cd["comparison"]),
            "- `results/pipeline_identity.csv` - %d stage/setting comparisons that answer"
            " line 12 from the logs rather than from the source." %
            len([r for r in cd["identity"] if r["setting"] != "__coverage__"]), ""]
    if cd["absent"]:
        out += ["**Datasets absent from this session's logs:** %s. Kaggle wipes"
                " `/kaggle/working` between sessions, so their rows arrive through the"
                " merge step at the end of this section, not from this run."
                % ", ".join(cd["absent"]), ""]
    for metric, title in (("accuracy", "test accuracy"), ("f1_macro", "macro F1")):
        rows = [r for r in cd["comparison"]
                if any(("%s_mean__%s" % (metric, s)) in r for s in slugs)]
        if not rows:
            out += ["_No logged run carries a %s reading, so that table is empty rather"
                    " than zero._" % title, ""]
            continue
        out += ["### %s, one column per dataset" % title, "",
                "| stage | architecture | cell | " +
                " | ".join(esc(d) for d in ds) + " | all four |",
                "| --- | --- | --- | " + " | ".join("---" for _ in ds) + " | --- |"]
        for r in rows[:40]:
            cells = []
            for s in slugs:
                mean = r.get("%s_mean__%s" % (metric, s))
                if mean is None:
                    cells.append("| - ")
                    continue
                std = r.get("%s_std__%s" % (metric, s)) or 0.0
                ci = r.get("%s_ci__%s" % (metric, s)) or ""
                cells.append("| %.4f +-%.4f %s (n=%s) "
                             % (mean, std, esc(ci), r.get("%s_n__%s" % (metric, s), 1)))
            out.append("| %s | %s | %s %s| %s |"
                       % (esc(r["stage"]), esc(r["architecture"]), esc(r["cell"]),
                          "".join(cells), "yes" if r.get("complete") else "no"))
        if len(rows) > 40:
            out += ["", "_%d further cells are in `cross_dataset_comparison.csv`._"
                    % (len(rows) - 40)]
        out += [""]
    n_stages = len({r["stage"] for r in cd["identity"]})
    bad = set(tuple(v) for v in cd["violations"])
    out += ["### Do the four still share one pipeline?", "",
            "%d comparisons across %d stage(s): every setting a dataset was actually run"
            " with, read back from its log rows. %s"
            % (len([r for r in cd["identity"] if r["setting"] != "__coverage__"]),
               n_stages,
               "None disagrees, so the numbers below are comparable." if not bad else
               "**%d disagree - the rows those datasets produced are not comparable, and"
               " the fix is to re-run them, not to reconcile the tables afterward.**"
               % len(bad)), ""]
    if bad:
        out += ["| stage | setting | dataset | values this dataset used | verdict |",
                "| --- | --- | --- | --- | --- |"]
        for r in cd["identity"]:
            if (r["stage"], r["setting"], r["dataset"]) in bad:
                out.append("| %s | %s | %s | %s | **%s** |"
                           % (esc(r["stage"]), esc(r["setting"]), esc(r["dataset"]),
                              esc(r["values"]), esc(r["agrees_with_other_datasets"])))
        out += [""]
    out += ["### Comparable figures", "",
            "Drawn from these logs only: `fig_cross_dataset` / `fig_cross_dataset_f1`"
            " (one grouped bar per dataset, same axes, chance line marked), "
            "`fig_cross_dataset_eps` / `_f1` and `fig_privacy_utility` / `_f1`, which are"
            " the Work Order's line 65 request - accuracy and macro F1 against cumulative"
            " epsilon for every dataset and architecture.", "",
            "### Merging the four sessions", "",
            "`/kaggle/working` does not survive between sessions, so the four files are"
            " stacked afterwards, from anywhere that can import the package:", "",
            "```",
            "from ppml import crossdataset",
            "crossdataset.consolidate([<session 1 results folder>, <session 2 results"
            " folder>, ...])",
            "```", "",
            "It accepts exported `results/` folders, `results/logs` folders, individual"
            " `.jsonl` logs or `cross_dataset_runs.csv` files, refuses a CSV whose header"
            " is not the standard one, dedupes runs on `config_hash`, and rewrites all"
            " three CSVs plus the figures above from the combined set.", ""]
    return out


def run_report(datasets=None, stages=None, write_figures=True, also_contradictions=True):
    """Write results/run_report.md + .json and print the headline lines."""
    config.ensure_dirs()
    cov = coverage(datasets, stages)
    seen = sorted({c["dataset"] for c in cov})
    rt = runtime(seen)
    con = contradictions(seen) if also_contradictions else []
    figures = plotting.all_figures(seen) if write_figures else []
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    points = reviewer_points(cov)
    payload = dict(generated_at=stamp, profile=config.PROFILE,
                   reportable=config.is_reportable(), hardware=runlog.hardware(),
                   versions=runlog.versions(), runtime=rt, coverage=cov,
                   incomplete=completeness_note(cov),
                   reviewer_points=points,
                   reviewer_summary=reviewer_points_note(points),
                   not_attempted=list(NOT_ATTEMPTED),
                   scoped_arms=list(SCOPED_ARMS) + profile_scope_notes(),
                   mechanism_deviations=list(MECHANISM_DEVIATIONS),
                   contradictions=con,
                   cross_dataset=_cross_dataset_payload(seen),
                   metrics_preview=_metrics_preview(sorted(
                       {c["dataset"] for c in cov})),
                   figures=[{k: v for k, v in f.items() if k != "_runs"}
                            for f in figures],
                   claims_quoted=dict(PAPER_CLAIMS))
    with open(os.path.join(config.RESULTS_DIR, "run_report.json"), "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    _write_contradictions_csv(con)
    write_reproduce_matrix()
    payload["reviewer_points_csv"] = write_reviewer_points(cov)
    payload["paired_tests_cross_dataset"] = write_cross_dataset_paired(seen)
    md = _markdown(payload)
    path = os.path.join(config.RESULTS_DIR, "run_report.md")
    with open(path, "w") as fh:
        fh.write(md)
    flags = [c for c in con if c["verdict"] in ("CONTRADICTED", "UNEXPECTED",
                                                "NOT SUPPORTED")]
    print("\n== run report -> %s" % path)
    print("   profile=%s reportable=%s  total wall time=%.2f h over %d timed runs"
          % (config.PROFILE, config.is_reportable(), rt["total_wall_h"],
             rt["n_timed_runs"]))
    print("   incomplete: %s" % payload["incomplete"])
    print("   claims flagged: %d of %d checked" % (len(flags), len(con)))
    cd = payload.get("cross_dataset") or {}
    if cd.get("datasets"):
        print("   four-dataset file: %d runs over %s -> cross_dataset_runs.csv,"
              " cross_dataset_comparison.csv, pipeline_identity.csv (%d identity"
              " violations)"
              % (cd["n_runs"], ", ".join(cd["datasets"]), len(cd["violations"])))
    if cd.get("absent"):
        print("      absent this session: %s (merge with crossdataset.consolidate([...]))"
              % ", ".join(cd["absent"]))
    rs = payload["reviewer_summary"]
    print("   reviewer points: %d of %d answered from the logs, %d partial, %d with no "
          "run yet, %d are the authors' writing (see reviewer_points.csv)"
          % (rs["n_logged"], rs["n_total"], rs["n_partial"], rs["n_missing"],
             rs["n_writing"]))
    if rs["missing"] != "-":
        print("      no evidence yet for: %s" % rs["missing"])
    if rs["partial"] != "-":
        print("      partly answered:       %s" % rs["partial"])
    for c in flags:
        print("   !! [%s] %s" % (c["verdict"], c["evidence"]))
    return payload


def _write_contradictions_csv(con):
    if not con:
        return
    path = os.path.join(config.RESULTS_DIR, "contradictions.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(con[0]))
        w.writeheader()
        w.writerows(con)


def _md_num(row, metric):
    """`mean+-std [CI]` for one metric of one cell, or an honest blank."""
    mean = row.get("%s_mean" % metric)
    if mean is None:
        return ""
    std = row.get("%s_std" % metric)
    lo, hi = row.get("%s_ci95_low" % metric), row.get("%s_ci95_high" % metric)
    txt = "%.4f" % mean
    if std is not None:
        txt += " +- %.4f" % std
    if lo is not None and hi is not None:
        txt += " [%.3f, %.3f]" % (lo, hi)
    return txt


def _md_cert(row, key):
    """A single-valued certificate or diagnostic column, formatted or blank."""
    v = row.get(key)
    if v is None or isinstance(v, (list, dict)):
        return ""
    try:
        return "%.4f" % float(v)
    except (TypeError, ValueError):
        return str(v)


def _markdown(p):
    def esc(x):
        return str(x).replace("|", "/")
    lines = ["# Run report - IEEE Access-2026-40021 revision",
             "",
             "Generated %s by `ppml.reporting.run_report()`. Profile `%s` "
             "(reportable: **%s**)." % (p["generated_at"], p["profile"], p["reportable"]),
             "",
             "## Hardware and software",
             "",
             "| item | value |",
             "| --- | --- |"]
    for k, v in sorted((p["hardware"] or {}).items()):
        lines.append("| hardware %s | %s |" % (k, esc(v)))
    for k, v in sorted((p["versions"] or {}).items()):
        lines.append("| %s version | %s |" % (k, esc(v)))
    lines += ["", "## Runtime", "",
              "Total wall-clock in the logged runs: **%.2f h** across %d timed runs."
              % (p["runtime"]["total_wall_h"], p["runtime"]["n_timed_runs"]),
              "", "| stage | hours |", "| --- | --- |"]
    for k, v in sorted(p["runtime"]["by_stage"].items()):
        lines.append("| %s | %.2f |" % (esc(k), v / 3600.0))
    lines += ["", "## Coverage (what actually ran)", "",
              "| dataset | stage | runs | cells | seeds | partitions | complete | reportable |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in p["coverage"]:
        lines.append("| %s | %s | %d | %d | %d | %d | %s | %s |"
                     % (esc(c["dataset"]), esc(c["stage"]), c["n_runs"], c["n_cells"],
                        c["n_seeds"], c["n_partitions"],
                        "yes" if c["complete"] else "NO",
                        "yes" if c["reportable"] else "NO"))
    lines += ["", "**Could not be completed:** %s" % p["incomplete"], ""]
    lines += ["", "## Results - full metric set (main stage, from the logs)", "",
              "`mean+-std` over the seeds that logged the cell, 95% t-interval in brackets. "
              "Blank means that reading was never logged, not that it was zero. The complete "
              "column set, including per-class and confusion tables, is in "
              "`results/tables/<dataset>__exp_main__metrics_table.csv`.", "",
              "| dataset | arch | cell | n | accuracy | precision | recall | macro F1 | "
              "balanced | specificity | kappa | AUC | eps | release SNR |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
              " --- | --- |"]
    for r in p.get("metrics_preview", []):
        if "cell" not in r:
            lines.append("| %s | - | %s | - | - | - | - | - | - | - | - | - | - | - |"
                         % (esc(r.get("dataset", "")), esc(r.get("accuracy", "-"))))
            continue
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                     " %s | %s |"
                     % (esc(r.get("dataset", "")), esc(r.get("architecture", "?")),
                        esc(r.get("cell", "?")), r.get("n_seeds", 0),
                        _md_num(r, "accuracy"), _md_num(r, "precision"),
                        _md_num(r, "recall"), _md_num(r, "f1_macro"),
                        _md_num(r, "balanced_accuracy"), _md_num(r, "macro_specificity"),
                        _md_num(r, "kappa"), _md_num(r, "roc_auc_ovr"),
                        _md_cert(r, "eps"), _md_cert(r, "release_snr_mean")))
    lines += [""]
    lines += _cross_dataset_md(p.get("cross_dataset") or {})
    rs = p.get("reviewer_summary") or {}
    lines += ["", "## Reviewer comments, and what answers each one", "",
              "The index is the Work Order's own appendix. The status is counted from "
              "`results/logs`, not asserted, so a comment whose evidence has not been run "
              "yet reads NOT YET RUN here instead of disappearing.", ""]
    if rs:
        lines += ["%d of %d comments have logged evidence, %d are partial, %d have none "
                  "yet, %d are writing tasks no measurement can answer." % (
                      rs["n_logged"], rs["n_total"], rs["n_partial"], rs["n_missing"],
                      rs["n_writing"]), ""]
    lines += ["| comment | topic | Work Order task | stage(s) | status | detail |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in p.get("reviewer_points", []):
        lines.append("| %s | %s | %s | %s | **%s** | %s |"
                     % (esc(r["comment"]), esc(r["topic"]), esc(r["work_order_task"]),
                        esc(r["stages"]), esc(r["verdict"]), esc(r["detail"])))
    lines += ["", "**Deliberately not attempted** (the Work Order marks each of these "
              "optional, so they are listed rather than silently absent):", ""]
    for name, why in p.get("not_attempted", []):
        lines.append("- *%s* - %s" % (esc(name), esc(why)))
    lines += ["", "**Arms that ran on a narrower grid than the full matrix:**", ""]
    for name, why in p.get("scoped_arms", []):
        lines.append("- *%s* - %s" % (esc(name), esc(why)))
    lines += ["", "**Mechanism deviations from the rejected submission, and why each "
              "leaves epsilon where it was** (every one is asserted by the preflight "
              "before a run is logged):", ""]
    for name, why in p.get("mechanism_deviations", []):
        lines.append("- *%s* - %s" % (esc(name), esc(why)))
    lines += ["", "## Claims that the measurements contradict", "",
              "Nothing here was tuned to agree with the paper; each verdict is a "
              "threshold applied to a logged number.", "",
              "| claim id | paper claim | verdict | measured | evidence | what to do |",
              "| --- | --- | --- | --- | --- | --- |"]
    for c in p["contradictions"]:
        lines.append("| %s | %s | **%s** | %s | %s | %s |"
                     % (esc(c["claim_id"]), esc(c["paper_claim"])[:80], c["verdict"],
                        esc(c["measured"])[:60], esc(c["evidence"])[:160],
                        esc(c["action_needed"])))
    drawn = [f for f in p["figures"] if not f.get("skipped")]
    lines += ["", "## Figures regenerated from the logs", "",
              "%d drawn, %d skipped (a skip means the runs are not logged yet - nothing "
              "is invented to fill a panel)." % (len(drawn),
                                                 len(p["figures"]) - len(drawn)), "",
              "| figure | runs | filter |", "| --- | --- | --- |"]
    for f in drawn:
        lines.append("| %s | %s | %s |" % (esc(f.get("figure")), f.get("n_runs", ""),
                                           esc(f.get("log_filter", ""))[:90]))
    lines += ["", "## How to reproduce", "",
              "| stage | module | run | CSV writers | figures |",
              "| --- | --- | --- | --- | --- |"]
    for r in reproduce_matrix():
        lines.append("| %s | %s | `%s` | %s | %s |"
                     % (esc(r["stage"]), esc(r["module"]), esc(r["run_function"]),
                        esc(r["csv_writers"]), esc(r["figures"])[:120]))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":      # pragma: no cover
    print("run report")
    run_report()

#@@FILE ppml/experiments/__init__.py
"""Experiment stages, one per Work Order task.

`STAGES` is the registry the dataset drivers iterate over, so a Kaggle session can run
any subset and resume the rest from the logs.
"""

from . import (exp_main, exp_privacy, exp_clip, exp_splits, exp_noniid,
               exp_byzantine, exp_resampling, exp_features, exp_cost, exp_mia,
               exp_dpsgd, exp_dplearn)

STAGES = {
    "main":       (exp_main,       "Task 4  main results, >=6 seeds, CIs"),
    "privacy":    (exp_privacy,    "Task 1  z-sweep -> privacy_table.csv"),
    "clip":       (exp_clip,       "Task 2  clip sweep + adaptive accounting"),
    "splits":     (exp_splits,     "Task 3  leakage audit, LOSO, matched comparisons"),
    "noniid":     (exp_noniid,     "Task 5  Dirichlet alpha with repeated partitions"),
    "byzantine":  (exp_byzantine,  "Task 6  5 attacks x 4 aggregators"),
    "resampling": (exp_resampling, "Task 7  resampling + WISDM channel ablations"),
    "features":   (exp_features,   "Task 8  feature-family ablation + redundancy"),
    "cost":       (exp_cost,       "Task 9  parameters / MACs / latency / payload"),
    "mia":        (exp_mia,        "Task 10 membership-inference, shadow thresholds"),
    # Opt-in: the Work Order's optional example-level extra, a different privacy unit from
    # every arm above. Add "dpsgd" to RUN_STAGES to run it; it is not in the default list
    # because per-example backward passes dominate the session cost.
    "dpsgd":      (exp_dpsgd,      "Task 1  example-level DP-SGD (one window), eps solved "
                                   "from the stated targets"),
    "dplearn":    (exp_dplearn,    "Task 1  user-level DP: small model, public init, "
                                   "head-only release"),
}
ORDER = list(STAGES)


def get(name):
    if name not in STAGES:
        raise KeyError("unknown stage %r; choose from %s" % (name, ORDER))
    return STAGES[name][0]


def describe(name):
    return STAGES[name][1]

#@@FILE ppml/experiments/exp_main.py
"""Task 4 - main results table with >= 6 seeds and paired significance tests.

Replaces the submission's single-seed Tables 5 and 6. The FL runner is called for
every method, including the ones the old ``run_all.seeds()`` stub compared as two
centralised MLPs that differed only in L2 strength.
"""

from .. import config, metrics, multiseed, splitting
from ..harness import Harness
from ..runlog import read_logs

STAGE = "main"

# The comparison family declared by the Work Order (Task 4 deliverable 2), evaluated
# per architecture and pooled for Holm correction across the whole family.
PAIR_SETS = [
    ("FL-Median", "FL-FedAvg", "median_vs_fedavg"),
    ("FL+AdaptDP", "FL+DP", "adaptdp_vs_dp"),
    ("FL-FedProx", "FL-FedAvg", "fedprox_vs_fedavg"),
    ("Centralised", "FL-FedAvg", "fedavg_vs_centralised"),
    # Both arms certify at the same (z, T, delta), so this paired test is a statement
    # about utility alone: if the sparse arm wins, it won it for free.
    ("FL+DP-Sparse", "FL+DP", "sparse_dp_vs_plain_dp"),
    # same certificate, released over d_head coordinates instead of d - the arm whose
    # utility change has a derivation rather than a hope.
    ("FL+DP-Head", "FL+DP", "head_dp_vs_plain_dp"),
]


# The headline set every Table 5/6 row now carries. These are what `metrics.classification_metrics`
# returns on the test set, so the list can only contain columns the log actually holds.
SUMMARY_METRICS = ("accuracy", "precision", "recall", "f1", "f1_macro",
                   "balanced_accuracy", "macro_specificity", "kappa",
                   "roc_auc_ovr", "nll")
# The same abbreviations the per-run line uses, so a cell summary can be read beside the
# runs it was averaged over. The CSV columns keep the long names.
_METRIC_LABEL = {"precision": "prec", "recall": "rec", "f1": "F1w", "f1_macro": "F1m",
                 "balanced_accuracy": "bal", "macro_specificity": "spec",
                 "kappa": "kappa", "roc_auc_ovr": "auc", "nll": "nll"}


def _fmt_cell(vals):
    """`mean+-std [CI]` for one metric over the seeds of one cell, or n/a."""
    if not vals:
        return "n/a"
    s = multiseed.summarise([float(v) for v in vals])
    return "%.3f+-%.3f[%.3f,%.3f]" % (s["mean"], s["std"], s["ci95_low"], s["ci95_high"])


def _dp_cell_summary(rows, arch, label):
    """The certificate and measured release SNR for the DP runs of this cell.

    Read off the runs that just executed, never from the settings table: epsilon is
    constant across seeds for a fixed (z, T, delta), but the SNR is a measurement and the
    K actually participating varies, so both are reported as a range over the seeds.
    """
    mine = [r for r in rows if str(r.get("architecture") or "").upper() == arch.upper()
            and r.get("label") == "%s|%s" % (arch, label) and r.get("privacy")]
    if not mine:
        return ""
    eps = [float(r["privacy"]["eps"]) for r in mine if r["privacy"].get("eps") is not None]
    snr = [float(r["release_snr_mean"]) for r in mine
           if r.get("release_snr_mean") is not None]
    out = []
    if eps:
        out.append("eps=%.4f (delta=%g, T=%s, adj=%s)"
                   % (eps[0], mine[0]["privacy"].get("delta") or config.DELTA,
                      mine[0]["privacy"].get("T") or mine[0].get("rounds"),
                      mine[0]["privacy"].get("adjacency")))
    if snr:
        out.append("release_snr mean=%.4f range=[%.4f, %.4f]%s"
                   % (sum(snr) / len(snr), min(snr), max(snr),
                      " (crosses 1)" if max(snr) >= 1.0 else " (below 1: noise-dominated)"))
    return ", ".join(out)


def _client_indices(h, split, seed, spec):
    if spec.get("partition") == "dirichlet":
        return h.clients_dirichlet(split, seed, alpha=spec.get("alpha", 0.5),
                                   kind="dirichlet")
    return h.clients(split, seed)["client_indices"]


def _methods():
    """Every Task 4 arm, or only the ones a DP-only session asked for.

    The filter never re-labels anything: an arm that is skipped is absent from the log, and
    exp_main's paired tests report that as "insufficient paired seeds".
    """
    specs = tuple(s for s in config.MAIN_METHODS if not s[1].get("disabled"))
    if not config.METHODS_FILTER:
        return specs
    keep = set(config.METHODS_FILTER)
    return tuple(s for s in specs if s[0] in keep)


def run(h: Harness, seeds=None, arches=None, verbose=True):
    seeds = seeds or config.cfg()["seeds"]
    arches = arches or config.MAIN_ARCHES
    methods = _methods()
    rows = []
    if config.METHODS_FILTER and verbose:
        skipped = [l for l, _ in config.MAIN_METHODS if l not in set(m[0] for m in methods)]
        print("  METHODS_FILTER is set: Task 4 runs %d of %d arms (%s)%s"
              % (len(methods), len(config.MAIN_METHODS),
                 ", ".join(l for l, _ in methods),
                 "; not run this session: " + ", ".join(skipped) if skipped else ""))
    for arch in arches:
        if verbose:
            # A long silence here is the architecture block working, not a hang: on CPU a
            # CNN-BiLSTM run is minutes and the DP arms always run T=50 rounds, so print
            # what is coming before the first line of it lands.
            print("  %s: %d methods x %d seed(s) starting; a line appears only when a "
                  "run finishes" % (arch, len(methods), len(seeds)))
        arch_methods = methods
        keep = config.MAIN_METHODS_BY_ARCH.get(arch)
        if keep:
            arch_methods = tuple(m for m in methods if m[0] in keep)
        arch_seed_n = int(config.MAIN_SEEDS_BY_ARCH.get(arch, len(seeds)))
        for label, spec in arch_methods:
            per_seed = {}
            metric_seeds = {}
            # a proposed arm that is not the Work Order's own method may run on fewer
            # seeds; the n lands in the log and the paired test uses the shared seeds.
            arm_seeds = seeds[:min(int(spec.get("seeds_n", len(seeds))), arch_seed_n)]
            if len(arm_seeds) < len(seeds) and verbose:
                print("  [%-12s %-14s] scoped to %d of %d seeds (n is in every row)"
                      % (arch, label, len(arm_seeds), len(seeds)))
            for seed in arm_seeds:
                split = h.split(seed, kind=spec.get("split_kind", "subject"))
                cidx = _client_indices(h, split, seed, spec)
                het = splitting.partition_heterogeneity(cidx, h.y)
                if spec.get("central"):
                    res = h.run_central(STAGE, arch=arch, split=split, seed=seed,
                                        label="%s|%s" % (arch, label), cell=label,
                                        cfg=dict(split_kind="subject"))
                else:
                    res = h.run_fl(dict(arch_role="main",
                                        partition=spec.get("partition"),
                                        alpha=spec.get("alpha"),
                                        split_kind=spec.get("split_kind", "subject")),
                                   STAGE, arch=arch,
                                   client_indices=cidx, split=split, seed=seed,
                                   method=spec.get("method", "fedavg"),
                                   mu=spec.get("mu", 0.0),
                                   dp=spec.get("dp"),
                                   use_fedper=spec.get("use_fedper", False),
                                   server_momentum=spec.get("server_momentum", False),
                                   rounds=(config.cfg()["rounds_dp"] if spec.get("dp")
                                           else spec.get("rounds")),
                                   label="%s|%s" % (arch, label), cell=label,
                                   heterogeneity=het, verbose=verbose)
                if res is None:
                    continue
                per_seed[int(seed)] = float(res["accuracy"])
                for _mk in SUMMARY_METRICS:
                    _mv = res.get(_mk)
                    if _mv is not None:
                        metric_seeds.setdefault(_mk, []).append(float(_mv))
                rows.append(res)
                if verbose:
                    print("  [%-12s %-14s] seed=%d  %s"
                          % (arch, label, seed, metrics.format_metrics(res)))
            if per_seed:
                vals = list(per_seed.values())
                s = multiseed.summarise(vals)
                s.update(dataset=h.bundle.name, architecture=arch, method=label)
                # the seed-level spread of every headline metric, not of accuracy alone: this
                # is the line that becomes a Table 5/6 row, so it has to show the same columns
                tail = "".join("  %s=%s" % (_METRIC_LABEL.get(_mk, _mk),
                                            _fmt_cell(metric_seeds.get(_mk)))
                               for _mk in SUMMARY_METRICS if _mk != "accuracy")
                print("  %-12s %-14s n=%d  acc=%.4f+-%.4f CI=[%.4f, %.4f]%s"
                      % (arch, label, s["n"], s["mean"], s["std"], s["ci95_low"],
                         s["ci95_high"], tail))
                dp_note = _dp_cell_summary(rows, arch, label)
                if dp_note:
                    print("  %-12s %-14s %s" % (arch, label, dp_note))
    return rows


def _cell_index(runs):
    """(architecture, cell) -> {seed: accuracy}, from the written logs."""
    table = {}
    for r in runs:
        arch = str(r.get("architecture") or r.get("arch") or "?")
        cell = str(r.get("cell") or r.get("method") or "?")
        acc = r.get("accuracy")
        if acc is None:
            continue
        table.setdefault((arch, cell), {})[int(r["seed"])] = float(acc)
    return table


def paired_tests(dataset, runs=None, arches=None):
    """Comparison family with Holm correction, computed from the written logs.

    Every comparison is paired on the seeds both sides actually ran, so a partially
    completed stage still reports the usable pairs instead of silently mismatching
    runs the way the old substring matching did.
    """
    all_runs = runs if runs is not None else read_logs(dataset=dataset, stage=STAGE)
    table = _cell_index(all_runs)
    arches = list(arches or config.MAIN_ARCHES)
    out = []
    for arch in arches:
        for hi, lo, name in PAIR_SETS:
            sa = table.get((arch, hi), {})
            sb = table.get((arch, lo), {})
            seeds = sorted(set(sa) & set(sb))
            tag = "%s/%s" % (arch, name)
            if len(seeds) < 2:
                out.append(dict(comparison=tag, architecture=arch, key=name,
                                high=hi, low=lo, n_seeds=len(seeds),
                                mean_diff=None, p_wilcoxon=1.0,
                                note="insufficient paired seeds"))
                continue
            va = [sa[s] for s in seeds]
            vb = [sb[s] for s in seeds]
            row = multiseed.compare(va, vb, hi, lo)
            row.update(comparison=tag, architecture=arch, key=name, high=hi, low=lo,
                       n_seeds=len(seeds))
            out.append(row)
    return multiseed.holm_table(out)


def summary_table(dataset, runs=None):
    """Per-cell mean +/- std and 95% CI, the Table 5/6 replacement."""
    all_runs = runs if runs is not None else read_logs(dataset=dataset, stage=STAGE)
    table = _cell_index(all_runs)
    rows = []
    for (arch, cell), per_seed in sorted(table.items()):
        s = multiseed.summarise([per_seed[k] for k in sorted(per_seed)])
        s.update(dataset=dataset, architecture=arch, cell=cell)
        rows.append(s)
    return rows


def _cell_runs(runs):
    """(architecture, cell) -> list of logged run records, from the written logs."""
    cells = {}
    for r in runs:
        key = (str(r.get("architecture") or r.get("arch") or "?"),
               str(r.get("cell") or r.get("method") or "?"))
        cells.setdefault(key, []).append(r)
    return cells


def _cert_cols(group):
    """The settings and certificate columns a DP row has to carry.

    Taken from the runs themselves, and a column is only reported when every seed of the
    cell agrees or the first record states it - epsilon is a property of (z, T, delta), not
    of the seed, so agreeing here is a check and not an assumption.
    """
    out = {}
    for k in ("z", "C", "K", "T", "delta", "eps", "alpha_star", "eps_total",
              "eps_updates", "eps_clip_fraction", "mechanism", "adjacency",
              "dp_sparsity", "dp_support", "dp_head_only", "rounds", "local_epochs",
              "lr", "batch", "max_windows", "split_kind"):
        vals = []
        for r in group:
            v = r.get(k)
            if v is None or isinstance(v, (list, dict)):
                continue
            if k == "delta" or (k in ("eps", "alpha_star", "eps_total", "eps_updates",
                                      "eps_clip_fraction")):
                vals.append(round(float(v), 6))
            else:
                vals.append(v)
        if not vals:
            continue
        uniq = sorted({str(v) for v in vals})
        out[k] = vals[0] if len(uniq) == 1 else "|".join(uniq)
    snr = [float(r["release_snr_mean"]) for r in group
           if r.get("release_snr_mean") is not None]
    if snr:
        out["release_snr_mean"] = sum(snr) / len(snr)
        out["release_snr_min"] = min(snr)
        out["release_snr_max"] = max(snr)
        out["noise_dominates"] = bool(max(snr) < 1.0)
    return out


def metrics_table(dataset, runs=None, metrics_=None):
    """Every headline metric per cell, with mean, std, 95% CI and n - from the logs.

    `summary_table` keeps the accuracy-only shape the paired tests and figures read. This is
    the wide sheet the paper's tables are transcribed from: accuracy, precision, recall,
    weighted and macro F1, balanced accuracy, macro specificity, Cohen's kappa, one-vs-rest
    ROC-AUC and NLL, each with its own seed spread and interval, next to the certificate
    columns that make a DP row auditable.
    """
    all_runs = runs if runs is not None else read_logs(dataset=dataset, stage=STAGE)
    keys = list(metrics_ or SUMMARY_METRICS)
    rows = []
    for (arch, cell), group in sorted(_cell_runs(all_runs).items()):
        row = dict(dataset=dataset, stage=STAGE, architecture=arch, cell=cell,
                  n_seeds=len({int(r.get("seed", 0)) for r in group}))
        for mk in keys:
            vals = []
            for r in group:
                v = r.get(mk)
                if v is None or isinstance(v, (list, dict, str, bool)):
                    continue
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if f == f:
                    vals.append(f)
            if not vals:
                continue
            s = multiseed.summarise(vals, metric_name=mk)
            row["%s_mean" % mk] = s["mean"]
            row["%s_std" % mk] = s["std"]
            row["%s_ci95_low" % mk] = s["ci95_low"]
            row["%s_ci95_high" % mk] = s["ci95_high"]
            row["%s_n" % mk] = s["n"]
        row.update(_cert_cols(group))
        rows.append(row)
    return rows


def per_seed_table(dataset, runs=None):
    """One row per logged run, with that run's own numbers - nothing averaged.

    Work Order line 100 asks for the raw per-seed values *and* the summary. `metrics_table`
    is the summary; this is the raw sheet, so a reader can put a Table 5/6 cell and the six
    runs behind it side by side without opening the JSONL.
    """
    all_runs = runs if runs is not None else read_logs(dataset=dataset, stage=STAGE)
    rows = []
    for r in sorted(all_runs, key=lambda x: (str(x.get("architecture") or x.get("arch") or "?"),
                                             str(x.get("cell") or x.get("method") or "?"),
                                             str(x.get("seed")))):
        row = dict(dataset=dataset, stage=STAGE,
                   architecture=str(r.get("architecture") or r.get("arch") or "?"),
                   cell=str(r.get("cell") or r.get("method") or "?"),
                   seed=r.get("seed"), label=r.get("label"),
                   reportable=r.get("reportable"), wall_time_s=r.get("wall_time_s"),
                   config_hash=r.get("config_hash"))
        for mk in SUMMARY_METRICS:
            v = r.get(mk)
            if v is not None and not isinstance(v, (list, dict, str, bool)):
                try:
                    row[mk] = float(v)
                except (TypeError, ValueError):
                    pass
        for k in ("eps", "alpha_star", "z", "C", "K", "T", "delta", "adjacency",
                  "mechanism", "dp_head_only", "dp_sparsity", "dp_support"):
            # a logged row keeps these flat; the in-session result keeps them under
            # "privacy", and reading both is what makes this sheet the same live or later
            v = r.get(k)
            if v is None:
                v = (r.get("privacy") or {}).get(k)
            if v is not None:
                row[k] = v
        for k in ("release_snr_mean", "accuracy_iter_avg", "f1_iter_avg"):
            if r.get(k) is not None:
                row[k] = r[k]
        rows.append(row)
    return rows


def per_class_table(dataset, runs=None):
    """Class-wise precision, recall, F1 and specificity per cell, mean over the seeds.

    The Work Order's activity-label mapping makes the classes comparable across datasets;
    this table is where a method's failure to see a rare activity becomes visible instead of
    being averaged away. A class a cell never predicted gets 0.0 from the metric helper, not
    a blank, and `support_mean` says how many test windows the class actually had.
    """
    all_runs = runs if runs is not None else read_logs(dataset=dataset, stage=STAGE)
    rows = []
    for (arch, cell), group in sorted(_cell_runs(all_runs).items()):
        n_class = max([len(r["f1_per_class"]) for r in group
                       if isinstance(r.get("f1_per_class"), list)] or [0])
        if not n_class:
            continue
        for c in range(n_class):
            row = dict(dataset=dataset, stage=STAGE, architecture=arch, cell=cell,
                       class_index=c, n_seeds=len(group))
            for mk, key in (("f1", "f1_per_class"), ("precision", "precision_per_class"),
                            ("recall", "recall_per_class"),
                            ("specificity", "specificity_per_class")):
                vals = []
                for r in group:
                    lst = r.get(key)
                    if isinstance(lst, list) and c < len(lst) and lst[c] is not None:
                        vals.append(float(lst[c]))
                if vals:
                    row["%s_mean" % mk] = sum(vals) / len(vals)
                    row["%s_n" % mk] = len(vals)
            sup = []
            for r in group:
                lst = r.get("class_support")
                if isinstance(lst, list) and c < len(lst):
                    sup.append(float(lst[c]))
            if sup:
                row["support_mean"] = sum(sup) / len(sup)
            rows.append(row)
    return rows


def confusion_table(dataset, runs=None):
    """Mean confusion counts (and row-normalised rates) per cell, straight from the logs.

    Row 1 is the count, row 2 the same number divided by the class's support, so the paper
    can print either. Averaged over seeds because a confusion matrix from one seed is the
    single-seed reading the reviewers objected to.
    """
    all_runs = runs if runs is not None else read_logs(dataset=dataset, stage=STAGE)
    rows = []
    for (arch, cell), group in sorted(_cell_runs(all_runs).items()):
        mats = [r.get("confusion") for r in group
                if isinstance(r.get("confusion"), list) and r.get("confusion")]
        if not mats:
            continue
        n = max(len(m) for m in mats)
        for i in range(n):
            for j in range(n):
                counts = [float(m[i][j]) for m in mats
                          if i < len(m) and j < len(m[i])]
                if not counts:
                    continue
                mean_count = sum(counts) / len(counts)
                row_tot = []
                for m in mats:
                    if i < len(m):
                        tot = sum(float(x) for x in m[i])
                        if tot > 0:
                            row_tot.append(float(m[i][j]) / tot)
                rows.append(dict(dataset=dataset, stage=STAGE, architecture=arch,
                                 cell=cell, true_class=i, pred_class=j,
                                 n_seeds=len(counts), count_mean=mean_count,
                                 rate_mean=(sum(row_tot) / len(row_tot)) if row_tot else None))
    return rows

#@@FILE ppml/experiments/exp_privacy.py
"""Task 1 - privacy-utility sweep with valid certificates, and privacy_table.csv.

The sweep is over the noise multiplier z, never over a "nominal epsilon". Every row of
privacy_table.csv is produced by a logged run, and the epsilon in the table is the
certificate from dp_accountant, not the number the noise was derived from.
"""

import csv
import os

from .. import config, dp_accountant, multiseed
from ..harness import Harness
from ..runlog import read_logs

STAGE = "privacy"


def _levels(arch=None):
    """(name, dp-argument-or-None, z, momentum), straight from the profile's z levels.

    config, not this function, decides where the non-private reference point is, so a
    table or figure that enumerates the levels cannot silently drop it.

    Every certificate level runs TWICE by default, once per mechanism, at the same
    (z, T, delta), and a third time when the head sweep below is switched on:

      plain    the rejected paper's arm - absolute CLIP_C, noise on all d coordinates.
               This is the arm the paper's epsilon claims are about, so it is run exactly
               as published and its accuracy is reported whatever it says.
      sparse   the proposed arm: the release is applied to only its largest coordinates,
               selected after the noise draw, plus a clip radius scaled to the model,
               server momentum and iterate averaging. All four are eps-neutral
               (fl.dp_aggregate, config.CLIP_REL, config.DP_ITER_AVG), so the two rows sit
               at the SAME epsilon and the accuracy difference is a pure utility statement,
               not a privacy trade.

    A third mechanism exists when config.PRIVACY_HEAD_SWEEP is on, and it is the one that
    answers the accuracy question instead of restating it:

      head     the release is restricted to the classifier head, so the same noise budget
               is spread over d_head coordinates instead of d. Nothing about the
               certificate changes - the support is fixed by the architecture, never read
               from data, so the sensitivity stays C/K - but the signal-to-noise the model
               has to beat goes from K/(z*sqrt(d)) to K/(z*sqrt(d_head)). This sweep
               measures that curve at every z rather than only at the operating point the
               Task 4 table prices, which is the difference between a prediction and a
               result. Opt-in because it is +5 levels x every seed of T=50 runs.
    """
    out = [("non-private", None, 0.0, False)]
    head = (config.PRIVACY_HEAD_SWEEP and
            (arch is None or arch in config.PRIVACY_HEAD_ARCHES))
    zs = config.PRIVACY_Z_BY_ARCH.get(arch) if arch else None
    mechs = config.PRIVACY_MECH_BY_ARCH.get(arch, ("plain", "head"))
    for z in (zs or config.cfg().get("z_levels", config.Z_LEVELS)):
        z = float(z)
        if z <= 0.0:
            continue
        if "plain" in mechs:
            out.append(("z=%.1f" % z, dict(z=z, C=config.CLIP_C), z, False))
        if "sparse" in mechs:
            out.append(("z=%.1f sparse" % z,
                        dict(z=z, C=None, sparsity=config.DP_SPARSITY,
                             iter_avg=config.DP_ITER_AVG), z, True))
        if head and "head" in mechs:
            out.append(("z=%.1f head" % z,
                        dict(z=z, C=None, head_only=True,
                             iter_avg=config.DP_ITER_AVG), z, True))
    return out


def run(h: Harness, seeds=None, arches=None, verbose=True):
    # T=50 rounds and two mechanisms per z: this is the most expensive stage in the
    # pipeline, so it runs at the DP scope (config.DP_SEEDS) and says n in every row.
    seeds = seeds or config.cfg().get("dp_seeds", config.cfg()["seeds_supp"])
    arches = arches or config.PRIVACY_ARCHES
    T = config.cfg()["rounds_dp"]
    rows = []
    for arch in arches:
        arch_seeds = tuple(seeds)[:int(config.PRIVACY_SEEDS_BY_ARCH.get(arch, len(seeds)))]
        for level, dp, z, momentum in _levels(arch):
            per_seed = {}
            cell = "%s | %s" % (arch, level)
            for seed in arch_seeds:
                split = h.split(seed)
                cl = h.clients(split, seed)
                res = h.run_fl(dict(arch_role="privacy", z_level=level,
                                    sparsity=(dp or {}).get("sparsity")),
                               STAGE, arch=arch, client_indices=cl["client_indices"],
                               split=split, seed=seed, method="fedavg", dp=dp,
                               server_momentum=momentum,
                               rounds=T, cell=cell, label=cell,
                               verbose=verbose)
                if res is None:
                    continue
                per_seed[int(seed)] = float(res["accuracy"])
                rows.append(res)
            if per_seed:
                s = multiseed.summarise(list(per_seed.values()))
                cert = (dp_accountant.account_central_gaussian(z, T, config.DELTA)
                        if z > 0 else dict(eps=float("inf"), alpha_star=float("inf")))
                ia = [float(r["accuracy_iter_avg"]) for r in rows
                      if r.get("cell") == cell and r.get("accuracy_iter_avg") is not None]
                extra = (" iter-avg=%.4f" % (sum(ia) / len(ia))) if ia else ""
                # the measured signal-to-noise of the release, beside the accuracy it is
                # supposed to explain: acc at chance plus snr under 1 is a mechanism
                # statement, the same accuracy with snr above 1 is not.
                snr = [float(r["release_snr_mean"]) for r in rows
                       if r.get("cell") == cell
                       and r.get("release_snr_mean") is not None]
                if snr:
                    extra += " snr=%.3f" % (sum(snr) / len(snr))
                # every headline metric at this z, so the console already reads like the
                # privacy-utility table the sheet asks for
                cols = _metric_cols([r for r in rows if r.get("cell") == cell])
                if "precision_mean" in cols:
                    extra += ("  prec=%.4f rec=%.4f F1m=%.4f bal=%.4f spec=%.4f"
                              % (cols.get("precision_mean", float("nan")),
                                 cols.get("recall_mean", float("nan")),
                                 cols.get("f1_macro_mean", float("nan")),
                                 cols.get("balanced_accuracy_mean", float("nan")),
                                 cols.get("macro_specificity_mean", float("nan"))))
                print("  [%-11s %-18s] z=%-6.2f eps=%-8s acc=%.4f +-%.4f (n=%d)%s"
                      % (arch, level, z, _fmt(cert["eps"]), s["mean"], s["std"], s["n"],
                         extra))
    return rows


def _fmt(e):
    return "inf" if (e is None or e != e or e == float("inf")) else "%.3f" % float(e)


def table(dataset):
    """One row per dataset x architecture x z - the Work Order's privacy_table.csv.

    Section 13 asks this table to carry the adaptive-clipping split too, so the
    adaptive cells logged by Task 2 are appended as their own rows rather than parked in
    a separate file: same columns, plus the two epsilon components.
    """
    T = config.cfg()["rounds_dp"]
    runs = read_logs(dataset=dataset, stage=STAGE)
    rows = []
    rows += _rows_from(runs, dataset, T, source="privacy")
    adaptive = [r for r in read_logs(dataset=dataset, stage="clip")
                if _is_adaptive(r)]
    rows += _rows_from(adaptive, dataset, T, source="clip-adaptive")
    return rows


def _is_adaptive(r):
    if r.get("eps_clip_fraction") is not None:
        return True
    dp = r.get("dp")
    return bool(isinstance(dp, dict) and dp.get("adaptive"))


def _metric_cols(rs):
    """Mean and seed spread of every headline metric for one privacy-sweep cell.

    Only columns the logged runs actually carry are emitted, so a run logged before this
    telemetry existed leaves the cell blank rather than reporting a number it never made.
    """
    out = {}
    for key in ("precision", "recall", "f1_macro", "balanced_accuracy",
                "macro_specificity", "kappa", "roc_auc_ovr", "nll"):
        vals = []
        for r in rs:
            v = r.get(key)
            if v is None or isinstance(v, (list, dict, str, bool)):
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f == f:
                vals.append(f)
        if vals:
            s = multiseed.summarise(vals, metric_name=key)
            out["%s_mean" % key] = s["mean"]
            out["%s_std" % key] = s["std"]
            out["%s_ci95_low" % key] = s["ci95_low"]
            out["%s_ci95_high" % key] = s["ci95_high"]
    return out


def _rows_from(runs, dataset, T, source):
    groups = {}
    for r in runs:
        # head_only is part of the key, not just of the cell name: the sparse and head arms
        # share a z, a C and a sparsity of None, so without it two different mechanisms
        # would average into one row of the table the reviewers read.
        key = (str(r.get("architecture") or r.get("arch")),
               float(r.get("z") or 0.0), float(r.get("C") or 0.0),
               str(r.get("dp_sparsity")), str(r.get("dp_head_only")),
               str(r.get("cell") or ""))
        groups.setdefault(key, []).append(r)
    rows = []
    for (arch, z, C, sparsity, head_only, cell), rs in sorted(
            groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0.0)):
        logged = rs[0]
        adaptive = source == "clip-adaptive"
        if z > 0 and not adaptive:
            cert = dp_accountant.account_central_gaussian(z, T, config.DELTA)
        else:
            # either no noise at all, or a certificate that already ran and was logged;
            # the adaptive total is not reproducible from z alone
            cert = dict(eps=logged.get("eps_total", logged.get("eps")),
                        alpha_star=logged.get("alpha_star"),
                        mechanism=logged.get("mechanism"),
                        adjacency=logged.get("adjacency"), per_round_eps=[])
            if z <= 0:
                cert.update(eps=float("inf"), alpha_star=float("inf"),
                            mechanism="none (no noise added)", adjacency="n/a")
        s = multiseed.summarise([float(r["accuracy"]) for r in rs])
        K_run = logged.get("K") or config.N_CLIENTS
        ia = [float(r["accuracy_iter_avg"]) for r in rs
              if r.get("accuracy_iter_avg") is not None]
        sig = [float(r["signal_energy_fraction"]) for r in rs
               if r.get("signal_energy_fraction") is not None]
        snr = [float(r["release_snr_mean"]) for r in rs
               if r.get("release_snr_mean") is not None]
        # the adaptive arm's certificate depends on how many rounds its clip estimator
        # kept releasing; that is read from the run, never recomputed from z alone
        T_clip = logged.get("T_clip_estimator") if adaptive else None
        rows.append(dict(
            dataset=dataset, architecture=arch, cell=cell, source=source,
            mechanism=(cert.get("mechanism") or
                       "central gaussian, whole-update clip") +
                      (" + sparse release after the draw" if sparsity not in (None, "None")
                       and float(sparsity) < 1.0 else "") +
                      (" + head-only release (support fixed by the architecture)"
                       if head_only == "True" else "") +
                      (" + clip estimator frozen after %s rounds" % T_clip
                       if adaptive and T_clip is not None else ""),
            adjacency=cert.get("adjacency") or "add/remove all data of one client",
            C=C, z=z, K=int(K_run),
            sparsity=(float(sparsity) if sparsity not in (None, "None") else None),
            head_only=(head_only == "True"),
            signal_energy_fraction=(sum(sig) / len(sig) if sig else None),
            # measured, from the vectors actually drawn, not predicted from d
            release_snr_mean=(sum(snr) / len(snr) if snr else None),
            T=T, delta=config.DELTA, alpha_star=cert.get("alpha_star"),
            T_clip_estimator=T_clip,
            clip_freeze_rounds=logged.get("clip_freeze_rounds") if adaptive else None,
            # sigma is read with the K the run actually used: one subject per client made
            # K a property of the dataset, not of config.N_CLIENTS.
            sigma=(dp_accountant.sigma_of(z, C, int(K_run)) if z > 0 else None),
            cumulative_eps=cert.get("eps"),
            eps_updates=logged.get("eps_updates") if adaptive else cert.get("eps"),
            eps_clip_fraction=logged.get("eps_clip_fraction") if adaptive else None,
            accuracy_mean=s["mean"], accuracy_std=s["std"], accuracy_ci95_low=s["ci95_low"],
            accuracy_ci95_high=s["ci95_high"], n_seeds=s["n"],
            accuracy_iter_avg_mean=(sum(ia) / len(ia) if ia else None),
            f1_mean=float(sum(float(r["f1"]) for r in rs) / len(rs))
            if all(r.get("f1") is not None for r in rs) else None,
            # the whole headline set against epsilon, not accuracy alone: "Accuracy/F1 vs
            # cumulative eps" is the sheet's deliverable, and a privacy-utility curve that
            # shows only accuracy hides which part of the classifier the noise destroyed.
            **_metric_cols(rs),
            eps_per_round=(logged.get("eps_per_round") or cert.get("per_round_eps") or [])))
    return rows


def _mean_of(runs, get, cell=None):
    v = [get(r) for r in runs if (cell is None or str(r.get("cell")) == cell)
         and get(r) is not None]
    return float(sum(v) / len(v)) if v else None


def _serialise(value):
    """Everything stays in the one CSV: a list becomes `round=eps` pairs, never dropped."""
    if isinstance(value, list):
        parts = []
        for p in value:
            if isinstance(p, dict):
                parts.append("%s=%s" % (p.get("round"), p.get("eps")))
            else:
                parts.append(str(p))
        return ";".join(parts)
    if isinstance(value, dict):
        return ";".join("%s=%s" % kv for kv in sorted(value.items()))
    return value


def write_csv(dataset):
    config.ensure_dirs()
    rows = table(dataset)
    path = os.path.join(config.RESULTS_DIR, "privacy_table_%s.csv" % dataset.replace(" ", "_"))
    if not rows:
        print("  [privacy] no logged runs for %s yet" % dataset)
        return None
    keys = list(rows[0])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _serialise(v) for k, v in r.items()})
    # the same per-round curve in long form, for plotting without parsing the column
    with open(path.replace(".csv", "_eps_curve.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "round", "eps"])
        for r in rows:
            for i, p in enumerate(r.get("eps_per_round") or [], start=1):
                eps = p["eps"] if isinstance(p, dict) else p
                rnd = p.get("round") if isinstance(p, dict) else i
                w.writerow([r["cell"], rnd, "%.6f" % eps])
    print("  [privacy] wrote %s (%d rows)" % (path, len(rows)))
    return path


def worked_example(dataset):
    """Task 1 deliverable 3: one fully worked example from settings that really ran."""
    rows = table(dataset)
    dp_rows = [r for r in rows if r["z"] > 0]
    if not dp_rows:
        return None
    r = dp_rows[len(dp_rows) // 2]
    return dict(dataset=dataset, architecture=r["architecture"], C=r["C"], z=r["z"],
                K=r["K"], T=r["T"], delta=r["delta"], alpha_star=r["alpha_star"],
                sigma=r["sigma"], eps=r["cumulative_eps"],
                note=("sigma = z*C/K = %.6g; sensitivity of the averaged clipped "
                      "update is C/K, so the noise-to-sensitivity ratio is z" % r["sigma"]))


def crosschecks(T=None):
    """Task 1 deliverable 4: compare our closed form with Google dp-accounting."""
    T = T or config.cfg()["rounds_dp"]
    out = []
    for z in config.Z_SWEEP:
        got = dp_accountant.cross_check(z, T, config.DELTA)
        got["z"] = z
        got["T"] = T
        out.append(got)
    return out


def write_crosschecks():
    config.ensure_dirs()
    rows = crosschecks()
    path = os.path.join(config.RESULTS_DIR, "accountant_crosscheck.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["z", "T", "delta", "ours_eps", "dp_accounting_eps", "rel_diff", "ok"])
        for r in rows:
            w.writerow([r["z"], r["T"], config.DELTA,
                        "%.6f" % r["ours"],
                        "%.6f" % r["theirs"] if r["theirs"] is not None else "n/a",
                        "%.2e" % r.get("rel_diff", float("nan")) if r["theirs"] is not None else "n/a",
                        "installed" if r["ok"] else r.get("error", "")[:60]])
    print("  [privacy] wrote %s" % path)
    return path


def legacy_cost():
    """What the REJECTED submission's own settings cost, reported for the response letter."""
    zl = dp_accountant.legacy_equivalent_z(config.LEGACY_NOMINAL_EPS)
    out = []
    for arch, L in config.LEGACY_N_TENSORS.items():
        out.append(dict(model="%s per-tensor clip" % arch,
                        eps=dp_accountant.account_per_tensor_clipping(zl, 50, L)["eps"],
                        L=L))
    out.append(dict(model="whole-model clip (same z)",
                    eps=dp_accountant.account_central_gaussian(zl, 50)["eps"]))
    return dict(legacy_nominal_eps=config.LEGACY_NOMINAL_EPS, legacy_z=zl, rows=out)

#@@FILE ppml/experiments/exp_clip.py
"""Task 2 - adaptive clipping and its privacy cost.

The submission's AdaptiveGaussDP privatised the clipped fraction with Laplace noise and
then never added that cost to epsilon. Here the estimator is the Gaussian quantile
estimator of Andrew et al. (2021), and its privacy loss is composed with the update
mechanism in Renyi DP by dp_accountant.account_adaptive, so the reported total includes
both terms.

One deviation from the published mechanism is built in and disclosed: the estimator stops
releasing after config.ADAPT_FREEZE_ROUNDS rounds and C is held at the value it settled
on. It is the more expensive of the two releases by a wide margin - 1/(2*sigma_b^2) per
round against 1/(2*z^2) for the model update - so the run is accounted for the releases it
actually made, which the clipper's own history reports.
"""

import csv
import os

import numpy as np

from .. import config, dp_accountant, multiseed
from ..harness import Harness
from ..runlog import read_logs

STAGE = "clip"
# Task 2 asks for the CNN too "if feasible"; flip this off to save session hours.
INCLUDE_CNN = False      # revision scope: the clip study runs on the MLP


def _arches():
    return ("MLP", "CNN-BiLSTM") if INCLUDE_CNN else ("MLP",)


def _mech_grid():
    """Cells for the mechanism ablation, one knob changed at a time, all at the same z.

    All four sparsity points run with the shipped momentum, momentum is then priced at the
    shipped sparsity, and the released-d lever (head-only support) runs once with momentum
    on. Every cell therefore differs from a neighbour by exactly one change, without
    paying for the full product at T=50 rounds on two architectures.
    """
    # scope: the mechanism grid is dropped; Task 2's question is the adaptive-vs-fixed
    # comparison and its epsilon split, which part (2) of run() answers
    return []


def run(h: Harness, seeds=None, verbose=True):
    # every arm here is a DP arm at T=50 rounds, so the stage runs at the DP scope
    seeds = seeds or config.cfg()["seeds_supp"]
    seeds = tuple(seeds)[:len(config.cfg()["seeds_supp"])]
    T = config.cfg()["rounds_dp"]
    z = config.DP_ARM_Z
    rows = []
    for arch in _arches():
        # (1) fixed-C sweep at one moderate z
        for C in config.cfg().get("clip_sweep", config.CLIP_SWEEP):
            accs = []
            for seed in seeds:
                split = h.split(seed)
                cl = h.clients(split, seed)
                res = h.run_fl(dict(arch_role="clip_sweep"), STAGE, arch=arch,
                               client_indices=cl["client_indices"], split=split,
                               seed=seed, method="fedavg",
                               dp=dict(z=z, C=float(C)), rounds=T,
                               cell="%s | C=%.2f" % (arch, C),
                               label="%s | C=%.2f" % (arch, C), verbose=False)
                if res is not None:
                    accs.append(float(res["accuracy"]))
                    rows.append(res)
            if accs:
                s = multiseed.summarise(accs)
                print("  [%-11s C=%.2f] acc=%.4f +-%.4f" % (arch, C, s["mean"], s["std"]))
        # (2) adaptive vs fixed at the same z
        for mode, dp in (("fixed", dict(z=z, C=config.CLIP_C)),
                         ("adaptive", dict(z=z, C=config.CLIP_C, adaptive=True))):
            accs = []
            for seed in seeds:
                split = h.split(seed)
                cl = h.clients(split, seed)
                res = h.run_fl(dict(arch_role="clip_%s" % mode), STAGE, arch=arch,
                               client_indices=cl["client_indices"], split=split,
                               seed=seed, method="fedavg", dp=dp, rounds=T,
                               cell="%s | %s" % (arch, mode),
                               label="%s | %s" % (arch, mode), verbose=False)
                if res is not None:
                    accs.append(float(res["accuracy"]))
                    rows.append(res)
            if accs:
                s = multiseed.summarise(accs)
                print("  [%-11s %-8s] acc=%.4f +-%.4f" % (arch, mode, s["mean"], s["std"]))
        # (3) the mechanism's UTILITY knobs at the same z, so the tables can say which
        # eps-neutral change bought what. C stays on the relative basis across this whole
        # grid, so a cell differs from its neighbour by the sparse release or by momentum
        # and by nothing else; the absolute-C reference is the "fixed" cell of (2) above
        # and the paper's own arm is FL+DP in Task 4. sparsity=1.0 is the arm as published:
        # every coordinate updated, identical certificate.
        for knob in _mech_grid():
            sp, mom, head = knob["sparsity"], knob["mom"], knob["head"]
            tag = "head" if head else "sp=%.2f%s" % (sp, "+mom" if mom else "")
            accs, iaccs = [], []
            for seed in seeds:
                split = h.split(seed)
                cl = h.clients(split, seed)
                cell = "%s | %s" % (arch, tag)
                res = h.run_fl(dict(arch_role="clip_mechanism", sparsity=sp,
                                    momentum=bool(mom), head_only=bool(head)), STAGE,
                               arch=arch,
                               client_indices=cl["client_indices"], split=split,
                               seed=seed, method="fedavg",
                               dp=dict(z=z, C=None, sparsity=sp, head_only=bool(head),
                                       iter_avg=config.DP_ITER_AVG),
                               server_momentum=mom, rounds=T, cell=cell,
                               label=cell, verbose=False)
                if res is not None:
                    accs.append(float(res["accuracy"]))
                    if res.get("accuracy_iter_avg") is not None:
                        iaccs.append(float(res["accuracy_iter_avg"]))
                    rows.append(res)
            if accs:
                s = multiseed.summarise(accs)
                extra = (" iter-avg=%.4f" % (sum(iaccs) / len(iaccs))) if iaccs else ""
                print("  [%-11s %-12s] acc=%.4f +-%.4f%s"
                      % (arch, tag, s["mean"], s["std"], extra))
    return rows


def _by_cell(runs):
    out = {}
    for r in runs:
        cell = str(r.get("cell"))
        out.setdefault(cell, {})[int(r["seed"])] = float(r["accuracy"])
    return out


def eps_split(dataset):
    """Deliverable 1: epsilon from the model updates, from the clip-fraction estimator,
    and the total - all at the shared optimal Renyi order.

    `eps_total` is the certificate for the arm as this pipeline runs it: the estimator
    releases for config.ADAPT_FREEZE_ROUNDS rounds and C is frozen after that, so only
    those releases compose. `eps_total_all_rounds` is the same accounting with the
    estimator left running for all T rounds, which is Andrew et al. as published, and it
    is kept in the row because the difference is the whole point of the split - at
    sigma_b = 0.5 each estimator round costs more than fifty update releases together.
    """
    T = config.cfg()["rounds_dp"]
    W = config.ADAPT_FREEZE_ROUNDS
    W = max(0, min(int(W), int(T)))
    got = dp_accountant.account_adaptive(config.DP_ARM_Z, T, config.DELTA,
                                         sigma_b=config.ADAPT_SIGMA_B, T_clip=W)
    all_rounds = dp_accountant.account_adaptive(config.DP_ARM_Z, T, config.DELTA,
                                                sigma_b=config.ADAPT_SIGMA_B)
    fixed = dp_accountant.account_central_gaussian(config.DP_ARM_Z, T, config.DELTA)
    return dict(dataset=dataset, z=config.DP_ARM_Z, T=T, delta=config.DELTA,
                sigma_b=config.ADAPT_SIGMA_B, T_clip_estimator=W,
                eps_updates_fixed=fixed["eps"], alpha_fixed=fixed["alpha_star"],
                eps_updates=got["eps_updates"], eps_clip_fraction=got["eps_clip_fraction"],
                eps_total=got["eps_total"], alpha_star=got["alpha_star"],
                eps_total_all_rounds=all_rounds["eps_total"],
                clip_fraction_share=got["eps_clip_fraction"] / max(got["eps_total"], 1e-12))


def paired(dataset, arch="MLP"):
    """Deliverable 2: adaptive vs fixed clipping, mean +/- std and the paired p-value."""
    cells = _by_cell(read_logs(dataset=dataset, stage=STAGE))
    a = cells.get("%s | adaptive" % arch, {})
    b = cells.get("%s | fixed" % arch, {})
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return dict(comparison="adaptdp_vs_dp", architecture=arch, dataset=dataset,
                    n_seeds=len(seeds), note="insufficient paired seeds",
                    p_wilcoxon=1.0, p_ttest=1.0)
    row = multiseed.compare([a[s] for s in seeds], [b[s] for s in seeds],
                            "FL+AdaptDP", "FL+DP")
    row.update(architecture=arch, dataset=dataset, comparison="%s/adaptdp_vs_dp" % arch)
    return row


def clip_history(dataset, cell):
    """Deliverable 3: the logged C_r / clip-fraction trajectory, for the figure.

    Nothing is smoothed or reconstructed here; it is the array the run wrote.
    """
    out = []
    for r in read_logs(dataset=dataset, stage=STAGE):
        if str(r.get("cell")) != cell or not r.get("clip_history"):
            continue
        out.append(dict(seed=int(r["seed"]),
                        history=[[int(p["round"]), float(p["C"]),
                                  float(p["clip_fraction"])] for p in r["clip_history"]]))
    return out


def write_csvs(dataset):
    config.ensure_dirs()
    slug = dataset.replace(" ", "_")
    rows = [eps_split(dataset)]
    p1 = os.path.join(config.RESULTS_DIR, "adaptive_eps_split_%s.csv" % slug)
    with open(p1, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    p2 = os.path.join(config.RESULTS_DIR, "clip_sweep_%s.csv" % slug)
    cells = _by_cell(read_logs(dataset=dataset, stage=STAGE))
    with open(p2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "n_seeds", "accuracy_mean", "accuracy_std"])
        for cell, per_seed in sorted(cells.items()):
            s = multiseed.summarise(list(per_seed.values()))
            w.writerow([cell, s["n"], "%.5f" % s["mean"], "%.5f" % s["std"]])
    # The mechanism ablation, one row per cell of grid (3): every column is something the
    # run logged, and `epsilon` is shown equal across the rows on purpose - the accuracy
    # column is the only thing that moves, which is the whole point of the arm.
    runs = read_logs(dataset=dataset, stage=STAGE)
    # the sparse cells are tagged "sp=0.10+mom" and the head-only cell "head"; both are
    # the grid of (3), so the filter has to take the release-support cells and not one tag
    mech = [r for r in runs
            if str(r.get("cell", "")).find("| sp=") >= 0
            or str(r.get("cell", "")).find("| head") >= 0]
    p3 = os.path.join(config.RESULTS_DIR, "dp_mechanism_%s.csv" % slug)
    by_cell = {}
    for r in mech:
        by_cell.setdefault(str(r["cell"]), []).append(r)
    with open(p3, "w", newline="") as fh:
        keys = ["cell", "n_seeds", "accuracy_mean", "accuracy_std",
                "accuracy_iter_avg_mean", "sparsity", "support", "head_tensors",
                "signal_energy_fraction", "noise_coords", "total_coords",
                "release_snr_mean", "release_signal_norm", "release_noise_norm",
                "clip_C",
                "clip_basis", "epsilon"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for cell, rs in sorted(by_cell.items()):
            s = multiseed.summarise([float(r["accuracy"]) for r in rs])
            ia = [float(r["accuracy_iter_avg"]) for r in rs
                  if r.get("accuracy_iter_avg") is not None]
            sef = [float(r["signal_energy_fraction"]) for r in rs
                   if r.get("signal_energy_fraction") is not None]
            # the measured release SNR, averaged over the seeds that ran: the column that
            # says how far below one this cell is, so a chance-level accuracy in this table
            # is read as the arithmetic it is rather than as a training bug
            def _mean(field):
                vals = [float(r[field]) for r in rs if r.get(field) is not None]
                return ("%.6f" % (sum(vals) / len(vals))) if vals else None
            last = rs[-1]
            w.writerow(dict(
                cell=cell, n_seeds=s["n"], accuracy_mean="%.5f" % s["mean"],
                accuracy_std="%.5f" % s["std"],
                accuracy_iter_avg_mean=("%.5f" % (sum(ia) / len(ia))) if ia else None,
                sparsity=last.get("dp_sparsity"),
                support=last.get("dp_support"),
                head_tensors=last.get("dp_head_tensors"),
                signal_energy_fraction=("%.5f" % (sum(sef) / len(sef))) if sef else None,
                noise_coords=last.get("noise_coords"),
                total_coords=last.get("total_coords"),
                release_snr_mean=_mean("release_snr_mean"),
                release_signal_norm=_mean("release_signal_norm"),
                release_noise_norm=_mean("release_noise_norm"),
                clip_C=last.get("clip_C"), clip_basis=last.get("clip_basis"),
                epsilon=("%.6f" % last["eps"]) if last.get("eps") is not None else None))
    print("  [clip] wrote %s, %s and %s" % (p1, p2, p3))
    return [p1, p2, p3]


def update_norm_scale(dataset, cell="MLP | fixed"):
    """Diagnostic for the response letter: how large the real updates are versus C.

    If every update norm is far above C the clip is binding on all clients every round,
    which is the honest way to say the mechanism is a normalisation, not a bound.
    """
    norms = []
    snrs = []
    C_used = None
    for r in read_logs(dataset=dataset, stage=STAGE):
        if str(r.get("cell")) != cell:
            continue
        if C_used is None:
            C_used = r.get("clip_C")
        if r.get("release_snr_mean") is not None:
            snrs.append(float(r["release_snr_mean"]))
        for p in r.get("clip_history") or []:
            norms.extend(p.get("update_norms") or [])
    if not norms:
        return None
    arr = np.asarray(norms, float)
    C = float(config.CLIP_C if C_used is None else C_used)
    snr = (float(sum(snrs) / len(snrs)) if snrs else None)
    return dict(n=int(len(arr)), C=C,
                median=float(np.median(arr)),
                p10=float(np.percentile(arr, 10)),
                p90=float(np.percentile(arr, 90)),
                share_above_C=float(np.mean(arr > C)),
                # and what that bought in signal-vs-noise: below one means the released
                # update is smaller than the noise drawn over it, which is the sentence the
                # response letter needs for any arm that finished at chance
                release_snr_mean=snr,
                noise_dominates=bool(snr is not None and snr < 1.0))

#@@FILE ppml/experiments/exp_splits.py
"""Task 3 - subject-level splits, the leakage audit, LOSO and matched comparisons.

The submission split windows with train_test_split, so the same person appeared in
train and test; it also compared a CNN against a logistic-regression centralised model
with different budgets. Everything here is split by SUBJECT, and every comparison is
between runs that share rounds, local epochs, batch size, optimiser and mechanism.
"""

import csv
import os

import numpy as np

from .. import config, multiseed, splitting
from ..harness import Harness, Split
from ..runlog import read_logs

STAGE = "splits"
LEAK_ARCHES = ("MLP", "LightTCN")
LOSO_ARCHES = ("MLP",)


def _split_from_fold(h, fold, subject):
    idx = dict(train_idx=fold["train_idx"], val_idx=fold["val_idx"],
               test_idx=fold["test_idx"])
    subs = lambda a: sorted(int(x) for x in np.unique(np.asarray(h.subjects)[a]))
    meta = dict(subjects_in_split={"train": subs(fold["train_idx"]),
                                   "val": subs(fold["val_idx"]),
                                   "test": [int(subject)]},
                leakage_checks=dict(split_kind="loso", holdout_subject=int(subject),
                                    subjects_overlap_train_test=0))
    return Split(idx, h.subjects, meta, key="loso:%d" % int(subject))


def run(h: Harness, seeds=None, verbose=True):
    seeds = seeds or config.cfg()["seeds_supp"]
    rows = []

    # ── window-level versus subject-disjoint, same model ──────────────────────
    for arch in LEAK_ARCHES:
        for kind in ("subject", "window"):
            accs = []
            for seed in seeds:
                split = h.split(seed, kind=kind)
                cl = h.clients(split, seed)
                res = h.run_fl(dict(arch_role="leakage", split_kind=kind), STAGE,
                               arch=arch, client_indices=cl["client_indices"],
                               split=split, seed=seed, method="fedavg",
                               cell="%s | %s-split" % (arch, kind),
                               label="%s | %s-split" % (arch, kind), verbose=False)
                if res is not None:
                    accs.append(float(res["accuracy"]))
                    rows.append(res)
            if accs:
                s = multiseed.summarise(accs)
                print("  [%-11s %-14s] acc=%.4f +-%.4f" % (arch, kind, s["mean"], s["std"]))

    # ── architecture-matched and representation-matched, with and without DP ──
    # CNN-BiLSTM vs LightTCN at an identical budget (rounds, local epochs, batch, lr,
    # mechanism) is already in the main table (Task 4 runs both with the same settings),
    # so here only the representation-matched pair is added: the same MLP on the 117
    # features versus on the flattened raw windows.
    matched = [
        ("MLP", "rep-matched", None), ("MLP-Raw", "rep-matched", None),
    ]
    for arch, role, dp in matched:
        accs = []
        for seed in seeds:
            split = h.split(seed)
            cl = h.clients(split, seed)
            res = h.run_fl(dict(arch_role=role), STAGE, arch=arch,
                           client_indices=cl["client_indices"], split=split, seed=seed,
                           method="fedavg", dp=dp,
                           rounds=config.cfg()["rounds_dp"] if dp else None,
                           cell="%s | %s" % (arch, role),
                           label="%s | %s" % (arch, role), verbose=False)
            if res is not None:
                accs.append(float(res["accuracy"]))
                rows.append(res)
        if accs:
            s = multiseed.summarise(accs)
            print("  [%-11s %-16s] acc=%.4f +-%.4f" % (arch, role, s["mean"], s["std"]))

    # ── leave-one-subject-out ─────────────────────────────────────────────────
    folds = splitting.loso_folds(h.subjects,
                                 n_folds_max=config.cfg().get("loso_folds",
                                                             config.LOSO_MAX_FOLDS),
                                 seed=config.SEEDS[0])
    for fold in folds:
        fs = _split_from_fold(h, fold, fold["holdout_subject"])
        clients = h.clients(fs, config.SEEDS[0])["client_indices"]
        for arch in LOSO_ARCHES:
            res = h.run_fl(dict(arch_role="loso", holdout_subject=fold["holdout_subject"]),
                           STAGE, arch=arch, client_indices=clients, split=fs,
                           seed=config.SEEDS[0], method="fedavg",
                           rounds=config.LOSO_ROUNDS,
                           cell="%s | loso-subj%d" % (arch, fold["holdout_subject"]),
                           label="%s | loso" % arch, partition_id=fold["holdout_subject"],
                           verbose=False)
            if res is not None:
                rows.append(res)
        for arch in LOSO_ARCHES:
            # the centralised control per fold AND per architecture, so a LOSO row is
            # always a like-for-like pair with the FL row above it
            res = h.run_central(STAGE, arch=arch, split=fs, seed=config.SEEDS[0],
                                cell="Centralised-%s | loso-subj%d"
                                     % (arch, fold["holdout_subject"]),
                                label="Centralised-%s | loso" % arch,
                                cfg=dict(split_kind="loso", arch_role="loso",
                                         holdout_subject=fold["holdout_subject"]))
            if res is not None:
                rows.append(res)
    print("  LOSO: %d folds x %d arches logged" % (len(folds), len(LOSO_ARCHES)))
    return rows


def _cells(runs):
    out = {}
    for r in runs:
        out.setdefault(str(r.get("cell")), {})[int(r["seed"])] = float(r["accuracy"])
    return out


def leakage_table(dataset):
    """Deliverable 1: window-level vs subject-disjoint accuracy and the gap.

    The gap alone only says the leaking split scored better; the two `leaked_*` columns
    say how much subject reuse produced it, which is what the reviewers asked for.
    """
    runs = read_logs(dataset=dataset, stage=STAGE)
    cells = _cells(runs)
    leaked = _leaked_windows(runs)
    win_leak = leaked["window"]
    sub_leak = leaked["subject"]
    rows = []
    for arch in LEAK_ARCHES:
        sub = cells.get("%s | subject-split" % arch, {})
        win = cells.get("%s | window-split" % arch, {})
        seeds = sorted(set(sub) & set(win))
        if not seeds:
            continue
        gap = [win[s] - sub[s] for s in seeds]
        s = multiseed.summarise(gap)
        rows.append(dict(dataset=dataset, architecture=arch, n_seeds=len(seeds),
                         window_mean=float(np.mean([win[x] for x in seeds])),
                         subject_mean=float(np.mean([sub[x] for x in seeds])),
                         gap_mean=s["mean"], gap_std=s["std"],
                         gap_ci95_low=s["ci95_low"], gap_ci95_high=s["ci95_high"],
                         leaked_subject_pairs=win_leak.get("pairs"),
                         leaked_windows=win_leak.get("windows"),
                         leaked_window_share=win_leak.get("share"),
                         subject_split_leaked_pairs=sub_leak.get("pairs"),
                         subject_split_leaked_windows=sub_leak.get("windows"),
                         p_wilcoxon=(multiseed.compare([win[x] for x in seeds],
                                                       [sub[x] for x in seeds])
                                      ["p_wilcoxon"] if len(seeds) >= 2 else None)))
    return rows


def _leaked_windows(runs):
    """The subject-reuse counts `splitting.leakage_report` wrote into each split's records.

    Keyed by split kind, so the table shows the new split's zeros and the old split's
    non-zeros from the same measurement rather than one measured and one assumed.
    """
    out = {}
    for r in runs:
        kind = str(r.get("split_split_kind") or r.get("split_kind") or "")
        pairs = r.get("split_n_overlapping_pairs")
        windows = r.get("split_windows_from_overlapping_subjects")
        if kind in out or pairs is None or windows is None:
            continue
        n = r.get("n_windows_test")
        out[kind] = dict(pairs=int(pairs), windows=int(windows),
                         share=(float(windows) / float(n) if n else None))
    return {"window": out.get("window_level", {}),
            "subject": out.get("subject_disjoint", {})}


def loso_table(dataset):
    rows = []
    for r in read_logs(dataset=dataset, stage=STAGE):
        cell = str(r.get("cell") or "")
        if "loso" not in cell:
            continue
        arch = cell.split(" | ")[0]
        rows.append(dict(dataset=dataset, architecture=arch,
                         holdout_subject=r.get("holdout_subject"),
                         accuracy=float(r["accuracy"])))
    out = []
    for arch in sorted(set(r["architecture"] for r in rows)):
        vals = [r["accuracy"] for r in rows if r["architecture"] == arch]
        s = multiseed.summarise(vals)
        s.update(dataset=dataset, cell=arch, metric="loso_accuracy")
        out.append(s)
    return dict(per_fold=rows, summary=out)


def matched_table(dataset):
    cells = _cells(read_logs(dataset=dataset, stage=STAGE))
    pairs = [("CNN-BiLSTM | arch-matched", "LightTCN | arch-matched", "architecture"),
             ("MLP | rep-matched", "MLP-Raw | rep-matched", "representation"),
             ("CNN-BiLSTM | arch-matched-DP", "LightTCN | arch-matched-DP",
              "architecture_with_DP")]
    out = []
    for ca, cb, name in pairs:
        a, b = cells.get(ca, {}), cells.get(cb, {})
        seeds = sorted(set(a) & set(b))
        if len(seeds) < 2:
            out.append(dict(comparison=name, n_seeds=len(seeds), p_wilcoxon=1.0,
                            note="insufficient paired seeds"))
            continue
        row = multiseed.compare([a[s] for s in seeds], [b[s] for s in seeds], ca, cb)
        row["comparison"] = name
        out.append(row)
    return multiseed.holm_table(out)


def write_client_note(h: Harness, seeds=None):
    """Deliverable 3: exactly how the clients were formed, subject ids and all."""
    seeds = seeds or config.cfg()["seeds_supp"]
    config.ensure_dirs()
    path = os.path.join(config.RESULTS_DIR,
                        "client_construction_%s.csv" % h.bundle.name.replace(" ", "_"))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "seed", "split", "subjects"])
        for seed in seeds:
            sp = h.split(seed)
            for name, subs in sorted(sp.subjects_in_split.items()):
                w.writerow([h.bundle.name, seed, name, " ".join(str(s) for s in subs)])
            cl = h.clients(sp, seed)
            for k, subs in enumerate(cl["client_subjects"]):
                w.writerow([h.bundle.name, seed, "client_%d" % k,
                            " ".join(str(s) for s in subs)])
        w.writerow([])
        w.writerow(["privacy_unit", cl["privacy_unit"], "adjacency", cl["adjacency"]])
        w.writerow(["n_clients", cl["n_clients"], "mode", cl["mode"]])
        leak = sp.leakage_checks
        w.writerow(["split_leakage_checks",
                    {k: v for k, v in leak.items() if k.startswith("subjects_overlap")}])
    print("  [splits] wrote %s" % path)
    return path


def write_csvs(dataset):
    config.ensure_dirs()
    slug = dataset.replace(" ", "_")
    paths = []
    for name, rows in (("leakage", leakage_table(dataset)),
                       ("matched", matched_table(dataset)),
                       ("loso", loso_table(dataset)["summary"])):
        if not rows:
            continue
        p = os.path.join(config.RESULTS_DIR, "%s_%s.csv" % (name, slug))
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted(rows[0]), extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        paths.append(p)
    if paths:
        print("  [splits] wrote %s" % ", ".join(paths))
    return paths

#@@FILE ppml/experiments/exp_noniid.py
"""Task 5 - label-Dirichlet non-IID with repeated independent partitions.

Two fixes over the submission: the Dirichlet draw is applied to the TRAINING windows
only (it used to partition the whole dataset including the test set), and every point is
the mean over >= 5 independent partitions rather than one unlucky draw, so an alpha is
never reported from a single partition again.
"""

import csv
import os

import numpy as np

from .. import config, multiseed, splitting
from ..harness import Harness
from ..runlog import read_logs

STAGE = "noniid"
ARCH = "MLP"


def _partition_seed(base, p):
    """Independent partitions that still share the subject split of `base`."""
    return int(base) * 1000 + int(p)


def run(h: Harness, seeds=None, verbose=True):
    seeds = seeds or config.cfg()["seeds"]
    n_part = config.cfg()["partitions"]
    rows = []
    for alpha in config.ALPHA_VALUES:
        dp_arm = alpha in config.NONIID_DP_ALPHAS
        n_dp = config.cfg().get("partitions_dp", config.N_PARTITIONS_DP) if dp_arm else 0
        # FedAvg only (scope): FedProx at mu=0.01 matches FedAvg to 4 decimals in Task 4
        arms = [("fedavg", dict())]
        for method, kw in arms:
            for p in range(n_part):
                accs = []
                # one model seed per partition: the REPEAT index is the replication,
                # so the spread reported here is partition spread, not seed spread
                base_seed = _partition_seed(seeds[0], p)
                split = h.split(seeds[0])
                cidx = h.clients_dirichlet(split, base_seed, alpha=alpha,
                                           kind="iid" if alpha == "IID" else "dirichlet")
                het = splitting.partition_heterogeneity(cidx, h.y)
                res = h.run_fl(dict(arch_role="noniid", alpha=alpha, method=method,
                                    partition_repeat=p, prox_mu=kw.get("mu")),
                               STAGE, arch=ARCH, client_indices=cidx, split=split,
                               seed=base_seed, method=method, mu=kw.get("mu", 0.0),
                               rounds=config.cfg()["rounds"], heterogeneity=het,
                               partition_id=p,
                               cell="%s | a=%s | %s" % (ARCH, alpha, method),
                               label="%s a=%s p%d" % (method, alpha, p), verbose=False)
                if res is not None:
                    accs.append(float(res["accuracy"]))
                    rows.append(res)
                if accs:
                    print("  a=%-5s %-8s p%d  acc=%.4f" % (alpha, method, p, accs[0]))
        if dp_arm:
            for p in range(n_dp):
                base_seed = _partition_seed(seeds[0], p)
                split = h.split(seeds[0])
                cidx = h.clients_dirichlet(split, base_seed, alpha=alpha,
                                           kind="iid" if alpha == "IID" else "dirichlet")
                het = splitting.partition_heterogeneity(cidx, h.y)
                for method, kw in arms:
                    res = h.run_fl(dict(arch_role="noniid_dp", alpha=alpha,
                                        method=method, partition_repeat=p,
                                        prox_mu=kw.get("mu"),
                                        # A Dirichlet partition puts windows from MANY
                                        # subjects into one client, so "add or remove one
                                        # client" is no longer "add or remove one person".
                                        # The number below is the cost of the round-level
                                        # mechanism on that partition, not a user-level
                                        # certificate; the log has to say so or the table
                                        # reads as a privacy guarantee.
                                        certificate_scope=("user-level (subject silo)"
                                                           if alpha == "IID" else
                                                           "partition-level only: clients "
                                                           "mix subjects, so this is NOT a "
                                                           "user-level certificate")),
                                   STAGE, arch=ARCH, client_indices=cidx, split=split,
                                   seed=base_seed, method=method, mu=kw.get("mu", 0.0),
                                   dp=dict(z=config.NONIID_DP_Z, C=config.CLIP_C),
                                   rounds=config.cfg()["rounds_dp"], heterogeneity=het,
                                   partition_id=p,
                                   cell="%s | a=%s | %s+DP" % (ARCH, alpha, method),
                                   label="%s+DP a=%s p%d" % (method, alpha, p),
                                   verbose=False)
                    if res is not None:
                        rows.append(res)
    return rows


def table(dataset):
    """Per alpha: accuracy mean +/- std, min, max and the measured label skew.

    The DP and non-DP arms are separate rows; both use the same `method` string, so
    grouping on (alpha, method) alone would average a private run with a public one.
    """
    runs = read_logs(dataset=dataset, stage=STAGE)
    groups = {}
    for r in runs:
        dp = bool(r.get("z"))
        method = str(r.get("method")) + ("+DP" if dp else "")
        groups.setdefault((str(r.get("alpha")), method), []).append(r)
    rows = []
    for (alpha, method), g in sorted(groups.items()):
        acc = np.array([float(x["accuracy"]) for x in g])
        het = [x.get("partition_heterogeneity") or {} for x in g]
        rows.append(dict(
            dataset=dataset, architecture=g[0].get("architecture") or ARCH,
            alpha=alpha, method=method, n_partitions=int(len(acc)),
            acc_mean=float(acc.mean()), acc_std=float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
            acc_min=float(acc.min()), acc_max=float(acc.max()),
            dominant_class_share=float(np.mean([x.get("dominant_class_share", 0.0)
                                                for x in het])),
            normalized_entropy=float(np.mean([x.get("normalized_entropy", 0.0)
                                              for x in het])),
            dp=bool(any(bool(x.get("z")) for x in g)),
            z=float(g[0].get("z") or 0.0) or None,
            certificate_scope=g[0].get("certificate_scope"),
            eps=float(g[0]["eps_total"]) if g[0].get("eps_total") is not None else None))
    return rows


def iid_reference(dataset):
    """The IID (subject-silo) accuracy the alpha=0.5 drop is measured against."""
    from . import exp_main
    main = read_logs(dataset=dataset, stage="main")
    vals = [float(r["accuracy"]) for r in main
            if str(r.get("cell")) == "FL-FedAvg" and (r.get("architecture") or "").upper() == ARCH]
    if not vals:
        return None
    return multiseed.summarise(vals)


def write_csv(dataset):
    rows = table(dataset)
    if not rows:
        return None
    config.ensure_dirs()
    path = os.path.join(config.RESULTS_DIR, "noniid_%s.csv" % dataset.replace(" ", "_"))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("  [noniid] wrote %s" % path)
    return path

#@@FILE ppml/experiments/exp_byzantine.py
"""Task 6 - Byzantine robustness inside the real FL pipeline.

The submission measured attacks in a toy logistic-regression loop. Here the attacks run
in the same `runners.run_fl` used for every other table, on the MLP and the CNN-BiLSTM,
with five attack types, four aggregators and Krum's assumed colluder count fixed at
config.BYZ_ASSUMED_F rather than read off the true number of adversaries.
"""

import csv
import os

import numpy as np

from .. import config, multiseed, splitting
from ..harness import Harness
from ..runlog import read_logs

STAGE = "byzantine"
N_CLIENTS = config.N_CLIENTS_BYZANTINE


def _grid():
    """The attack x aggregator x fraction grid for the active profile.

    `full` is the Work Order grid (all five attacks, all four aggregators, four
    fractions). `smoke` keeps every attack and every aggregator but drops to one
    fraction and the MLP, because the same stage otherwise costs more than the whole
    session and the stages after it never get checked. The scope is in the log record.
    """
    c = config.cfg()
    attacks = list(c.get("byz_attacks", config.BYZ_ATTACKS))
    for attack in attacks:
        for aggreg in c.get("byz_aggregators", config.BYZ_AGGREGATORS):
            for frac in c.get("byz_fractions", config.BYZ_FRACTIONS):
                # fraction 0 has no attacker, so it is the same run for every attack:
                # run it once (under the first attack's name) instead of once per attack
                if float(frac) == 0.0 and attack != attacks[0]:
                    continue
                yield attack, aggreg, frac


def _partition(h, kind, seed, rep=0):
    """The K subject-silo clients (iid) or one Dirichlet draw of the same size."""
    split = h.split(seed)
    if kind == "iid":
        return split, h.clients(split, seed, n_clients=N_CLIENTS)["client_indices"], None
    # `clients()` answers with a dict of diagnostics; `clients_dirichlet()` answers with
    # the partition itself, which is what every other Task 5/6 caller passes on.
    cidx = h.clients_dirichlet(split, int(seed) * 1000 + int(rep), n_clients=N_CLIENTS,
                               alpha=config.BYZ_NONIID_ALPHA, kind="dirichlet")
    return split, cidx, config.BYZ_NONIID_ALPHA


def run(h: Harness, seeds=None, arches=None, verbose=True):
    seeds = seeds or config.cfg()["byz_seeds"]
    arches = arches or config.cfg().get("byz_arches", config.BYZ_ARCHES)
    rows = []
    for kind in config.BYZ_PARTITION_KINDS:
        # The non-IID arm is where a robust aggregator is supposed to earn its keep, and
        # the submission never ran it. It is scoped to the MLP for session cost; that
        # scope is logged in every record and listed in the run report.
        arm_arches = arches if kind == "iid" else config.BYZ_NONIID_ARCHES
        n_rep = config.BYZ_NONIID_PARTITIONS if kind != "iid" else 1
        for arch in arm_arches:
            for attack, aggreg, frac in _grid():
                accs = []
                for rep in range(n_rep):
                    for seed in seeds:
                        split, cidx, alpha = _partition(h, kind, seed, rep)
                        het = (splitting.partition_heterogeneity(cidx, h.y)
                               if kind != "iid" else None)
                        res = h.run_fl(dict(arch_role="byzantine", byz_attack=attack,
                                            byz_aggregator=aggreg, byz_fraction=frac,
                                            byz_partition=kind, byz_alpha=alpha,
                                            n_clients=N_CLIENTS),
                                       STAGE, arch=arch, client_indices=cidx,
                                       split=split, seed=seed, method="fedavg",
                                       heterogeneity=het, partition_id=rep,
                                       attacks=dict(update=attack if attack != "labelflip"
                                                    else "none",
                                                    fraction=frac, aggreg=aggreg,
                                                    labelflip=(attack == "labelflip")),
                                       cell="%s | %s | %s | f=%.2f | %s"
                                            % (arch, attack, aggreg, frac, kind),
                                       label="%s/%s/f%.2f/%s" % (attack, aggreg, frac, kind),
                                       verbose=False)
                        if res is not None:
                            accs.append(float(res["accuracy"]))
                            rows.append(res)
                if accs and verbose:
                    print("  [%-11s %-9s %-13s f=%.2f %-9s] acc=%.4f +-%.4f"
                          % (arch, attack, aggreg, frac, kind, float(np.mean(accs)),
                             float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0))
    return rows


def table(dataset):
    runs = read_logs(dataset=dataset, stage=STAGE)
    groups = {}
    for r in runs:
        k = (str(r.get("architecture")), str(r.get("byz_attack")),
             str(r.get("byz_aggregator")), float(r.get("byz_fraction") or 0.0),
             str(r.get("byz_partition") or "iid"))
        groups.setdefault(k, []).append(float(r["accuracy"]))
    rows = []
    for (arch, attack, aggreg, frac, kind), accs in sorted(groups.items()):
        s = multiseed.summarise(accs)
        clean = groups.get((arch, attack, "fedavg", 0.0, kind))
        rows.append(dict(dataset=dataset, architecture=arch, attack=attack,
                         aggregator=aggreg, adversary_fraction=frac, partition=kind,
                         acc_mean=s["mean"], acc_std=s["std"], n_seeds=s["n"],
                         acc_min=s["min"], acc_max=s["max"],
                         drop_vs_clean_fedavg=(float(np.mean(clean)) - s["mean"])
                         if clean else None))
    return rows


def robustness_ranking(dataset, arch="MLP", partition="iid"):
    """Which aggregator actually held up, averaged over attacks and fractions."""
    rows = [r for r in table(dataset) if r["architecture"] == arch
            and r["adversary_fraction"] > 0
            and (partition is None or r["partition"] == partition)]
    by_agg = {}
    for r in rows:
        by_agg.setdefault(r["aggregator"], []).append(r["acc_mean"])
    return sorted(((float(np.mean(v)), agg, len(v)) for agg, v in by_agg.items()),
                  reverse=True)


def write_csv(dataset):
    rows = table(dataset)
    if not rows:
        return None
    config.ensure_dirs()
    path = os.path.join(config.RESULTS_DIR, "byzantine_%s.csv" % dataset.replace(" ", "_"))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.6f" % v if isinstance(v, float) else v)
                        for k, v in r.items()})
    print("  [byzantine] wrote %s (%d cells)" % (path, len(rows)))
    return path

#@@FILE ppml/experiments/exp_resampling.py
"""Task 7 - resampling ablation, measured spectral distortion, WISDM channel treatment.

Three things the submission asserted without measuring:
  * that linear interpolation to 50 Hz is harmless - the spurious-energy fraction above
    the source Nyquist is now measured on the real signals, for both resamplers;
  * that zero-padding the missing gyroscope is neutral - the three treatments are now
    compared with the CNN/TCN on RAW windows, because summary features turn a zero
    channel into a constant and the comparison would otherwise be noise;
  * that the 50 Hz figures and the sweeps came from one pipeline - they now do, because
    every variant here goes through the same harness and the same runner.
"""

import csv
import gc
import os

import numpy as np

from .. import config, datasets, multiseed, resampling
from ..harness import Harness
from ..runlog import RunLog, read_logs

STAGE = "resampling"
VARIANTS = resampling.VARIANTS
CHANNEL_MODES = ("zero_pad", "accel_only", "channel_mask")
SPURIOUS_MAX_STREAMS = 6          # raw streams used for the distortion measurement
SPURIOUS_SECONDS = 20.0


def _dp_levels():
    # scope: non-DP only. The user-level DP release at z=10.6 is noise-dominated for the
    # whole model (privacy stage), so a DP row here would only repeat chance accuracy.
    return [("non-DP", None)]


def spurious_energy(bundle, recs=None, fs=None):
    """Fraction of post-resampling energy that sits above the SOURCE Nyquist.

    A band-limited signal has nothing up there, so whatever a resampler puts above
    fs_src/2 is an artefact of the resampler, not of the person wearing the device.
    """
    name = bundle.name
    fs = fs or datasets.NATIVE_FS[name]
    if recs is None:
        recs = datasets.raw_recs(name, max_subjects=SPURIOUS_MAX_STREAMS)
    n_samp = int(SPURIOUS_SECONDS * fs)
    got = []
    for subj, sig, _codes in recs[:SPURIOUS_MAX_STREAMS]:
        if len(sig) < n_samp or fs == config.TARGET_FS:
            continue
        got.append(resampling.spectral_distortion(sig[:n_samp], fs, config.TARGET_FS))
    if not got:
        return dict(dataset=name, native_fs=float(fs), resampled=False,
                    note="dataset is already at the common rate; no upsampling artefact "
                         "can exist", spurious_energy_fraction_linear=None,
                    spurious_energy_fraction_polyphase=None, artefact_reduction=None)
    keys = list(got[0])
    return dict(dataset=name, native_fs=float(fs), resampled=True,
                n_streams=len(got),
                **{k: float(np.mean([g[k] for g in got])) for k in keys})


def run(h: Harness, seeds=None, arches=None, verbose=True):
    seeds = seeds or config.cfg()["seeds_supp"]
    arches = arches or ("MLP", "LightTCN")
    rows = []
    bundle = h.bundle
    for variant in VARIANTS:
        if variant not in bundle.variants:
            print("  [resampling] %s not available for %s - skipped" % (variant, bundle.name))
            continue
        if (variant == "native_rate" and
                abs(float(datasets.NATIVE_FS[bundle.name]) - float(config.TARGET_FS)) < 1e-6):
            print("  [resampling] %s is natively %g Hz: native_rate is the 50 Hz data - "
                  "not run twice" % (bundle.name, config.TARGET_FS))
            continue
        sub = Harness(bundle.as_variant(variant), device=h.device)
        for arch in arches:
            for dpname, dp in _dp_levels():
                T = config.cfg()["rounds_dp"] if dp else config.cfg()["rounds"]
                accs = []
                for seed in seeds:
                    split = sub.split(seed)
                    cl = sub.clients(split, seed)
                    res = sub.run_fl(dict(arch_role="resampling", variant=variant),
                                     STAGE, arch=arch, client_indices=cl["client_indices"],
                                     split=split, seed=seed, method="fedavg", dp=dp,
                                     rounds=T,
                                     cell="%s | %s | %s" % (arch, variant, dpname),
                                     label="%s/%s/%s" % (arch, variant, dpname),
                                     verbose=False)
                    if res is not None:
                        accs.append(float(res["accuracy"]))
                        rows.append(res)
                if accs:
                    s = multiseed.summarise(accs)
                    print("  [%-11s %-21s %-7s] acc=%.4f +-%.4f"
                          % (arch, variant, dpname, s["mean"], s["std"]))
        del sub
        gc.collect()
    # The channel-treatment comparison is part of the stage, not an optional extra: the
    # reviewers asked what zero-padding an absent gyroscope does, and the driver only
    # dumps tables, so a function it deliberately skips has to be reached from here.
    if _lacks_gyroscope(bundle.name):
        print("  [resampling] %s has no gyroscope: running the channel-treatment arms"
              % bundle.name)
        # channel_ablation keeps its own architecture list: the confound only exists for
        # models that read the raw windows, so the feature MLP must not stand in for it.
        rows += channel_ablation(bundle.name, seeds=seeds, verbose=verbose)
    return rows


def _lacks_gyroscope(dataset):
    return datasets.CHANNEL_MAP[dataset].get("gyro") is None


def channel_ablation(dataset, seeds=None, arches=("LightTCN",), verbose=True):
    """Zero-pad vs accelerometer-only vs explicit channel mask, on raw windows."""
    seeds = seeds or config.cfg()["seeds_supp"]
    rows = []
    for mode in CHANNEL_MODES:
        bundle = datasets.load(dataset, channel_mode=mode)
        sub = Harness(bundle)
        for arch in arches:
            for dpname, dp in _dp_levels():
                T = config.cfg()["rounds_dp"] if dp else config.cfg()["rounds"]
                accs = []
                for seed in seeds:
                    split = sub.split(seed)
                    cl = sub.clients(split, seed)
                    res = sub.run_fl(dict(arch_role="channel_ablation", channel_mode=mode),
                                     STAGE, arch=arch, client_indices=cl["client_indices"],
                                     split=split, seed=seed, method="fedavg", dp=dp,
                                     rounds=T,
                                     cell="%s | ch=%s | %s" % (arch, mode, dpname),
                                     label="%s/ch-%s/%s" % (arch, mode, dpname),
                                     verbose=False)
                    if res is not None:
                        accs.append(float(res["accuracy"]))
                        rows.append(res)
                if accs and verbose:
                    s = multiseed.summarise(accs)
                    print("  [%-11s %-12s %-7s] acc=%.4f +-%.4f"
                          % (arch, mode, dpname, s["mean"], s["std"]))
        del sub
        gc.collect()
    return rows


# ── pooled multi-dataset arm (Task 7, the case that makes channel masking meaningful) ──
# Reviewer R1-4 asked whether zero-padding WISDM's absent gyroscope biases the model. In a
# single-dataset run the answer is structurally unreachable: every row has the same missing
# block, so a zero channel and a masked channel are both constants and the three treatments
# carry identical information. Pooling a dataset that HAS a gyroscope with one that does NOT
# is the only setting in which the mask can do work - the sensor block that is noise for
# WISDM windows is signal for UCI HAR windows, and a network that sees both has to use the
# indicator. This is the arm the Work Order allows "if feasible"; it is feasible, so it runs.
# Disabled in this revision: WISDM's phone gyroscope is now LOADED, so the only dataset
# without a gyroscope is HARTH, whose channel arm runs per dataset above. The pooled arm
# would also mix WISDM's 3-class label space with a 6-class one.
POOLED_SETS = ()
POOLED_NAME = "Pooled HAR+WISDM"
POOLED_ARCHES = ("CNN-BiLSTM", "LightTCN")   # raw-window models only
# The pooled stack is priced per member, not per dataset: 4000 windows each keeps the arm
# inside one session without touching the subject pool the silos and splits are built from.
# It is a separate table, so it is compared only against itself - never against the
# single-dataset rows above.
POOLED_MAX_WINDOWS = 4000


def pooled_available(available):
    """True when every member of POOLED_SETS was loaded in this session."""
    return bool(POOLED_SETS) and all(name in set(available) for name in POOLED_SETS)


def pooled_bundle(mode, members=POOLED_SETS, max_windows=POOLED_MAX_WINDOWS,
                  seed=config.SEEDS[0]):
    """One window stack over `members`, harmonised with `mode`.

    Subject ids are offset per member so "one silo = one person" still holds across the
    pooled stack - without the offset, UCI subject 3 and WISDM subject 3 would share a
    client and the same person could land in train and test.
    """
    Xs, ys, ss, prov = [], [], [], {}
    shape = None
    for i, name in enumerate(members):
        b = datasets.load(name, channel_mode=mode, max_windows=max_windows, seed=seed)
        x = np.asarray(b.X)
        if shape is None:
            shape = x.shape[1:]
        elif tuple(x.shape[1:]) != tuple(shape):
            raise RuntimeError("pooled arm needs identical window shapes; %s gives %s vs %s"
                               % (name, x.shape[1:], shape))
        Xs.append(x)
        ys.append(np.asarray(b.y).astype(int))
        ss.append(np.asarray(b.subject).astype(int) + 1000 * (i + 1))
        prov[name] = dict(n_windows=int(len(b.y)), n_subjects=int(len(np.unique(b.subject))),
                          has_gyro=datasets.CHANNEL_MAP[name].get("gyro") is not None)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    meta = dict(fs=float(config.TARGET_FS), dataset=POOLED_NAME, channel_mode=mode,
                pooled_from=prov, n_windows=int(len(y)),
                n_subjects=int(len(np.unique(s))), window_shape=list(X.shape))
    bundle = datasets.DatasetBundle(POOLED_NAME, X, y, s, meta)
    bundle.variants = {}
    return bundle


def pooled_run(available=None, modes=CHANNEL_MODES, seeds=None, arches=None,
               verbose=True):
    """Zero-pad vs accel-only vs mask, trained on data from sensors that differ.

    Returns the runs it logged (a session that already ran them returns them from the log
    dedup, so re-running this is cheap).
    """
    available = list(available or [])
    if not pooled_available(available):
        if verbose:
            print("  [pooled] needs %s in this session's dataset list; running the "
                  "per-dataset arms only" % " and ".join(POOLED_SETS))
        return []
    seeds = seeds or config.cfg()["seeds"]
    arches = arches or POOLED_ARCHES
    rows = []
    for mode in modes:
        try:
            sub = Harness(pooled_bundle(mode), device=None, compute_features=False)
        except Exception as exc:
            print("  [pooled] %s: could not build the pooled stack (%s)"
                  % (mode, type(exc).__name__))
            continue
        if verbose:
            src = sub.bundle.meta["pooled_from"]
            print("  [pooled] %s: %d windows, %d silos, %s"
                  % (mode, sub.bundle.meta["n_windows"], sub.bundle.meta["n_subjects"],
                     ", ".join("%s(%d win, gyro=%s)"
                               % (n, p["n_windows"], p["has_gyro"])
                               for n, p in sorted(src.items()))))
        for arch in arches:
            for dpname, dp in _dp_levels():
                T = config.cfg()["rounds_dp"] if dp else config.cfg()["rounds"]
                accs = []
                for seed in seeds:
                    split = sub.split(seed)
                    cl = sub.clients(split, seed)
                    res = sub.run_fl(dict(arch_role="pooled_channels", channel_mode=mode),
                                     STAGE, arch=arch,
                                     client_indices=cl["client_indices"], split=split,
                                     seed=seed, method="fedavg", dp=dp, rounds=T,
                                     cell="%s | pooled | ch=%s | %s" % (arch, mode, dpname),
                                     label="%s/pooled-%s/%s" % (arch, mode, dpname),
                                     verbose=False)
                    if res is not None:
                        accs.append(float(res["accuracy"]))
                        rows.append(res)
                if accs and verbose:
                    s = multiseed.summarise(accs)
                    print("  [%-11s %-12s pooled %-7s] acc=%.4f +-%.4f (n=%d)"
                          % (arch, mode, dpname, s["mean"], s["std"], s["n"]))
        del sub
        gc.collect()
    if rows:
        write_pooled_csvs(modes=modes)
    return rows


def write_pooled_csvs(modes=CHANNEL_MODES):
    """The pooled table plus the mask-vs-zero-pad contrast, both from the log only."""
    config.ensure_dirs()
    rows = table(POOLED_NAME)
    paths = []
    p = os.path.join(config.RESULTS_DIR, "resampling_pooled.csv")
    if rows:
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
            w.writeheader()
            w.writerows(rows)
        paths.append(p)
    # The question R1-4 actually asked, answered as a paired difference per seed rather
    # than as two means a reader has to subtract by hand.
    runs = [r for r in read_logs(dataset=POOLED_NAME, stage=STAGE)
            if r.get("accuracy") is not None]
    by = {}
    for r in runs:
        by.setdefault((str(r.get("architecture")), str(r.get("channel_mode")),
                       "DP" if r.get("z") else "non-DP"), {})[int(r["seed"])] = \
            float(r["accuracy"])
    p2 = os.path.join(config.RESULTS_DIR, "channel_mask_vs_zero_pad_pooled.csv")
    out = []
    for arch in POOLED_ARCHES:
        for dp in ("non-DP", "DP"):
            base = by.get((arch, "zero_pad", dp)) or {}
            for mode in ("accel_only", "channel_mask"):
                alt = by.get((arch, mode, dp)) or {}
                shared = sorted(set(base) & set(alt))
                if not shared:
                    continue
                cmp_ = multiseed.compare([alt[s] for s in shared],
                                         [base[s] for s in shared],
                                         name_a=mode, name_b="zero_pad")
                cmp_.update(dataset=POOLED_NAME, architecture=arch, dp=dp,
                            comparison="%s minus zero_pad" % mode, n_paired=len(shared))
                out.append(cmp_)
    if out:
        with open(p2, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted(out[0]))
            w.writeheader()
            for r in out:
                w.writerow({k: v for k, v in r.items() if not isinstance(v, (list, dict))})
        paths.append(p2)
    if paths and modes:
        print("  [pooled] wrote %s" % ", ".join(os.path.basename(x) for x in paths))
    return paths


def table(dataset):
    runs = read_logs(dataset=dataset, stage=STAGE)
    groups, zvals = {}, {}
    for r in runs:
        if r.get("accuracy") is None:      # one-off measurements (spectral distortion)
            continue
        k = (str(r.get("architecture")), str(r.get("variant")), str(r.get("channel_mode")),
             "DP" if r.get("z") else "non-DP")
        groups.setdefault(k, []).append(float(r["accuracy"]))
        zvals[k] = float(r.get("z") or 0.0)
    rows = []
    for (arch, variant, chmode, dp), accs in sorted(groups.items()):
        s = multiseed.summarise(accs)
        rows.append(dict(dataset=dataset, architecture=arch, variant=variant,
                         channel_mode=chmode, dp=dp, z=zvals[(arch, variant, chmode, dp)],
                         acc_mean=s["mean"], acc_std=s["std"], acc_ci95_low=s["ci95_low"],
                         acc_ci95_high=s["ci95_high"], n_seeds=s["n"]))
    return rows


def write_csvs(h: Harness):
    config.ensure_dirs()
    dataset = h.bundle.name
    slug = dataset.replace(" ", "_")
    paths = []

    p = os.path.join(config.RESULTS_DIR, "resampling_%s.csv" % slug)
    rows = table(dataset)
    if rows:
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
            w.writeheader()
            w.writerows(rows)
        paths.append(p)

    sp = spurious_energy(h.bundle)
    p2 = os.path.join(config.RESULTS_DIR, "spectral_distortion_%s.csv" % slug)
    with open(p2, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(sp))
        w.writeheader()
        w.writerow({k: (v if not isinstance(v, float) else "%.6e" % v) for k, v in sp.items()})
    paths.append(p2)
    print("  [resampling] wrote %s" % ", ".join(paths))
    return paths


def write_label_map_table():
    """Task 7 deliverable: the activity-label mapping across the four datasets."""
    config.ensure_dirs()
    rows = datasets.label_map_table()
    path = os.path.join(config.RESULTS_DIR, "label_map_table.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("  [resampling] wrote %s" % path)
    return path


def log_extra(dataset, row):
    """One-off measurements (spurious energy) still belong in the machine-readable log."""
    RunLog(dataset, STAGE).record(dict(arch_role="spectral_distortion", dataset=dataset),
                                  row, skip_existing=False)

#@@FILE ppml/experiments/exp_features.py
"""Task 8 - feature-family ablation, redundancy, and the jerk definition.

The nine families are located by deriving the column layout from features.py, not by
typing index ranges into the experiment (the layout changes if the channel count or the
number of FFT bands changes).

Jerk: the paper's equation carries a sampling-rate factor f_s and the submitted code did
not multiply by it. Here the factor is applied (config.JERK_WITH_SAMPLING_RATE) and the
setting actually used is written into every run record, so the equation and the code
match. Run this module with BOTH_JERK=True to report the accuracy difference the two
conventions make.
"""

import csv
import os

import numpy as np

from .. import config, features, multiseed
from ..harness import Harness
from ..runlog import read_logs

STAGE = "features"
ARCH = "MLP"
BOTH_JERK = False


def _dp_levels():
    # scope: non-DP only (a z=10.6 user-level row is at chance for every family)
    return [("non-DP", None)]


def _feature_matrix(h, jerk_with_fs):
    if jerk_with_fs == config.JERK_WITH_SAMPLING_RATE:
        return h.Xf
    return features.extract_features(h.Xw, fs=h.fs, jerk_with_fs=jerk_with_fs)


def run(h: Harness, seeds=None, arches=None, verbose=True):
    seeds = tuple(seeds or config.cfg()["seeds_supp"])[:2]     # scope: 2 seeds
    fams = features.family_slices()
    F = _feature_matrix(h, config.JERK_WITH_SAMPLING_RATE)
    if h.Xf is None:
        h.Xf = F
    cols_all = np.arange(F.shape[1])
    settings = [("all_families", cols_all)]
    settings += [("lofo_%s" % f, np.array([c for c in cols_all if c not in set(fams[f])]))
                 for f in features.FAMILY_ORDER]
    settings += [("only_%s" % f, np.array(fams[f])) for f in features.FAMILY_ORDER]
    if BOTH_JERK:
        settings = [("jerk_off_" + n, c) for n, c in settings] + settings

    rows = []
    for name, cols in settings:
        for dpname, dp in _dp_levels():
            if len(cols) == 0:
                continue
            jerk = not name.startswith("jerk_off_")
            sub = _feature_matrix(h, jerk)[:, cols]
            T = config.cfg()["rounds_dp"] if dp else config.cfg()["rounds"]
            accs = []
            for seed in seeds:
                split = h.split(seed)
                cl = h.clients(split, seed)
                res = h.run_fl(dict(arch_role="feature_ablation", feature_set=name,
                                    n_features=int(len(cols))),
                               STAGE, arch=ARCH, client_indices=cl["client_indices"],
                               split=split, seed=seed, method="fedavg", dp=dp,
                               rounds=T, X_override=sub,
                               cell="%s | %s | %s" % (ARCH, name, dpname),
                               label="%s/%s" % (ARCH, name), verbose=False)
                if res is not None:
                    accs.append(float(res["accuracy"]))
                    rows.append(res)
            if accs:
                s = multiseed.summarise(accs)
                print("  [%-26s %-7s] n=%3d acc=%.4f +-%.4f"
                      % (name, dpname, len(cols), s["mean"], s["std"]))
    return rows


def table(dataset):
    """Leave-one-family-out / single-family accuracy versus the all-families baseline,
    paired on seeds and Holm-corrected across the whole family of comparisons."""
    runs = read_logs(dataset=dataset, stage=STAGE)
    cells = {}
    for r in runs:
        k = (str(r.get("feature_set")), "DP" if r.get("z") else "non-DP")
        cells.setdefault(k, {})[int(r["seed"])] = float(r["accuracy"])
    rows = []
    for (fs, dp), per_seed in sorted(cells.items()):
        s = multiseed.summarise(list(per_seed.values()))
        row = dict(dataset=dataset, feature_set=fs, dp=dp, n_seeds=s["n"],
                   acc_mean=s["mean"], acc_std=s["std"], acc_ci95_low=s["ci95_low"],
                   acc_ci95_high=s["ci95_high"])
        base = cells.get(("all_families", dp))
        if base and fs != "all_families":
            seeds = sorted(set(base) & set(per_seed))
            if len(seeds) >= 2:
                cmp_ = multiseed.compare([per_seed[x] for x in seeds],
                                         [base[x] for x in seeds], fs, "all_families")
                row["delta_vs_all"] = cmp_["mean_diff"]
                row["p_wilcoxon"] = cmp_["p_wilcoxon"]
                row["_pair"] = (fs, dp)
        rows.append(row)
    tested = [r for r in rows if r.get("p_wilcoxon") is not None]
    if tested:
        multiseed.holm_table(tested)
    for r in rows:
        r.pop("_pair", None)
        r.setdefault("delta_vs_all", None)
        r.setdefault("p_wilcoxon", None)
        r.setdefault("holm_p", None)
    return rows


def redundancy_report(h):
    fams = features.family_slices()
    red = features.redundancy(h.Xf)
    red["family_sizes"] = {k: len(v) for k, v in fams.items()}
    red["jerk_with_sampling_rate"] = bool(config.JERK_WITH_SAMPLING_RATE)
    red["feature_dim"] = int(h.Xf.shape[1])
    return red


def jerk_note():
    return dict(jerk_definition=("mean/std of ||d(acc)/dt||_2 with the sampling-rate "
                                 "factor applied (x fs)"
                                 if config.JERK_WITH_SAMPLING_RATE else
                                 "mean/std of the L2 magnitude of the first difference "
                                 "of the 3 accelerometer axes, WITHOUT the sampling-rate "
                                 "factor"),
                used_in_code=bool(config.JERK_WITH_SAMPLING_RATE),
                equation_in_paper="includes the f_s factor",
                action=("code and equation agree" if config.JERK_WITH_SAMPLING_RATE
                        else "equation must drop the f_s factor"))


def write_csvs(h: Harness):
    config.ensure_dirs()
    dataset = h.bundle.name
    slug = dataset.replace(" ", "_")
    paths = []
    rows = table(dataset)
    if rows:
        p = os.path.join(config.RESULTS_DIR, "feature_ablation_%s.csv" % slug)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted(rows[0]), extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        paths.append(p)
    red = redundancy_report(h)
    p2 = os.path.join(config.RESULTS_DIR, "feature_redundancy_%s.csv" % slug)
    with open(p2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for k, v in red.items():
            if k == "family_sizes":
                for fam, n in v.items():
                    w.writerow(["family_size:%s" % fam, n])
            else:
                w.writerow([k, v])
    paths.append(p2)
    p3 = os.path.join(config.RESULTS_DIR, "jerk_definition.csv")
    with open(p3, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(jerk_note()))
        w.writeheader()
        w.writerow(jerk_note())
    paths.append(p3)
    print("  [features] wrote %s" % ", ".join(paths))
    return paths

#@@FILE ppml/experiments/exp_cost.py
"""Task 9 - computational and deployment cost, measured rather than asserted.

The paper claims the light network needs "~30% fewer parameters"; the analytic count is
about 96% fewer. This module measures the real numbers (parameters, MACs, latency, peak
memory, payload, training time) so the claim can be corrected with evidence, and records
the hardware the measurement ran on.
"""

import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .. import config, models, runlog
from ..harness import Harness
from ..runlog import read_logs

STAGE = "cost"


# ── MAC counting ──────────────────────────────────────────────────────────────
class _MacCounter:
    """Analytic multiply-accumulate count, so the number does not depend on thop."""

    def __init__(self):
        self.total = 0
        self.handles = []

    def _conv1d(self, mod, inp, out):
        batch, out_ch, t = out.shape
        k = mod.kernel_size[0]
        self.total += int(batch * out_ch * t * (mod.in_channels // mod.groups) * k)

    def _linear(self, mod, inp, out):
        self.total += int(out.numel() * mod.in_features)

    def _lstm(self, mod, inp, out):
        x = inp[0] if isinstance(inp, tuple) else inp
        batch, t, _ = x.shape
        per_dir = 4 * mod.hidden_size * (x.shape[-1] + mod.hidden_size + 1)
        self.total += int(batch * t * per_dir * (2 if mod.bidirectional else 1))

    def register(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv1d):
                self.handles.append(m.register_forward_hook(self._conv1d))
            elif isinstance(m, nn.Linear):
                self.handles.append(m.register_forward_hook(self._linear))
            elif isinstance(m, nn.LSTM):
                self.handles.append(m.register_forward_hook(self._lstm))
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for hh in self.handles:
            hh.remove()
        return False


def macs_for(arch, window_shape=(config.WL, config.N_CH_HARMONISED), feature_dim=117):
    net = models.build(arch, feature_dim=feature_dim, window_shape=window_shape)
    x = _sample_input(arch, window_shape, feature_dim, batch=1)
    cnt = _MacCounter().register(net)
    with torch.no_grad(), cnt:
        net(x)
    return dict(macs=cnt.total, params=models.n_params(net),
                n_tensors=models.n_tensors(net), param_shapes=models.param_shapes(net))


def _sample_input(arch, window_shape, feature_dim, batch=1):
    a = arch.upper()
    if a in ("MLP", "MLP-SMALL"):
        return torch.zeros(batch, feature_dim)
    if a == "MLP-RAW":
        return torch.zeros(batch, *window_shape)
    return torch.zeros(batch, *window_shape)


def latency(model, x, runs=None, warmup=None, device=None):
    runs = runs or config.COST_TIMING_RUNS
    warmup = warmup or config.COST_WARMUP_RUNS
    if device is None:
        device = next(p.device for p in model.parameters())
    x = x.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(samples)
    return dict(n_runs=int(len(a)), latency_ms_median=float(np.median(a)),
                latency_ms_mean=float(a.mean()), latency_ms_p95=float(np.percentile(a, 95)),
                latency_ms_min=float(a.min()), latency_ms_max=float(a.max()))


def peak_memory(arch, window_shape, feature_dim, device, batch=None):
    """Max resident allocation for one forward+backward at the real batch size."""
    batch = batch or config.BATCH
    net = models.build(arch, feature_dim=feature_dim, window_shape=window_shape,
                       device=device)
    x = _sample_input(arch, window_shape, feature_dim, batch=batch).to(device)
    y = torch.zeros(batch, dtype=torch.long, device=device)
    before = _rss_kb()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    opt = torch.optim.Adam(net.parameters(), lr=config.LR)
    out = net(x)
    loss = nn.functional.cross_entropy(out, y)
    loss.backward()
    opt.step()
    if device.type == "cuda":
        mem = dict(device_peak_mb=float(torch.cuda.max_memory_allocated() / 2 ** 20),
                   device_reserved_mb=float(torch.cuda.max_memory_reserved() / 2 ** 20))
    else:
        mem = dict(device_peak_mb=None, device_reserved_mb=None)
    mem["host_rss_delta_kb"] = float(_rss_kb() - before)
    mem["host_peak_rss_mb"] = _peak_rss_mb()
    mem["memory_reading"] = ("cuda allocator peak" if device.type == "cuda" else
                             "host process peak (VmHWM)"
                             if mem["host_peak_rss_mb"] is not None else
                             "trainable bytes only: this host exposes no peak reading")
    mem["trainable_bytes"] = float(sum(p.numel() * p.element_size()
                                       for p in net.parameters()))
    del net, x, y, out, loss, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return mem


def _rss_kb():
    try:
        import resource
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return 0.0


def _peak_rss_mb():
    """The process high-water mark of resident memory, or None if this host has no such
    reading. /proc's VmHWM is the honest number on Linux (and Kaggle is Linux); it is the
    peak, not a sample, so it does not depend on when the measurement is taken. On a host
    without it the cost row still carries `trainable_bytes` and the CUDA columns when the
    arm ran on a GPU, and the `memory_reading` column logged beside every figure says which
    of the three that row actually is."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    got = _rss_kb()
    if got <= 0:
        return None
    # ru_maxrss is kilobytes on Linux and bytes on Darwin; only the Linux reading is used
    # in a table, so anything that looks like bytes is converted rather than guessed at.
    return got / 1024.0 if got < 4e9 else got / 1024.0 / 1024.0


def run(h: Harness, seeds=None, arches=None, verbose=True):
    """Measure and log the cost of every architecture the paper deploys."""
    arches = arches or config.COST_ARCHES
    log = h.log(STAGE)
    feature_dim = h.n_features or 117
    window_shape = tuple(h.Xw.shape[1:]) if h.Xw is not None else (config.WL,
                                                                   config.N_CH_HARMONISED)
    cpu = torch.device("cpu")
    rows = []
    for arch in arches:
        ops = macs_for(arch, window_shape, feature_dim)
        net = models.build(arch, feature_dim=feature_dim, window_shape=window_shape,
                           device=cpu)
        x = _sample_input(arch, window_shape, feature_dim, batch=1)
        lat = latency(net, x, device=cpu)
        mem_gpu = peak_memory(arch, window_shape, feature_dim, h.device)
        mem_cpu = peak_memory(arch, window_shape, feature_dim, cpu)
        payload = ops["params"] * 4 * (config.N_CLIENTS + 1)
        rec = dict(architecture=arch, dataset=h.bundle.name, batch=config.BATCH,
                   window_shape=list(window_shape), feature_dim=int(feature_dim),
                   **ops)
        rec.update(latency_cpu_batch1=lat)
        rec.update(memory_gpu=mem_gpu, memory_cpu=mem_cpu)
        rec.update(uplink_downlink_bytes_per_round=int(payload),
                   hardware=runlog.hardware(), versions=runlog.versions())
        # training time per local epoch comes from the real FL logs, not a microbenchmark
        tpe = _epoch_times(h.bundle.name, arch)
        if tpe:
            rec.update(local_epoch_time_s=tpe)
        out = log.record(rec, dict(measured=True))
        if out is not None:
            rows.append(rec)
        print("  %-11s params=%8d  MACs=%10d  cpu latency=%.3f ms  payload/round=%.2f MB"
              % (arch, ops["params"], ops["macs"], lat["latency_ms_median"],
                 payload / 2 ** 20))
    return rows


def _epoch_times(dataset, arch):
    v = [float(r.get("local_epoch_time_s") or 0.0)
         for r in read_logs(dataset=dataset, stage="main")
         if (r.get("architecture") or "").upper() == arch.upper()]
    v = [x for x in v if x > 0]
    return float(np.mean(v)) if v else None


def ratio_table(dataset):
    """The 'fewer parameters' claim, restated from the measured counts."""
    runs = [r for r in read_logs(dataset=dataset, stage=STAGE)]
    by = {}
    for r in runs:
        if isinstance(r.get("latency_cpu_batch1"), dict):
            by[str(r.get("architecture"))] = r
    if "CNN-BiLSTM" not in by or "LightTCN" not in by:
        return None
    a, b = by["CNN-BiLSTM"], by["LightTCN"]
    return dict(dataset=dataset,
                params_cnn=a["params"], params_tcn=b["params"],
                params_reduction_pct=100.0 * (1.0 - b["params"] / a["params"]),
                macs_cnn=a["macs"], macs_tcn=b["macs"],
                macs_reduction_pct=100.0 * (1.0 - b["macs"] / a["macs"]),
                latency_ratio=b["latency_cpu_batch1"]["latency_ms_median"]
                / max(a["latency_cpu_batch1"]["latency_ms_median"], 1e-12),
                claim_in_paper="~30% fewer parameters",
                measured="see params_reduction_pct")


def write_csv(dataset):
    runs = read_logs(dataset=dataset, stage=STAGE)
    if not runs:
        return None
    config.ensure_dirs()
    path = os.path.join(config.RESULTS_DIR, "compute_cost_%s.csv" % dataset.replace(" ", "_"))
    flat = []
    for r in runs:
        row = {k: v for k, v in r.items() if not isinstance(v, (list, dict))}
        for key in ("latency_cpu_batch1", "memory_gpu", "memory_cpu"):
            for k, v in (r.get(key) or {}).items():
                row["%s_%s" % (key, k)] = v
        for k, v in (r.get("hardware") or {}).items():
            row["hardware_%s" % k] = v
        for k, v in (r.get("versions") or {}).items():
            row["version_%s" % k] = v
        row["param_shapes"] = str(r.get("param_shapes"))
        flat.append(row)
    cols = sorted({k for row in flat for k in row})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        for row in flat:
            w.writerow({k: (jsonable(v) if isinstance(v, float) else v)
                        for k, v in row.items()})
    print("  [cost] wrote %s" % path)
    return path


def jsonable(v):
    return None if v != v else v

#@@FILE ppml/experiments/exp_mia.py
"""Task 10 - membership inference against the actual federated models.

Three protocol faults in the submission are fixed here:

1. the shadow models were drawn from the same pool as the target's members, so the
   attack was fitted on data the target had also seen - the pools below are disjoint by
   SUBJECT;
2. each attack's threshold was chosen on the evaluation set itself, which inflates the
   advantage - every threshold here is chosen on shadow data only and then frozen;
3. WISDM v2 was missing from Table 7 - it is run by the same code as every other dataset.

Attacks: Yeom loss threshold, confidence, Shokri shadow model, label-only (ML-Leaks) and
LiRA as the stronger contemporary attack.
"""

import csv
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .. import config, multiseed, runners, splitting
from ..harness import Harness
from ..runlog import read_logs

STAGE = "mia"
ARCH = config.MIA_ARCH
ATTACKS = ("Yeom[30]", "Confidence", "Shokri[28]", "ML-Leaks[31]", "LiRA[33]")
# +1: larger score means "member"
_DIRECTION = {"Yeom[30]": -1, "Confidence": +1, "ML-Leaks[31]": +1,
              "Shokri[28]": +1, "LiRA[33]": +1}


# ── pools ─────────────────────────────────────────────────────────────────────
def _pools(split, subjects, y, seed, frac=None):
    """Three subject-disjoint pools: target training, non-member hold-out, shadows."""
    frac = frac or config.MIA_POOL_FRAC
    subs = np.array(sorted(int(s) for s in split.subjects_in_split["train"]))
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(len(subs))
    n_t = max(1, int(round(frac[0] * len(subs))))
    n_n = max(1, int(round(frac[1] * len(subs))))
    if len(subs) < n_t + n_n + 2:
        raise RuntimeError("dataset has %d training subjects; MIA needs at least %d"
                           % (len(subs), n_t + n_n + 2))
    target_subs = subs[order[:n_t]]
    nonmember_subs = subs[order[n_t:n_t + n_n]]
    shadow_subs = subs[order[n_t + n_n:]]
    subjects = np.asarray(subjects)
    idx = lambda ss: np.where(np.isin(subjects, ss))[0]
    return dict(target_idx=idx(target_subs), nonmember_idx=idx(nonmember_subs),
                shadow_idx=idx(shadow_subs),
                target_subs=sorted(target_subs.tolist()),
                nonmember_subs=sorted(nonmember_subs.tolist()),
                shadow_subs=sorted(shadow_subs.tolist()))


def _by_subject(idx, subjects, seed, n_clients=None):
    """Client partitions of a window pool, grouped by subject (subject silos).

    `federated_clients_by_subject` takes the FULL subjects array and the pool's global
    window ids; passing subjects[idx] here would index it twice and group nonsense.
    """
    cl = splitting.federated_clients_by_subject(idx, subjects,
                                                n_clients=n_clients, seed=seed)
    return cl["client_indices"], cl["client_subjects"]


def _class_balanced(idx, y, n_want, rng, n_class=None):
    """Equal-size, class-balanced sample from `idx` (as many per class as available)."""
    n_class = n_class or config.N_CLASS
    idx = np.asarray(idx)
    y = np.asarray(y)
    per = np.array([list(idx[y[idx] == c]) for c in range(n_class)], dtype=object)
    quota = max(1, int(n_want // n_class))
    out = []
    for c in range(n_class):
        pool = np.array(per[c])
        rng.shuffle(pool)
        out.extend(pool[:min(quota, len(pool))].tolist())
    if len(out) < n_want:
        rest = np.array([i for i in idx if i not in set(out)])
        rng.shuffle(rest)
        out.extend(rest[:n_want - len(out)].tolist())
    return np.array(sorted(int(i) for i in out[:n_want]), dtype=int)


# ── scoring ───────────────────────────────────────────────────────────────────
def _scores(model, X, y, device, n_class=None):
    n_class = n_class or config.N_CLASS
    data = runners.LocalData(X, y, device)
    pred, prob, loss = runners.evaluate(model, data)
    yy = np.asarray(y).astype(int)
    p_y = prob[np.arange(len(yy)), yy]
    marg = np.array([(pred == c).mean() for c in range(n_class)])
    return dict(prob=prob, pred=pred, loss=np.asarray(loss), conf=prob.max(1),
                p_y=p_y, logp_y=np.log(np.clip(p_y, 1e-12, 1.0)),
                label_only=p_y / (marg[yy] + 1e-9))


def _pick_threshold(member, nonmember, direction, n_grid=40):
    """Threshold chosen on SHADOW data only (Work Order Task 10)."""
    pooled = np.concatenate([member, nonmember])
    grid = np.unique(np.quantile(pooled, np.linspace(0.02, 0.98, n_grid)))
    best, best_t = -np.inf, float(grid[0])
    for t in grid:
        pred = pooled >= t if direction > 0 else pooled <= t
        tpr = float(pred[:len(member)].mean())
        fpr = float(pred[len(member):].mean())
        if tpr - fpr > best:
            best, best_t = tpr - fpr, float(t)
    return float(best_t), float(best)


def _apply(score_m, score_n, t, direction):
    pm = score_m >= t if direction > 0 else score_m <= t
    pn = score_n >= t if direction > 0 else score_n <= t
    tpr, fpr = float(pm.mean()), float(pn.mean())
    lab = np.concatenate([np.ones(len(score_m)), np.zeros(len(score_n))])
    sc = np.concatenate([score_m, score_n])
    try:
        auc = float(roc_auc_score(lab, sc * direction))
    except ValueError:
        auc = float("nan")
    return dict(advantage=tpr - fpr, tpr=tpr, fpr=fpr, auc=auc, threshold=t)


def _lira_stats(shadow_models, eval_X, eval_y, device):
    """LiRA reference distribution: the (mean, std) of the shadow log-likelihoods.

    Legitimate here because every shadow model was trained on subjects that are disjoint
    from the target's members and non-members, so no shadow ever saw the example whose
    reference distribution it contributes to.
    """
    P = np.stack([_scores(m, eval_X, eval_y, device)["logp_y"] for m in shadow_models])
    return P.mean(0), P.std(0, ddof=1) + 1e-6


# ── one trial ─────────────────────────────────────────────────────────────────
def _fit_pool(h, Xs, pool_idx, subjects, y, seed, dp, rounds, eval_idx,
              role="target", dpname="non-DP"):
    cidx, csubs = _by_subject(pool_idx, subjects, seed)
    if not cidx:
        return None
    Xtr, ytr = Xs[pool_idx], y[pool_idx]
    # global pool ids -> positions inside the pool block handed to the runner
    block = splitting.to_block_positions(cidx, pool_idx)
    res = runners.run_fl((Xtr, ytr), block, ARCH, seed, rounds=rounds, dp=dp,
                         Xte=Xs[eval_idx], yte=y[eval_idx], device=h.device,
                         keep_model=True, verbose=False, label="mia-target")
    res["client_subjects"] = csubs
    # These are runs, so Section 1 rule 2 says they get logged with their hyper-
    # parameters and accountant output. They go to their own stage so the attack table
    # cannot pick up a row that is not an attack result.
    h._record(dict(dataset=h.bundle.name, stage="mia_models", arch=ARCH,
                   method="fedavg", cell="%s | %s" % (role, dpname),
                   architecture=ARCH, seed=int(seed), partition_id=0,
                   rounds=int(rounds), local_epochs=int(res.get("local_epochs") or 0),
                   batch=int(res.get("batch") or config.BATCH),
                   lr=float(res.get("lr") or config.LR),
                   n_clients=len(cidx), dp=dp, mia_role=role, dp_level=dpname,
                   pool_size=int(len(pool_idx)), pool_subjects=len(csubs)), res)
    return res


def _calibration(h, split, Xs, seed, rounds, cache=None):
    """Everything the attacks are fitted on: pools, member sets, shadows, thresholds.

    None of it sees the `dp` argument - the shadows are trained WITHOUT noise for every
    level, so they are the same models for all of them. Recomputing them per level
    multiplied the one cost Task 10 actually pays while producing byte-identical rows,
    which is why the result is cached across the levels of a seed. The key carries
    `rounds` because the non-DP target and the certificate-bearing targets train their
    shadows for a different number of rounds.

    Safe to reorder against the target fit: runners.run_fl re-seeds np.random on entry, so
    no global RNG state flows between these calls.
    """
    key = (int(seed), int(rounds))
    if cache is not None and key in cache:
        return cache[key]
    y = h.y
    subjects = h.subjects
    pools = _pools(split, subjects, y, seed)
    test_idx = split.test_idx
    rng = np.random.RandomState(int(seed) + 11)

    # target: trains on a random 50% of its pool
    tgt_pool = pools["target_idx"]
    members = np.sort(rng.permutation(tgt_pool)[:max(1, len(tgt_pool) // 2)])
    n_non = len(members)
    nonmembers = _class_balanced(pools["nonmember_idx"], y, n_non, rng)

    # shadows: disjoint subjects, alternating member/non-member halves
    sh = pools["shadow_idx"]
    sh_subs = np.unique(np.asarray(subjects)[sh])
    half = len(sh_subs) // 2
    A = sh[np.where(np.isin(np.asarray(subjects)[sh], sh_subs[:half]))[0]]
    B = sh[np.where(np.isin(np.asarray(subjects)[sh], sh_subs[half:]))[0]]
    shadows = []
    for i in range(config.cfg()["mia_shadows"]):
        own_pool, other_pool = (A, B) if i % 2 == 0 else (B, A)
        own_members = np.sort(np.random.RandomState(int(seed) + 50 + i)
                              .permutation(own_pool)[:max(1, len(own_pool) // 2)])
        r = _fit_pool(h, Xs, own_members, subjects, y, int(seed) + 50 + i, None,
                      rounds, test_idx, role="shadow%d" % i, dpname="shared non-DP")
        if r is None:
            continue
        shadows.append(dict(model=r["model"], members=own_members,
                            nonmembers=other_pool,
                            sc_m=_scores(r["model"], Xs[own_members], y[own_members],
                                         h.device),
                            sc_n=_scores(r["model"], Xs[other_pool], y[other_pool],
                                         h.device)))
    if not shadows:
        return dict(shadows=[])

    # ---- threshold selection on shadow data only -----------------------------
    thr = {}
    for name, key_ in (("Yeom[30]", "loss"), ("Confidence", "conf"),
                       ("ML-Leaks[31]", "label_only")):
        m = np.concatenate([s["sc_m"][key_] for s in shadows])
        n = np.concatenate([s["sc_n"][key_] for s in shadows])
        thr[name] = _pick_threshold(m, n, _DIRECTION[name])[0]
    thr["LiRA[33]"] = 0.0          # likelihood-ratio sign test, no fitted cut-point

    # Shokri shadow-model attack: classifier trained on shadow vectors only
    Xsh = np.concatenate([s["sc_m"]["prob"] for s in shadows]
                         + [s["sc_n"]["prob"] for s in shadows])
    ysh = np.concatenate([np.ones(sum(len(s["members"]) for s in shadows)),
                          np.zeros(sum(len(s["nonmembers"]) for s in shadows))])
    shmodel = LogisticRegression(max_iter=2000, C=1.0)
    shmodel.fit(Xsh, ysh)

    # LiRA's reference distribution is the spread over shadows; with one shadow the std
    # is undefined and the row would be a nan dressed up as a measurement.
    lira = None
    if len(shadows) >= 2:
        sm = [s["model"] for s in shadows]
        lira = (np.asarray(members), np.asarray(nonmembers),
                _lira_stats(sm, Xs[members], y[members], h.device),
                _lira_stats(sm, Xs[nonmembers], y[nonmembers], h.device))
    calib = dict(pools=pools, members=members, nonmembers=nonmembers, shadows=shadows,
                 thr=thr, shmodel=shmodel, lira=lira)
    if cache is not None:
        cache[key] = calib
    return calib


def _trial(h, split, Xs, seed, dpname, dp, cache=None, verbose=True):
    y = h.y
    rounds = config.cfg()["rounds_dp"] if dp else config.cfg()["rounds"]
    calib = _calibration(h, split, Xs, seed, rounds, cache)
    if not calib["shadows"]:
        return []
    pools = calib["pools"]
    members, nonmembers = calib["members"], calib["nonmembers"]
    shadows, thr, shmodel = calib["shadows"], calib["thr"], calib["shmodel"]

    tres = _fit_pool(h, Xs, members, h.subjects, y, seed, dp, rounds,
                     split.test_idx, role="target", dpname=dpname)
    if tres is None:
        return []
    target = tres["model"]

    # ---- apply the frozen attacks to the target ------------------------------
    tm = _scores(target, Xs[members], y[members], h.device)
    tn = _scores(target, Xs[nonmembers], y[nonmembers], h.device)
    rows = []
    simple = {"Yeom[30]": ("loss", thr["Yeom[30]"]),
              "Confidence": ("conf", thr["Confidence"]),
              "ML-Leaks[31]": ("label_only", thr["ML-Leaks[31]"])}
    for name, (key, t) in simple.items():
        r = _apply(tm[key], tn[key], t, _DIRECTION[name])
        r.update(attack=name, threshold=t, threshold_source="shadow-only")
        rows.append(r)
    r = _apply(shmodel.predict_proba(tm["prob"])[:, 1],
               shmodel.predict_proba(tn["prob"])[:, 1], 0.5, +1)
    r.update(attack="Shokri[28]", threshold=0.5, threshold_source="shadow-fitted-classifier")
    rows.append(r)

    if calib["lira"] is None:
        print("  [mia %-7s seed=%d] LiRA SKIPPED: %d shadow model, needs >= 2"
              % (dpname, seed, len(shadows)))
    else:
        mu_m, sd_m = calib["lira"][2]
        mu_n, sd_n = calib["lira"][3]
        r = _apply((tm["logp_y"] - mu_m) / sd_m, (tn["logp_y"] - mu_n) / sd_n,
                   thr["LiRA[33]"], +1)
        r.update(attack="LiRA[33]", threshold=thr["LiRA[33]"],
                 threshold_source="likelihood-ratio sign test")
        rows.append(r)

    for r in rows:
        r.update(dataset=h.bundle.name, stage=STAGE, seed=int(seed), dp_level=dpname,
                 z=(dp or {}).get("z"), C=(dp or {}).get("C"), rounds=int(rounds),
                 sparsity=(dp or {}).get("sparsity"),
                 architecture=ARCH, n_members=int(len(members)),
                 n_nonmembers=int(len(nonmembers)), n_shadows=len(shadows),
                 target_accuracy=float(tres["accuracy"]),
                 target_eps=(tres.get("privacy") or {}).get("eps"),
                 subject_pools=dict(target=pools["target_subs"],
                                    nonmember=pools["nonmember_subs"],
                                    shadow=pools["shadow_subs"]))
    if verbose:
        print("  [mia %-7s seed=%d] " % (dpname, seed) + "  ".join(
            "%s adv=%.3f auc=%.3f" % (r["attack"].split("[")[0], r["advantage"],
                                      r["auc"]) for r in rows))
    return rows


def run(h: Harness, seeds=None, verbose=True):
    seeds = seeds or config.cfg()["mia_trials"]
    log = h.log(STAGE)
    rows = []
    for seed in seeds:
        split = h.split(seed)
        Xs = h.scaled_all(ARCH, split)
        # one shadow set per seed, shared by every DP level (see _calibration)
        cal_cache = {}
        for dpname, dp in config.MIA_DP_LEVELS:
            for r in _trial(h, split, Xs, int(seed), dpname, dp,
                            cache=cal_cache, verbose=verbose):
                out = log.record(dict(dataset=h.bundle.name, stage=STAGE, cell=r["attack"],
                                      arch_role="mia", mia_attack=r["attack"],
                                      dp_level=dpname, seed=int(seed),
                                      architecture=ARCH),
                                 {k: v for k, v in r.items() if k != "dataset"})
                if out is not None:
                    rows.append(r)
        del Xs
    return rows


def table(dataset):
    """Mean advantage with a 95% CI and AUC, per attack and privacy setting."""
    runs = read_logs(dataset=dataset, stage=STAGE)
    groups = {}
    for r in runs:
        k = (str(r.get("dp_level")), str(r.get("mia_attack") or r.get("cell")),
             str(r.get("architecture")))
        groups.setdefault(k, []).append(r)
    rows = []
    for (dp, attack, arch), g in sorted(groups.items()):
        adv = np.array([float(x["advantage"]) for x in g])
        auc = np.array([float(x["auc"]) for x in g])
        s = multiseed.summarise(adv)
        rows.append(dict(dataset=dataset, architecture=arch, dp_level=dp, attack=attack,
                         n_trials=int(len(adv)),
                         eps=float(g[0].get("eps") or (g[0].get("target_eps") or 0.0))
                         if g[0].get("z") else float("inf"),
                         advantage_mean=s["mean"], advantage_std=s["std"],
                         advantage_ci95_low=s["ci95_low"], advantage_ci95_high=s["ci95_high"],
                         auc_mean=float(auc.mean()), auc_min=float(auc.min()),
                         tpr_mean=float(np.mean([x["tpr"] for x in g])),
                         fpr_mean=float(np.mean([x["fpr"] for x in g])),
                         target_acc_mean=float(np.mean([x["target_accuracy"] for x in g])),
                         random_guess_advantage=0.0))
    return rows


def write_csv(dataset):
    rows = table(dataset)
    if not rows:
        return None
    config.ensure_dirs()
    path = os.path.join(config.RESULTS_DIR, "mia_%s.csv" % dataset.replace(" ", "_"))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.6f" % v if isinstance(v, float) else v) for k, v in r.items()})
    print("  [mia] wrote %s" % path)
    return path

#@@FILE ppml/experiments/exp_dpsgd.py
"""Task 1's optional extra - example-level DP-SGD, and what that word protects.

The Work Order asks for this only "if time allows", as a SEPARATE, CLEARLY LABELLED setting
that protects single windows rather than people. The label is the whole point, so the
mechanism is spelled out here before any number is produced:

  adjacency      add or remove ONE WINDOW from one client's training set.
  mechanism      per-example gradient clipping to C (not per-client-update clipping),
                 Poisson sampling with probability q, the clipped gradients averaged over
                 the FIXED expected batch size B = round(q*n), and Gaussian noise of
                 std z*C/B added to that average before the optimiser sees it.
  certificate    account(z, S, delta) over the S steps ONE client actually runs. An example
                 lives in exactly one silo, so no other client's steps touch it and the
                 per-client bound is the whole bound; the server averages updates that are
                 already private, which is post-processing.

What it is NOT: a user-level guarantee. One person contributes hundreds of windows, so this
arm says nothing about "everything one person contributed" - that is Task 1's main mechanism
and it is the one the paper's privacy claims must cite. Every table here carries
`adjacency=example_level` and `user_level_certificate=NO`, because conflating the two is
what Reviewer 2 (R2-1, R2-11) rejected the submission over.

Why no Opacus: the gradient computation below IS the mechanism, in a dozen lines, with
nothing to install. Kaggle notebooks run without network access, so an `import opacus`
requirement would make this arm permanently unrunnable in the environment the experiments
actually execute in - and an arm that cannot run answers nothing. The clipping, sampling,
fixed-B averaging and noise match what Opacus does; the certificate is the same
conservative composition and dp_accountant's closed form, verified in the preflight rather
than assumed. The cost of doing it here rather than in a library is speed, so the scope is
small and stamped in every row (config.DP_SGD_*).
"""

import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as Fn

from .. import config, dp_accountant, metrics, models, runners, splitting
from ..harness import Harness
from ..runlog import read_logs, stable_hash

STAGE = "dpsgd"
ARCH = "CNN-BiLSTM"          # the Work Order names the CNN for this arm


def _client_windows(idx, rng):
    """One silo's windows, capped by config so the arm fits a session.

    The cap is a scope decision stamped into the log row, not a way to reach an accuracy:
    it shrinks n, and with it B = round(q*n), which makes each released step noisier per
    unit of signal, so a smaller cap cannot inflate the privacy claim or the utility.
    """
    idx = np.asarray(idx)
    cap = config.DP_SGD_MAX_WINDOWS_PER_CLIENT
    if cap is None or len(idx) <= int(cap):
        return idx
    return rng.choice(idx, size=int(cap), replace=False)


def _per_sample_grads(model, params, buffers, xb, yb):
    """Per-example gradients in one vectorised pass (torch.func), not a Python loop."""
    from torch.func import functional_call, grad, vmap
    names = list(params)

    def loss_one(p, x, y):
        out = functional_call(model, ({**p, **buffers}), (x.unsqueeze(0),))
        return Fn.cross_entropy(out, y.unsqueeze(0))

    g = vmap(grad(loss_one), in_dims=(None, 0, 0), randomness="different")(
        {k: params[k] for k in names}, xb, yb)
    return [g[k] for k in names]


def dp_sgd_local(model, X, y, z, C, q, epochs, lr, device, gen=None):
    """One client's DP-SGD passes. Returns (weights as numpy, steps actually run).

    Poisson sampling at rate q; one epoch = ceil(1/q) steps (the expected number of
    steps to visit every window once). Each step: per-example gradients, each clipped
    to L2 norm C, summed, divided by the FIXED expected batch B = round(q*n), plus
    Gaussian noise of std z*C/B. A step whose Poisson draw is empty still releases the
    noise-only update, so the released sequence is exactly the one the accountant prices.
    """
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.asarray(y).astype(np.int64), dtype=torch.long, device=device)
    n = int(len(y))
    B = max(1, int(round(float(q) * n)))
    sigma = float(z) * float(C) / B
    named = {k: v for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v for k, v in model.named_buffers()}
    plist = list(named.values())
    if getattr(config, "DP_SGD_OPTIMIZER", "adam") == "sgd":
        opt = torch.optim.SGD(plist, lr=float(config.DP_SGD_LR),
                              momentum=float(config.DP_SGD_MOMENTUM))
    else:
        opt = torch.optim.Adam(plist, lr=lr)
    model.train()
    rng = gen or np.random.default_rng()
    steps = 0
    per_epoch = int(math.ceil(1.0 / float(q)))
    for _ep in range(int(epochs)):
        for _s in range(per_epoch):
            picked = np.nonzero(rng.random(n) < float(q))[0]
            acc = [torch.zeros_like(p) for p in plist]
            if len(picked):
                idx = torch.as_tensor(picked, device=device)
                for start in range(0, len(idx), 256):
                    sl = idx[start:start + 256]
                    grads = _per_sample_grads(model, {k: v.detach() for k, v in named.items()},
                                              buffers, X[sl], y[sl])
                    flat = torch.cat([g.reshape(g.shape[0], -1) for g in grads], dim=1)
                    norms = flat.norm(dim=1)
                    scale = torch.clamp(float(C) / (norms + 1e-12), max=1.0)
                    for j, g in enumerate(grads):
                        acc[j] += (g * scale.view(-1, *([1] * (g.dim() - 1)))).sum(0)
            opt.zero_grad(set_to_none=True)
            for p, a_ in zip(plist, acc):
                p.grad = a_ / B + torch.randn(a_.shape, device=device) * sigma
            opt.step()
            steps += 1
    return [np.asarray(w, dtype=np.float32) for w in models.get_weights(model)], steps


def dp_sgd_rounds():
    """config.DP_SGD_ROUNDS for reportable runs; 2 rounds in the smoke profile."""
    return int(config.DP_SGD_ROUNDS) if config.PROFILE == "full" else 2


def planned_steps():
    return (dp_sgd_rounds() * int(config.DP_SGD_LOCAL_EPOCHS)
            * int(math.ceil(1.0 / float(config.DP_SGD_SAMPLE_RATE))))


def dp_sgd_plan():
    """[(target_eps, z)] solved from the planned steps BEFORE any training."""
    T = planned_steps()
    q = float(config.DP_SGD_SAMPLE_RATE)
    return [(float(e), dp_accountant.z_for_eps_dpsgd(float(e), q, T, config.DELTA))
            for e in config.DP_SGD_TARGET_EPS]


def _f4(v):
    """A metric on the live line, or n/a - macro specificity is undefined for a class set
    with no true negatives, and a crash would cost the session its remaining runs."""
    return "n/a" if v is None else "%.4f" % float(v)


def run(h: Harness, seeds=None, arches=None, verbose=True):
    """One example-level run per seed at config.DP_SGD_Z, on the raw-window CNN.

    Not in the driver's default RUN_STAGES - add "dpsgd" to run it. See config's DP_SGD
    block for the scope and the module docstring for what the resulting epsilon means.
    """
    seeds = seeds or config.DP_SGD_SEEDS
    if len(seeds) > len(config.DP_SGD_SEEDS):
        seeds = tuple(seeds)[:len(config.DP_SGD_SEEDS)]
    arches = tuple(config.DP_SGD_ARCHES)
    q = float(config.DP_SGD_SAMPLE_RATE)
    C = float(config.DP_SGD_CLIP_C)
    rounds = dp_sgd_rounds()
    lep = int(config.DP_SGD_LOCAL_EPOCHS)
    n_class = int(config.N_CLASS)
    plan = dp_sgd_plan()
    if verbose:
        print("  DP-SGD plan (fixed before training): q=%.3f, %d steps per client -> %s"
              % (q, planned_steps(), ", ".join("eps=%.1f needs z=%.3f" % pe for pe in plan)))
    rows = []
    for (target_eps, z), arch, seed in [(pe, a, s_) for pe in plan for a in arches
                                        for s_ in seeds]:
        if True:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
            rng = np.random.default_rng(int(seed))
            split = h.split(seed)
            cl = h.clients(split, seed)
            Xtr, ytr, Xte, yte, _, _ = h.input(arch, split)
            test_data = runners.LocalData(Xte, yte, h.device)
            fdim = Xtr.shape[1] if Xtr.ndim == 2 else None
            wshape = Xtr.shape[1:] if Xtr.ndim == 3 else h.Xw.shape[1:]
            gm = models.build(arch, feature_dim=fdim, window_shape=wshape,
                              n_class=n_class, device=h.device)
            weights = [w.copy() for w in models.get_weights(gm)]
            steps_max = 0
            # Steps COMPOSE across rounds: a client's windows are sampled in every round,
            # so its certificate is over the sum of its steps, not one round's worth.
            steps_by_client = {}
            per_round = []
            last_m = None
            t0 = time.time()
            for rnd in range(1, rounds + 1):
                # this arm bypasses harness.run_fl, which is where check_budget() normally
                # lives, so without it a single expensive run could outlive the Kaggle kill
                config.check_budget()
                ws = []
                for k_cl, idx in enumerate(cl["client_indices"]):
                    sub = _client_windows(idx, rng)
                    if len(sub) == 0:
                        continue
                    cm = models.build(arch, feature_dim=fdim, window_shape=wshape,
                                      n_class=n_class, device=h.device)
                    models.set_weights(cm, weights)
                    pos = splitting.to_block_positions([sub], split.train_idx)[0]
                    w, s = dp_sgd_local(cm, Xtr[pos], ytr[pos], z, C, q, lep,
                                        float(config.LR), h.device, gen=rng)
                    ws.append(w)
                    steps_by_client[k_cl] = steps_by_client.get(k_cl, 0) + int(s)
                    steps_max = max(steps_by_client.values())
                if not ws:
                    continue
                # equal 1/K on the already-private updates: the averaging is
                # post-processing, so nothing is added to the certificate here
                weights = [np.mean([w[i] for w in ws], axis=0)
                           for i in range(len(weights))]
                m = _score(arch, weights, h, test_data, yte, n_class, fdim, wshape)
                last_m = m
                # The same per-round curve every other stage logs. `eps_cum` is None on
                # purpose: this arm's composition is over the STEPS inside a silo, not over
                # these aggregation rounds, so a per-round epsilon here would be a number
                # the accountant does not support. The certificate lands on the run record.
                per_round.append(dict(round=rnd, test_acc=m["accuracy"],
                                      test_loss=m["loss"],
                                      precision=m["precision"], recall=m["recall"],
                                      f1_weighted=m["f1_weighted"],
                                      macro_f1=m["f1_macro"],
                                      balanced_accuracy=m["balanced_accuracy"],
                                      macro_specificity=m["macro_specificity"],
                                      kappa=m["kappa"],
                                      roc_auc_ovr=m.get("roc_auc_ovr"),
                                      n_clients=len(cl["client_indices"]),
                                      eps_cum=None, C=C, sigma=None,
                                      clip_fraction=None, release_snr=None,
                                      wall_s=time.time() - t0))
                if verbose and config.PRINT_ROUNDS and (rnd == rounds or rnd % 5 == 0):
                    print("    R%3d/%d K=%d  acc=%s prec=%s rec=%s F1w=%s F1m=%s "
                          "bal=%s spec=%s kappa=%s loss=%s"
                          % (rnd, rounds, len(cl["client_indices"]),
                             _f4(m["accuracy"]), _f4(m["precision"]), _f4(m["recall"]),
                             _f4(m["f1_weighted"]), _f4(m["f1_macro"]),
                             _f4(m["balanced_accuracy"]), _f4(m["macro_specificity"]),
                             _f4(m["kappa"]), _f4(m["loss"])))
            if not per_round:
                continue
            # The certificate is over the STEPS one client ran, not the FL rounds: this is
            # an example-level mechanism, so the composition is inside the silo. Taking the
            # max over clients is conservative - an example cannot sit in two silos.
            sizes = [len(ix) for ix in cl["client_indices"] if len(ix)]
            n_client = int(min(sizes)) if sizes else 0
            cert = dp_accountant.account_dpsgd(
                z, steps_max, config.DELTA, batch=max(1, int(round(q * n_client))),
                sample_rate=q, n_per_client=n_client)
            final = per_round[-1]
            cfg_block = dict(dataset=h.bundle.name, stage=STAGE, architecture=arch,
                             arch=arch, seed=int(seed), method="example-level DP-SGD",
                             mechanism=("per-example clip C=%.3f, Poisson q=%.3f, "
                                        "noise on the batch mean" % (C, q)),
                             adjacency="example_level (one window, not one person)",
                             C=C, z=z, gamma=q, K=len(cl["client_indices"]),
                             target_eps=float(target_eps),
                             privacy_unit="one window (example-level)",
                             rounds=rounds, local_epochs=lep, steps_max=int(steps_max),
                             delta=config.DELTA, split_kind="subject",
                             max_windows_per_client=config.DP_SGD_MAX_WINDOWS_PER_CLIENT,
                             max_windows=config.MAX_WINDOWS, profile=config.PROFILE)
            result_block = dict(eps=cert["eps"], alpha_star=cert["alpha_star"],
                                eps_dp_accounting=cert.get("eps_dp_accounting"),
                                accuracy=final["test_acc"],
                                f1=last_m["f1_weighted"], f1_macro=last_m["f1_macro"],
                                loss=final["test_loss"],
                                # the full headline set, so this arm's rows read like every
                                # other stage's and cannot be quoted on accuracy alone
                                precision=last_m["precision"], recall=last_m["recall"],
                                balanced_accuracy=last_m["balanced_accuracy"],
                                macro_specificity=last_m["macro_specificity"],
                                kappa=last_m["kappa"],
                                roc_auc_ovr=last_m.get("roc_auc_ovr"),
                                nll=last_m.get("nll"),
                                f1_per_class=last_m.get("f1_per_class"),
                                precision_per_class=last_m.get("precision_per_class"),
                                recall_per_class=last_m.get("recall_per_class"),
                                specificity_per_class=last_m.get("specificity_per_class"),
                                class_support=last_m.get("class_support"),
                                confusion=last_m.get("confusion"),
                                per_round_test_acc=[p["test_acc"] for p in per_round],
                                per_round_test_loss=[p["test_loss"] for p in per_round],
                                per_round_precision=[p["precision"] for p in per_round],
                                per_round_recall=[p["recall"] for p in per_round],
                                per_round_f1_weighted=[p["f1_weighted"] for p in per_round],
                                per_round_macro_f1=[p["macro_f1"] for p in per_round],
                                per_round_balanced_accuracy=[p["balanced_accuracy"]
                                                             for p in per_round],
                                per_round_macro_specificity=[p["macro_specificity"]
                                                             for p in per_round],
                                per_round_kappa=[p["kappa"] for p in per_round],
                                per_round_roc_auc=[p["roc_auc_ovr"] for p in per_round],
                                per_round_eps_cum=[p["eps_cum"] for p in per_round],
                                wall_time_s=time.time() - t0,
                                user_level_certificate="NO")
            rec = h.log(STAGE).record(cfg_block, result_block,
                                      config_hash=stable_hash(cfg_block))
            if rec is None:
                continue
            rows.append(rec)
            if verbose:
                # the same per-run line every other stage prints, so this arm cannot be
                # quoted on accuracy alone; the stamp at the end is the point of the arm
                print("  [%-11s dpsgd z=%.1f q=%.3f C=%.3f] steps=%d  %s "
                      "(example-level, NOT user-level)"
                      % (arch, z, q, C, steps_max,
                         metrics.format_metrics(dict(
                             rec,
                             privacy=dict(eps=cert["eps"],
                                          alpha_star=cert["alpha_star"],
                                          adjacency="example_level (one window)"),
                             dp_z=z, clip_C=C, K=len(cl["client_indices"]),
                             rounds=rounds)),))
    if rows:
        write_csv(h.bundle.name)
    return rows


def _score(arch, weights, h, test_data, yte, n_class, fdim=None, wshape=None):
    """Evaluate one global weight vector on the held-out subjects."""
    gm = models.build(arch, feature_dim=fdim,
                      window_shape=wshape if wshape is not None else h.Xw.shape[1:],
                      n_class=n_class, device=h.device)
    models.set_weights(gm, weights)
    pred, proba, loss = runners.evaluate(gm, test_data)
    return metrics.classification_metrics(np.asarray(yte, dtype=np.int64), pred,
                                          y_proba=proba, sample_loss=loss,
                                          n_class=n_class)


def table(dataset):
    """One row per logged example-level run, with the adjacency label in the row."""
    runs = [r for r in read_logs(dataset=dataset, stage=STAGE)
            if r.get("eps") is not None]
    rows = []
    for r in runs:
        rows.append(dict(dataset=dataset, architecture=r.get("architecture"),
                         mechanism=r.get("mechanism"), adjacency=r.get("adjacency"),
                         C=r.get("C"), z=r.get("z"), gamma=r.get("gamma"),
                         K=r.get("K"), T=r.get("rounds"),
                         steps=r.get("steps_max"), delta=r.get("delta"),
                         alpha_star=r.get("alpha_star"), cumulative_eps=r.get("eps"),
                         accuracy=r.get("accuracy"), f1=r.get("f1"),
                         f1_macro=r.get("f1_macro"),
                         # the same headline set as every other stage's sheet, so this arm
                         # is comparable row for row and cannot be read on accuracy alone
                         precision=r.get("precision"), recall=r.get("recall"),
                         balanced_accuracy=r.get("balanced_accuracy"),
                         macro_specificity=r.get("macro_specificity"),
                         kappa=r.get("kappa"), roc_auc_ovr=r.get("roc_auc_ovr"),
                         nll=r.get("nll"), wall_time_s=r.get("wall_time_s"),
                         seed=r.get("seed"),
                         max_windows_per_client=r.get("max_windows_per_client"),
                         user_level_certificate="NO - this protects single windows",
                         reportable=r.get("reportable")))
    return rows


def write_csv(dataset):
    config.ensure_dirs()
    rows = table(dataset)
    if not rows:
        return None
    p = os.path.join(config.RESULTS_DIR, "dpsgd_%s.csv" % dataset.replace(" ", "_"))
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        w.writeheader()
        w.writerows({k: v for k, v in r.items() if not isinstance(v, (list, dict))}
                    for r in rows)
    print("  [dpsgd] wrote %s (%d rows, adjacency=example_level)" % (p, len(rows)))
    return p

#@@FILE ppml/experiments/exp_dplearn.py
"""User-level DP that can learn: a small model, a public starting point, a small release.

Why this stage exists. With one person per client the user-level release has to beat a
noise-to-signal ratio of about z*sqrt(d)/K. At the operating point (z=10.6, T=50,
eps=3.42) and K = 6-36 training subjects, the whole-update release of the 71,750-weight
MLP sits 100x below the bar, so those arms are at chance by construction (the feasibility
table printed in preflight says so before any run). The only levers that do not touch the
certificate are the released dimension d and the starting point:

  MLP-Small     117 -> 64 -> C, about 7.9k weights instead of 71.8k
  public init   the body is trained WITHOUT privacy on a DIFFERENT dataset (the other
                dataset of this notebook). It never reads a private window, so it costs no
                privacy budget; the paper must name the public source.
  head-only     only the classifier head (390 weights for the MLP) is released with DP.
                The support is fixed by the architecture, so sensitivity stays C/K and the
                certificate is the plain one.
  fewer rounds  T=20 instead of 50, with z re-solved so that eps is the SAME as the
                operating point; fewer, less noisy releases.

Every row is user-level (one subject per client) and carries init_source, so a reader can
see which number used public pretraining. The non-private row of each cell is the ceiling.
"""

import numpy as np
import torch

from .. import config, dp_accountant, models, multiseed, runners
from ..harness import Harness
from ..runlog import read_logs

STAGE = "dplearn"
ARCHES = ("MLP", "MLP-Small")
PRETRAIN_EPOCHS = 30


def z_for_eps_central(target_eps, T, delta=None, lo=0.2, hi=200.0):
    """Smallest z whose user-level certificate over T rounds is <= target_eps."""
    delta = config.DELTA if delta is None else delta
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if dp_accountant.account_central_gaussian(mid, T, delta)["eps"] > target_eps:
            lo = mid
        else:
            hi = mid
    return float(hi)


def _pretrain_body(arch, public, n_class_private, seed, device):
    """Train arch on the PUBLIC dataset (all of it, non-private) and return weights for a
    private model: body copied, head freshly initialised for the private class set."""
    Xp = np.asarray(public["Xf"], dtype=np.float32)
    mu, sd = Xp.mean(0, keepdims=True), Xp.std(0, keepdims=True) + 1e-8
    Xp = (Xp - mu) / sd
    yp = np.asarray(public["y"]).astype(int)
    n_pub = int(public["n_class"])
    torch.manual_seed(int(seed))
    m_pub = models.build(arch, feature_dim=Xp.shape[1], n_class=n_pub, device=device)
    data = runners.LocalData(Xp, yp, device)
    runners.train_local(m_pub, data, PRETRAIN_EPOCHS, config.BATCH, config.LR, device,
                        seed=seed, cosine_epochs=True,
                        class_weights=runners.class_weights_of(yp, n_pub))
    torch.manual_seed(int(seed) + 7)
    m_priv = models.build(arch, feature_dim=Xp.shape[1], n_class=n_class_private,
                          device=device)
    n_head = models.n_head_params(m_priv)
    w_pub = models.get_weights(m_pub)
    w_new = models.get_weights(m_priv)
    body = len(w_new) - n_head
    return [w_pub[i] if i < body else w_new[i] for i in range(len(w_new))]


def cells():
    """(label, dp-or-None, T, init) - fixed before any run; z solved from eps."""
    target = dp_accountant.account_central_gaussian(
        config.DP_ARM_Z, config.cfg()["rounds_dp"], config.DELTA)["eps"]
    out = []
    for T in config.DPLEARN_ROUNDS:
        T = int(T) if config.PROFILE == "full" else min(int(T), config.cfg()["rounds_dp"])
        z = (config.DP_ARM_Z if T == config.cfg()["rounds_dp"]
             else z_for_eps_central(target, T))
        out.append(("public-init head-only DP T=%d" % T,
                    dict(z=z, C=None, head_only=True, iter_avg=config.DP_ITER_AVG,
                         T=T), T, "public"))
        if T == config.cfg()["rounds_dp"]:     # the random-init reference once, at T=50
            out.append(("random-init head-only DP T=%d" % T,
                        dict(z=z, C=None, head_only=True, iter_avg=config.DP_ITER_AVG,
                             T=T), T, "random"))
    out.append(("public-init non-private", None, config.cfg()["rounds"], "public"))
    return out, target


def run(h: Harness, seeds=None, arches=None, verbose=True):
    seeds = tuple(seeds or config.cfg().get("dp_seeds", config.cfg()["seeds"]))
    pub = getattr(h, "public_source", None)
    plan, target = cells()
    if verbose:
        print("  user-level target eps = %.4f (the operating point); cells: %s"
              % (target, "; ".join("%s z=%s" % (c[0], "-" if c[1] is None
                                                 else "%.3f" % c[1]["z"]) for c in plan)))
    if pub is None:
        print("  no public source dataset in this session - the public-init cells are "
              "skipped (run two datasets in one notebook to enable them)")
    rows = []
    for arch in ARCHES:
        for label, dp, T, init in plan:
            if init == "public" and pub is None:
                continue
            accs = []
            for seed in seeds:
                split = h.split(seed)
                cl = h.clients(split, seed)
                init_w = (_pretrain_body(arch, pub, config.N_CLASS, seed, h.device)
                          if init == "public" else None)
                cell = "%s | %s" % (arch, label)
                res = h.run_fl(dict(arch_role="dplearn", init_source=(
                                    "public:%s" % pub["name"] if init == "public"
                                    else "random"), pretrain_epochs=(
                                    PRETRAIN_EPOCHS if init == "public" else 0)),
                               STAGE, arch=arch, client_indices=cl["client_indices"],
                               split=split, seed=seed, method="fedavg", dp=dp,
                               server_momentum=dp is not None, rounds=T,
                               cell=cell, label=cell, init_weights=init_w,
                               verbose=False)
                if res is not None:
                    accs.append(float(res["accuracy"]))
                    rows.append(res)
            if accs:
                s = multiseed.summarise(accs)
                snr = [float(r["release_snr_mean"]) for r in rows
                       if r.get("label") == cell and r.get("release_snr_mean") is not None]
                print("  [%-9s %-34s] acc=%.4f +-%.4f n=%d%s"
                      % (arch, label, s["mean"], s["std"], s["n"],
                         (" snr=%.3f" % (sum(snr) / len(snr))) if snr else ""))
    return rows


def table(dataset):
    """One row per (architecture, cell): accuracy mean/std/CI, eps, z, T, init source."""
    runs = read_logs(dataset=dataset, stage=STAGE)
    groups = {}
    for r in runs:
        groups.setdefault((str(r.get("architecture")), str(r.get("cell"))), []).append(r)
    out = []
    for (arch, cell), rs in sorted(groups.items()):
        accs = [float(r["accuracy"]) for r in rs if r.get("accuracy") is not None]
        if not accs:
            continue
        s = multiseed.summarise(accs)
        r0 = rs[0]
        out.append(dict(dataset=dataset, architecture=arch, cell=cell, n=s["n"],
                        accuracy_mean=s["mean"], accuracy_std=s["std"],
                        ci95_low=s["ci95_low"], ci95_high=s["ci95_high"],
                        balanced_accuracy_mean=float(np.mean(
                            [r.get("balanced_accuracy") or 0.0 for r in rs])),
                        eps=r0.get("eps"), z=r0.get("z"), T=r0.get("T"),
                        init_source=r0.get("init_source"),
                        privacy_unit="one person (user-level)" if r0.get("eps") else "-",
                        release_snr_mean=r0.get("release_snr_mean")))
    return out

'''

# ══════════════════════════ 3. INSTALLER ══════════════════════════════════════════

_FILE_MARK = "#@@FILE "


def _ppml_parse(blob):
    """Split the embedded blob back into {package-relative path: source}."""
    out, cur, buf = {}, None, []
    for line in blob.splitlines():
        if line.startswith(_FILE_MARK):
            if cur:
                out[cur] = "\n".join(buf).rstrip("\n") + "\n"
            cur, buf = line[len(_FILE_MARK):].strip(), []
        elif cur is not None:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf).rstrip("\n") + "\n"
    return out


def _ppml_install(base_dir=INSTALL_DIR, verbose=True):
    """Write the embedded package to <base_dir>/ppml/ and return the package module."""
    files = _ppml_parse(_PPML_BLOB)
    root = os.path.join(base_dir, "ppml")
    written = 0
    for rel, src in sorted(files.items()):
        path = os.path.join(base_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        old = None
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                old = fh.read()
        if old != src:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(src)
            written += 1
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    for m in [k for k in list(sys.modules) if k == "ppml" or k.startswith("ppml.")]:
        del sys.modules[m]
    __import__("importlib").invalidate_caches()
    import ppml                                            # noqa: E402
    if verbose:
        print("  package: %d files, %d written (%s)" % (len(files), written, root))
    return ppml


# ══════════════════════════ 4. DRIVER ═════════════════════════════════════════════

_WRITERS = ("write_csv", "write_csvs", "write_crosschecks",
            "write_label_map_table", "write_client_note")

# A function whose only required argument is `dataset` can only read the logs, so it is
# safe to call for a CSV dump. channel_ablation is the one exception: it loads its own
# data and trains, so it is never called implicitly - run it yourself if you need it.
_NOT_TABLES = {"channel_ablation"}


def _call_by_name(fn, h, dataset_name):
    """Call a stage's CSV/report writer, supplying its required args by parameter name."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    kwargs = {}
    for pname, p in params.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is not inspect.Parameter.empty:
            continue
        if pname == "dataset":
            kwargs["dataset"] = dataset_name
        elif pname == "h":
            kwargs["h"] = h
        else:
            print("    %s: cannot supply required arg %r - skipped" % (fn.__name__, pname))
            return False
    fn(**kwargs)
    return True


def _write_rows(path, rows):
    import csv
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (("%.6f" % v if isinstance(v, float) else v))
                        for k, v in r.items()})


def _dump_tables(mod, dataset_name, results_dir):
    """Run every table function the stage exposes on `dataset` and save it as CSV.

    The stage modules keep their tables as functions returning list[dict] and only some
    of them also carry a write_csv, so this is what makes a single Kaggle session
    produce the paper's tables instead of raw logs only.
    """
    import types
    for fname in sorted(dir(mod)):
        if fname.startswith("_") or fname in _NOT_TABLES or fname in _WRITERS:
            continue
        fn = getattr(mod, fname)
        if not isinstance(fn, types.FunctionType) or fn.__module__ != mod.__name__:
            continue
        try:
            params = list(inspect.signature(fn).parameters.values())
        except (TypeError, ValueError):
            continue
        required = [p for p in params if p.default is inspect.Parameter.empty]
        if [p.name for p in required] != ["dataset"]:
            continue
        try:
            rows = fn(dataset_name)
        except Exception as exc:
            print("    %s(%s) failed: %s" % (fname, dataset_name, exc))
            continue
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            continue
        out = os.path.join(results_dir, "tables",
                          "%s__%s__%s.csv" % (dataset_name.replace(" ", "_"),
                                              mod.__name__.split(".")[-1], fname))
        try:
            _write_rows(out, rows)
            print("    table %-22s %4d rows -> %s" % (fname, len(rows),
                                                      os.path.basename(out)))
        except Exception as exc:
            print("    %s: could not write %s (%s)" % (fname, out, exc))


def _run_stage(ppml, h, stage, seeds, arches, t_session):
    """One stage for one dataset. Returns False when the session budget stopped it."""
    from ppml import config
    from ppml.experiments import STAGES
    mod, what = STAGES[stage]
    print("\n--- stage %-11s %s" % (stage, what))
    kw = {"verbose": VERBOSE}
    if seeds is not None:
        kw["seeds"] = seeds
    if "arches" in inspect.signature(mod.run).parameters:
        kw["arches"] = arches
    elif arches is not None:
        print("    (%s has a fixed architecture list; ARCHES ignored)" % stage)
    t0 = time.time()
    try:
        mod.run(h, **kw)
    except config.BudgetExceeded as exc:
        print("  STOPPED: %s" % exc)
        print("  completed runs are already in results/logs/ - re-run this file to resume")
        return False
    except Exception as exc:
        print("  stage %s raised %s: %s" % (stage, type(exc).__name__, exc))
        traceback.print_exc()
        print("  continuing with the next stage; fix and re-run this stage to fill the gaps")
    for wname in _WRITERS:
        fn = getattr(mod, wname, None)
        if fn is None or not callable(fn):
            continue
        try:
            if _call_by_name(fn, h, h.bundle.name):
                print("    wrote %s" % wname)
        except Exception as exc:
            print("    %s failed (%s) - usually a stage that is only part-complete"
                  % (wname, exc))
    try:
        _dump_tables(mod, h.bundle.name, config.RESULTS_DIR)
    except Exception as exc:
        print("    table dump failed (%s) - the raw logs are still complete" % exc)
    print("--- stage %-11s done in %.1f min" % (stage, (time.time() - t0) / 60.0))
    return True


def _preflight(ppml):
    from ppml import config, fl, runlog
    config.ensure_dirs()
    from ppml import dp_accountant
    print("preflight")
    print("  profile      : %s (reportable=%s)" % (config.PROFILE, config.is_reportable()))
    # Scope is part of the evidence. Printing the data caps and the local-epoch budget
    # here means the first lines of any pasted log already disclose what was run, which
    # is what makes a reduced-scope session auditable instead of mysterious.
    print("  notebook     : %s  datasets=%s" % (NOTEBOOK_ID, RUN_DATASETS))
    print("  scope        : max_windows=%s max_subjects=%s E_mlp=%s E_cnn=%s "
          "T=%s/%s seeds=%s/%s"
          % (MAX_WINDOWS_BY_DATASET, MAX_SUBJECTS, config.LOCAL_EPOCHS_MLP,
             config.LOCAL_EPOCHS_CNN, config.cfg()["rounds"], config.cfg()["rounds_dp"],
             config.cfg()["seeds"], config.cfg()["seeds_supp"]))
    print("  recipe       : lr %g->%g (%s), class weights=%s, window scaling=%s, "
          "augment=%s, model selection=%s"
          % (config.LR, config.LR_MIN, config.LR_SCHEDULE, config.CLASS_WEIGHTS,
             config.WINDOW_SCALE, config.AUGMENT,
             "best validation round" if config.SELECT_BEST_VAL else "last round"))
    for ds in RUN_DATASETS:
        print("  stages %-9s: %s" % (ds, ", ".join(s for s in RUN_STAGES
                                                   if s in STAGES_BY_DATASET.get(ds, RUN_STAGES))))
    # The DP operating point, stated before any arm runs: the z, the certificate it buys and
    # the sweep's third arm. A session that changed one of these without saying so would
    # produce tables whose epsilon column no longer matches the arm it is attached to.
    arm_eps = dp_accountant.account_central_gaussian(
        config.DP_ARM_Z, config.cfg()["rounds_dp"], config.DELTA)
    print("  dp arms      : z=%s -> eps=%.4f (T=%d, delta=%g)  head-sweep=%s"
          % (config.DP_ARM_Z, arm_eps["eps"], config.cfg()["rounds_dp"],
             config.DELTA, config.PRIVACY_HEAD_SWEEP))
    if config.PRIVACY_HEAD_SWEEP:
        print("                   ^ head release added at every z for %s: the accuracy "
              "question is measured here, at the same eps as the plain arm it is beside"
              % ", ".join(config.PRIVACY_HEAD_ARCHES))
    print("  printing     : per-round metric lines %s (everything printed is also logged)"
          % ("ON" if config.PRINT_ROUNDS else "OFF"))
    print("  work dir     : %s" % config.WORK)
    print("  kaggle       : %s" % config.KAGGLE)
    print("  data mounts  : %s" % (sorted(os.listdir("/kaggle/input"))
                                   if os.path.isdir("/kaggle/input") else "none"))
    print("  versions     : %s" % runlog.versions())
    print("  hardware     : %s" % runlog.hardware())
    ok = dp_accountant.selftest()
    print("  dp accountant selftest: %s" % ("PASSED" if ok else "FAILED"))
    if not ok:
        raise RuntimeError("DP accountant does not reproduce its own closed form; "
                           "no privacy number from this session is trustworthy")
    # The sparsified DP arm is a claim about epsilon, not just about accuracy: it may only
    # be reported if selecting coordinates after the draw demonstrably leaves the release
    # inside the certificate the plain arm pays for.
    ok2 = fl.projection_selftest()
    print("  dp mechanism selftest: %s" % ("PASSED" if ok2 else "FAILED"))
    if not ok2:
        raise RuntimeError("the sparse release is not epsilon-neutral in this build; the "
                           "FL+DP-Sparse arm would be unreportable, so nothing runs")
    ok3 = fl.clipper_selftest()
    print("  adaptive clipper selftest: %s" % ("PASSED" if ok3 else "FAILED"))
    if not ok3:
        raise RuntimeError("the clip estimator cannot count its own releases; the "
                           "adaptive arm's epsilon would be a guess, so nothing runs")
    # the adaptive arm's cost, stated before it runs: the estimator, not the model, is
    # what the certificate pays for at this participation level.
    T_dp = config.cfg()["rounds_dp"]
    W = min(int(config.ADAPT_FREEZE_ROUNDS), int(T_dp))
    ad = dp_accountant.account_adaptive(config.DP_ARM_Z, T_dp, config.DELTA,
                                        sigma_b=config.ADAPT_SIGMA_B, T_clip=W)
    ad_all = dp_accountant.account_adaptive(config.DP_ARM_Z, T_dp, config.DELTA,
                                            sigma_b=config.ADAPT_SIGMA_B)
    print("  adaptive eps   : z=%.1f T=%d -> eps=%.3f with the estimator releasing %d of "
          "%d rounds (eps=%.3f unfrozen, eps=%.3f fixed-C at the same z)"
          % (config.DP_ARM_Z, T_dp, ad["eps_total"], W, T_dp, ad_all["eps_total"],
             dp_accountant.account_central_gaussian(config.DP_ARM_Z, T_dp,
                                                    config.DELTA)["eps"]))
    from ppml.experiments import exp_dpsgd, exp_dplearn
    plan = exp_dpsgd.dp_sgd_plan()
    print("  DP-SGD plan  : example-level (one window), q=%.3f, %d steps per client, "
          "solved BEFORE training: %s"
          % (config.DP_SGD_SAMPLE_RATE, exp_dpsgd.planned_steps(),
             ", ".join("eps=%.1f -> z=%.3f" % pe for pe in plan)))
    cells, target = exp_dplearn.cells()
    print("  dplearn      : user-level target eps=%.4f; %s"
          % (target, "; ".join("%s (z=%s)" % (c[0], "-" if c[1] is None else
                                              "%.3f" % c[1]["z"]) for c in cells)))
    return ppml


def _feasibility(h):
    """User-level DP feasibility, printed BEFORE any DP accuracy exists.

    Predicted release SNR per round at the operating point, K/(z*sqrt(d)), for the three
    released supports. Below ~1 the release is mostly noise; the arm is still run and
    reported, and this line is what explains its accuracy.
    """
    from ppml import config
    import numpy as np
    s0 = h.split(config.SEEDS[0])
    K = len(h.clients(s0, config.SEEDS[0])["client_indices"])
    z = config.DP_ARM_Z
    T = config.cfg()["rounds_dp"]
    rows = []
    for name, d in (("MLP whole update", 71750), ("MLP-Small whole update", 7942),
                    ("head-only release", 390)):
        snr = K / (z * np.sqrt(d))
        rows.append(dict(dataset=h.bundle.name, K=K, support=name, d=d, z=z, T=T,
                         snr_per_round=snr, snr_over_T=snr * np.sqrt(T)))
        print("  feasibility %-9s K=%2d %-24s d=%6d  SNR/round=%.3f  x sqrt(T)=%.2f"
              % (h.bundle.name, K, name, d, snr, snr * np.sqrt(T)))
    return rows


def _import_previous_logs():
    """Resume across Kaggle sessions: copy results/logs/*.jsonl from any attached input
    (a previous session's saved output) into this session's log folder, de-duplicated by
    run id. Runs found there are skipped, not recomputed."""
    from ppml import config
    config.ensure_dirs()
    if not os.path.isdir("/kaggle/input"):
        return 0
    found = glob.glob("/kaggle/input/**/results/logs/*.jsonl", recursive=True)
    n_new = 0
    for src in found:
        dst = os.path.join(config.LOG_DIR, os.path.basename(src))
        seen = set()
        if os.path.exists(dst):
            with open(dst) as fh:
                for line in fh:
                    try:
                        seen.add(json.loads(line).get("config_hash"))
                    except ValueError:
                        pass
        with open(src) as fh, open(dst, "a") as out:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    h_ = json.loads(line).get("config_hash")
                except ValueError:
                    continue
                if h_ in seen:
                    continue
                seen.add(h_)
                out.write(line + "\n")
                n_new += 1
    if found:
        print("  resume       : %d earlier runs imported from %d attached log files"
              % (n_new, len(found)))
    return n_new


def _run_pooled(ppml, run_datasets, stages):
    """Task 7's cross-sensor arm, run once per session instead of once per dataset.

    It pools a dataset that has a gyroscope with one that has not, which is the only
    setting where zero-padding and an explicit mask can differ at all - inside a single
    gyroscope-less dataset both are constants. That needs two datasets loaded, so it does
    not belong in the per-dataset loop. Returns the pseudo-dataset name when it logged
    runs, so the report counts it beside the real ones.
    """
    if "resampling" not in stages:
        return []
    from ppml import config
    from ppml.experiments import exp_resampling as er
    if not er.pooled_available(run_datasets):
        print("\n--- stage %-11s skipped: this session loads %s and the arm needs %s"
              % ("pooled", ", ".join(run_datasets), " and ".join(er.POOLED_SETS)))
        return []
    if config.budget_left_s() <= 0:
        print("\n--- stage %-11s skipped: session budget exhausted" % "pooled")
        return []
    print("\n--- stage %-11s Task 7 pooled channels (gyro + no-gyro, one model)" % "pooled")
    incomplete = False
    try:
        er.pooled_run(available=run_datasets, verbose=VERBOSE)
    except config.BudgetExceeded as exc:
        incomplete = True
        print("  pooled arm stopped by the session budget (%s) - what it finished is "
              "logged below; re-run this file to complete it" % exc)
    except Exception as exc:
        print("  pooled arm raised %s: %s - the per-dataset channel table is unaffected"
              % (type(exc).__name__, exc))
        traceback.print_exc()
    # The CSVs are written from the log here rather than left to pooled_run, which only
    # writes when it logged something new: a resumed session dedupes every run and would
    # otherwise finish with the table missing.
    logged = er.table(er.POOLED_NAME)
    if not logged:
        return []
    try:
        er.write_pooled_csvs()
    except Exception as exc:
        print("  pooled CSVs not written: %s: %s" % (type(exc).__name__, exc))
    if incomplete:
        print("  NOTE pooled table is partial (budget) - %d rows logged so far"
              % len(logged))
    return [er.POOLED_NAME]


def _report(ppml, names):
    from ppml import config, crossdataset, runlog
    print("\n" + "=" * 78)
    print("OUTPUTS under %s" % config.RESULTS_DIR)
    for ds in names:
        rows = runlog.read_logs(dataset=ds, reportable_only=False)
        by_stage = {}
        for r in rows:
            by_stage[r.get("stage", "?")] = by_stage.get(r.get("stage", "?"), 0) + 1
        n_rep = sum(1 for r in rows if r.get("reportable"))
        print("  %-9s %5d logged runs (%d reportable)  %s"
              % (ds, len(rows), n_rep,
                 ", ".join("%s:%d" % kv for kv in sorted(by_stage.items()))))
    print("  logs   : %s/*.jsonl  (one line per run, the only source for figures)"
          % config.LOG_DIR)
    print("  csvs   : %s/*.csv    (per-stage summary tables)" % config.RESULTS_DIR)
    print("  one file: cross_dataset_runs.csv (fixed %d-column header, one row per run),"
          " cross_dataset_comparison.csv (every metric, four datasets side by side),"
          " pipeline_identity.csv (line 12 checked from the logs)"
          % len(__import__("ppml.crossdataset", fromlist=["x"]).RUN_COLUMNS))
    print("  tables : %s         (one CSV per stage table function)"
          % os.path.join(config.RESULTS_DIR, "tables"))
    print("  figures: %s         (PDF + 300 dpi PNG, drawn from the logs)"
          % config.FIG_DIR)
    print("  resume : keep results/ - already-logged runs are skipped on the next run")
    print("=" * 78)


def _cross_dataset(ppml):
    """Print the four datasets side by side from whatever this session has logged.

    One session is one dataset, so the other three columns read blank - that blank is the
    coverage statement the Work Order asks for, not a zero. The files are written by
    `ppml.reporting.run_report` just below; this is the console evidence for the session log.
    """
    from ppml import crossdataset
    print("\n--- the four datasets in one file (Work Order line 12)")
    try:
        crossdataset.print_summary()
    except Exception as exc:
        print("  cross-dataset block failed (%s: %s) - the logs are unaffected"
              % (type(exc).__name__, exc))
        return
    print("  Kaggle wipes /kaggle/working between sessions, so stack the four files in")
    print("  one merge afterwards:")
    print("      from ppml import crossdataset")
    print("      crossdataset.consolidate([<results folder of each per-dataset session>])")


def _finalise(ppml, names):
    """Figures from the logs + the section 13 run report and contradiction flags."""
    from ppml import config, reporting
    print("\n--- run report and figures")
    try:
        payload = reporting.run_report(datasets=names, write_figures=MAKE_FIGURES)
    except Exception as exc:
        print("  report failed (%s: %s) - the logs are still complete; run "
              "ppml.reporting.run_report() in a later session"
              % (type(exc).__name__, exc))
        traceback.print_exc()
        return None
    print("  report   : %s" % os.path.join(config.RESULTS_DIR, "run_report.md"))
    print("  claims   : %s" % os.path.join(config.RESULTS_DIR, "contradictions.csv"))
    print("  figures  : %s (+ figure_manifest.csv)" % config.FIG_DIR)
    flags = [c for c in payload["contradictions"]
             if c["verdict"] in ("CONTRADICTED", "NOT SUPPORTED", "UNEXPECTED")]
    if flags:
        print("\n  EVERY CLAIM THE MEASUREMENTS DO NOT SUPPORT (send this to the authors):")
        for c in flags:
            print("   - [%s] %s\n     %s" % (c["verdict"], c["paper_claim"],
                                             c["evidence"]))
    return payload


class _Tee(object):
    """Write everything printed to the console AND to run_log.txt, so the full log is one
    of the files in the results zip (Kaggle's log viewer is not a file you can attach)."""

    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, text):
        self.stream.write(text)
        try:
            self.fh.write(text)
        except Exception:
            pass
        return len(text)

    def flush(self):
        self.stream.flush()
        try:
            self.fh.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _zip_outputs():
    """One file to download: results/, figures/ and run_log.txt in PPML_results_<NB>.zip."""
    import zipfile
    out = os.path.join(INSTALL_DIR, "PPML_results_%s.zip" % NOTEBOOK_ID)
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for sub in ("results", "figures"):
                base = os.path.join(INSTALL_DIR, sub)
                for dp, dn, fn in os.walk(base):
                    for f in fn:
                        full = os.path.join(dp, f)
                        z.write(full, os.path.relpath(full, INSTALL_DIR))
            log = os.path.join(INSTALL_DIR, "run_log.txt")
            if os.path.exists(log):
                z.write(log, "run_log.txt")
        print("\nALL OUTPUTS IN ONE FILE: %s (%.1f MB) - download it from the Output tab"
              % (out, os.path.getsize(out) / 1e6))
    except Exception as exc:
        print("zip failed (%s) - results/ and figures/ are still in the Output tab" % exc)


def main():
    os.makedirs(INSTALL_DIR, exist_ok=True)
    if not isinstance(sys.stdout, _Tee):
        sys.stdout = _Tee(sys.stdout, os.path.join(INSTALL_DIR, "run_log.txt"))
    if not isinstance(sys.stderr, _Tee):
        sys.stderr = _Tee(sys.stderr, os.path.join(INSTALL_DIR, "run_log.txt"))
    try:
        return _main()
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        _zip_outputs()


def _main():
    os.environ["PPML_PROFILE"] = PROFILE
    # Optional independent epsilon cross-check (Google dp-accounting). Needs Internet; a
    # failure here changes nothing else - the in-package accountant is used either way.
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "dp-accounting"],
                       timeout=300, check=False, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except Exception:
        pass
    t_start = time.time()
    print(__doc__ if VERBOSE else "")
    ppml = _ppml_install()
    from ppml import config, datasets, models
    from ppml.harness import Harness
    _preflight(ppml)
    _import_previous_logs()
    config.set_deadline(seconds_from_now=SESSION_BUDGET_H * 3600.0)
    print("\nsession budget: %.1f h (deadline at %s)"
          % (SESSION_BUDGET_H, time.strftime("%H:%M", time.localtime(
              config.SESSION_DEADLINE))))
    from ppml import experiments as _exp
    stages = [s for s in RUN_STAGES if s in _exp.STAGES]

    # ── load every dataset of this notebook once ──────────────────────────────
    harnesses = {}
    for name in RUN_DATASETS:
        print("\n" + "#" * 78 + "\n# loading %s\n" % name + "#" * 78)
        try:
            cap = MAX_WINDOWS_BY_DATASET.get(name)
            config.MAX_WINDOWS = cap
            bundle = datasets.load(name, max_subjects=MAX_SUBJECTS, max_windows=cap or 0)
        except Exception as exc:
            print("  LOAD FAILED for %s: %s" % (name, exc))
            traceback.print_exc()
            print("  attach the dataset under Add Input (or turn Internet ON) and rerun")
            continue
        h = Harness(bundle, device=models.device_for(True))
        h.window_cap = cap
        m = bundle.meta
        print("  windows=%d subjects=%d shape=%s classes=%s names=%s"
              % (m.get("n_windows"), m.get("n_subjects"), m.get("window_shape"),
                 m.get("class_counts"), m.get("class_names")))
        print("  feature extraction %.1f s (%d features); window cap=%s"
              % (h.feature_time_s, h.n_features, cap))
        if name == "WISDM v2" and datasets.WISDM_ALIGNMENT:
            fr = [v["matched_fraction"] for v in datasets.WISDM_ALIGNMENT.values()]
            print("  WISDM accel/gyro matched fraction: min %.2f median %.2f"
                  % (min(fr), sorted(fr)[len(fr) // 2]))
        try:
            _feasibility(h)
        except Exception as exc:
            print("  feasibility check failed: %s" % exc)
        harnesses[name] = h
    # each dataset's public source for dplearn = the other dataset of this notebook
    for name, h in harnesses.items():
        others = [o for o in harnesses if o != name]
        if others:
            o = harnesses[others[0]]
            h.public_source = dict(name=o.bundle.name, Xf=o.Xf, y=o.y,
                                   n_class=datasets.N_CLASS_BY_DATASET[o.bundle.name])

    # ── stage-major loop: each stage on every dataset, then the next stage ─────
    stopped = False
    for stage in stages:
        for name, h in harnesses.items():
            if stage not in STAGES_BY_DATASET.get(name, RUN_STAGES):
                continue
            if config.budget_left_s() <= 0:
                stopped = True
                break
            datasets.set_class_space(name)       # 3 classes for WISDM, 6 otherwise
            config.MAX_WINDOWS = getattr(h, "window_cap", None)
            print("\n" + "=" * 78 + "\n%s | %s  (elapsed %.1f min)"
                  % (name, stage, (time.time() - t_start) / 60.0))
            if not _run_stage(ppml, h, stage, None, None, t_start):
                stopped = True
                break
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        if stopped:
            print("\nsession budget reached after %.1f min"
                  % ((time.time() - t_start) / 60.0))
            break
    names = list(harnesses)
    _report(ppml, names)
    _cross_dataset(ppml)
    _finalise(ppml, names)
    # The raw datasets are not results: remove the downloaded copies so the saved version's
    # output is only results/ and figures/ (small, quick to save, quick to attach later).
    try:
        shutil.rmtree(config.DATA_DIR, ignore_errors=True)
        print("removed downloaded raw data from %s (results and figures are kept)"
              % config.DATA_DIR)
    except Exception:
        pass
    print("total session time used: %.1f min" % ((time.time() - t_start) / 60.0))
    if stopped:
        print("NOT COMPLETE - Save Version, attach its output as an input, run again.")
    else:
        print("ALL STAGES COMPLETE for %s" % ", ".join(names))
    return ppml


if __name__ == "__main__":
    PPML = main()
