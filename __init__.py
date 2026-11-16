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
