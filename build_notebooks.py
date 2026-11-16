"""Assemble the two Kaggle notebooks from ppml/ + template_head.py + template_tail.py."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # folder that contains ppml/
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "dist")

CORE = ["main", "privacy", "dplearn", "dpsgd", "splits", "mia", "noniid"]
DPSGD2 = ["dpsgd_e085", "dpsgd_cnn83", "dpsgd_e03", "dpsgd_clip0", "dpsgd_cnn3",
          "dpsgd_cnn085", "dpsgd_cnn0"]
NOTEBOOKS = {
    "A": dict(datasets=["HARTH", "PAMAP2"],
              stages={"HARTH": CORE + ["clip", "byzantine", "resampling"],
                      "PAMAP2": CORE + ["resampling", "cost"]},
              fname="ppml_kaggle_A_harth_pamap2.py"),
    "B": dict(datasets=["UCI HAR", "WISDM v2"],
              stages={"UCI HAR": CORE + ["clip", "byzantine", "features"],
                      "WISDM v2": CORE + ["resampling"]},
              fname="ppml_kaggle_B_uci_wisdm.py"),
    # Rerun after the first session: UCI HAR failed to load in B (macOS folder in the zip),
    # and the Byzantine attacks were redefined, so HARTH's Byzantine stage is rerun here.
    "C": dict(datasets=["UCI HAR", "HARTH"],
              stages={"UCI HAR": CORE + ["clip", "byzantine", "features"],
                      "HARTH": ["byzantine"]},
              fname="ppml_kaggle_C_uci_rerun.py"),
    # Revision round 2: example-level DP-SGD at the paper's quoted budgets (eps 0.85 for
    # the small models, eps 8.3 for CNN-BiLSTM) plus eps 0.3 and clipping-only controls.
    "D": dict(datasets=["WISDM v2", "PAMAP2"],
              stages={"WISDM v2": DPSGD2, "PAMAP2": DPSGD2},
              fname="ppml_kaggle_D_dpsgd_wisdm_pamap2.py"),
    "E": dict(datasets=["HARTH", "UCI HAR"],
              stages={"HARTH": DPSGD2, "UCI HAR": DPSGD2},
              fname="ppml_kaggle_E_dpsgd_harth_uci.py"),
}


def blob():
    order = [l.strip() for l in open(os.path.join(HERE, "file_order.txt")) if l.strip()]
    if "ppml/experiments/exp_dplearn.py" not in order:
        order.append("ppml/experiments/exp_dplearn.py")
    parts = []
    for rel in order:
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        assert "'''" not in src, "triple single quote in %s would break the blob" % rel
        parts.append("#@@FILE %s\n%s" % (rel, src.rstrip("\n") + "\n"))
    # every module on disk must be embedded
    on_disk = sorted(os.path.relpath(os.path.join(dp, f), ROOT)
                     for dp, dn, fn in os.walk(os.path.join(ROOT, "ppml")) for f in fn
                     if f.endswith(".py"))
    missing = [p for p in on_disk if p not in order]
    assert not missing, "not embedded: %s" % missing
    return "\n" + "\n".join(parts)


def write_ipynb(path, nb, spec, code):
    import json
    md = ("# PPML revision experiments - Notebook %s (%s)\n\n"
          "IEEE Access-2026-40021. **Do not edit the code cell.**\n\n"
          "1. Settings (right panel): **Accelerator = GPU T4 x1**, **Internet = On**.\n"
          "2. Optional: Add Input -> attach the datasets; otherwise they are downloaded "
          "from UCI automatically.\n"
          "3. Click **Save Version -> Save & Run All (Commit)**. The run stops itself "
          "before the time limit and writes `results/` and `figures/` to the Output tab.\n"
          % (nb, " + ".join(spec["datasets"])))
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": md.splitlines(True)},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": code.splitlines(True)},
    ]
    nbj = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "gpu", "isInternetEnabled": True,
                   "isGpuEnabled": True}},
        "nbformat": 4, "nbformat_minor": 4}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(nbj, fh, indent=1)
    print("wrote %s" % path)


def main():
    os.makedirs(OUT, exist_ok=True)
    head = open(os.path.join(HERE, "template_head.py"), encoding="utf-8").read()
    tail = open(os.path.join(HERE, "template_tail.py"), encoding="utf-8").read()
    b = blob()
    only = os.environ.get("NB_ONLY")
    for nb, spec in NOTEBOOKS.items():
        if only and nb not in only:
            continue
        h = (head.replace("@@NB_ID@@", nb)
                 .replace("@@NB_DATASETS@@", " + ".join(spec["datasets"]))
                 .replace("@@RUN_DATASETS@@", repr(spec["datasets"]))
                 .replace("@@STAGES_BY_DATASET@@", "{\n" + "".join(
                     "    %r: %r,\n" % (k, v) for k, v in spec["stages"].items()) + "}"))
        assert "@@" not in h.replace("#@@FILE", "")
        text = h + b + tail
        path = os.path.join(OUT, spec["fname"])
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        compile(text, path, "exec")
        print("wrote %s (%d lines)" % (path, text.count("\n")))
        write_ipynb(path[:-3] + ".ipynb", nb, spec, text)


if __name__ == "__main__":
    main()
