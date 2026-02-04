
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run an ablation & sensitivity suite for MARSNet_exp.py.

It calls the training + prediction pipeline multiple times with different switches and
saves a metrics JSON per run. Designed to produce reviewer-friendly tables quickly.

Example:
python run_ablation_suite.py \
  --train_pos ./GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.train.positives.fa \
  --train_neg ./GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.train.negatives.fa \
  --test_pos  ./GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.ls.positives.fa \
  --test_neg  ./GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.ls.negatives.fa
"""

import argparse
import os
import subprocess
import json
from datetime import datetime

def run(cmd):
    print(" ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pos", required=True)
    ap.add_argument("--train_neg", required=True)
    ap.add_argument("--test_pos", required=True)
    ap.add_argument("--test_neg", required=True)
    ap.add_argument("--python", default="python")
    ap.add_argument("--script", default="MARSNet_exp.py")
    ap.add_argument("--out_dir", default="results/ablations")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_filters", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0,1,2,3])
    ap.add_argument("--window_sizes", type=int, nargs="+", default=[101,201,301,401])
    ap.add_argument("--channels", type=int, nargs="+", default=[7,3,2,1])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    base_prefix = os.path.join(args.out_dir, "model")
    profile_out = os.path.join(args.out_dir, "profile.jsonl")

    # Experiments: (name, extra_flags)
    exps = [
        ("full", []),
        ("no_shrinkage", ["--disable_shrinkage"]),
        ("no_eca", ["--disable_eca"]),
        ("no_cbam", ["--disable_cbam"]),
        ("no_eca_no_cbam", ["--disable_eca", "--disable_cbam"]),
        # single-scale sensitivity (use only 401)
        ("single_scale_401", ["--window_sizes", "401", "--channels", "1", "--ensemble_weights", "1"]),
        # uniform weights sensitivity
        ("uniform_weights", ["--ensemble_weights", "0.25", "0.25", "0.25", "0.25"]),
    ]

    # Train once per experiment, then predict.
    for name, extra in exps:
        exp_dir = os.path.join(args.out_dir, name)
        os.makedirs(exp_dir, exist_ok=True)
        model_prefix = os.path.join(exp_dir, "model")
        metrics_json = os.path.join(exp_dir, "metrics.json")

        common = [
            args.python, args.script,
            "--posi", args.train_pos,
            "--nega", args.train_neg,
            "--model_type", "ARSNet",
            "--model_file", model_prefix,
            "--batch_size", str(args.batch_size),
            "--n_epochs", str(args.epochs),
            "--num_filters", str(args.num_filters),
            "--seeds", *[str(s) for s in args.seeds],
            "--window_sizes", *[str(w) for w in args.window_sizes],
            "--channels", *[str(c) for c in args.channels],
            "--profile",
            "--profile_out", profile_out,
        ]

        # train
        run(common + ["--train"] + extra)

        # predict
        pred_cmd = [
            args.python, args.script,
            "--testfile", args.test_pos,
            "--nega", args.test_neg,
            "--model_type", "ARSNet",
            "--model_file", model_prefix,
            "--batch_size", str(args.batch_size),
            "--n_epochs", str(args.epochs),
            "--num_filters", str(args.num_filters),
            "--seeds", *[str(s) for s in args.seeds],
            "--window_sizes", *[str(w) for w in args.window_sizes],
            "--channels", *[str(c) for c in args.channels],
            "--profile",
            "--profile_out", profile_out,
            "--save_metrics_json",
            "--metrics_json", metrics_json,
            "--predict",
        ] + extra

        run(pred_cmd)

        with open(metrics_json, "r") as f:
            met = json.load(f)
        print(f"[{name}] AUC={met['auc']:.4f} AP={met['ap']:.4f} MCC={met['mcc']:.4f}")

if __name__ == "__main__":
    main()
