
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize ablation results (metrics.json files) into a CSV table.

Usage:
python summarize_results.py --root results/ablations --out results/ablation_summary.csv
"""

import argparse
import os
import json
import csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/ablations")
    ap.add_argument("--out", default="results/ablation_summary.csv")
    args = ap.parse_args()

    rows = []
    for dirpath, dirnames, filenames in os.walk(args.root):
        if "metrics.json" in filenames:
            p = os.path.join(dirpath, "metrics.json")
            with open(p, "r") as f:
                m = json.load(f)
            exp_name = os.path.basename(dirpath)
            rows.append({
                "exp": exp_name,
                "auc": m.get("auc"),
                "ap": m.get("ap"),
                "acc": m.get("acc"),
                "mcc": m.get("mcc"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "seconds_total_predict": m.get("seconds_total_predict"),
                "n_samples": m.get("n_samples"),
                "disable_shrinkage": m.get("disable_shrinkage"),
                "disable_eca": m.get("disable_eca"),
                "disable_cbam": m.get("disable_cbam"),
            })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = ["exp","auc","ap","acc","mcc","precision","recall","f1","seconds_total_predict","n_samples",
            "disable_shrinkage","disable_eca","disable_cbam"]

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["exp"]):
            w.writerow(r)

    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
