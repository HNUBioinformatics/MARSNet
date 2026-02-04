
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Leakage / redundancy check for train-vs-test RNA sequences.

Idea:
- Build k-mer sets for training sequences.
- For each test sequence, compute max Jaccard similarity to any *candidate* train sequence
  using an inverted index (kmer -> train seq ids).
- Report distribution + count above thresholds, and save a histogram PDF.

This is not a perfect biological identity measure, but it is a practical, reviewer-friendly
sanity check for potential train-test overlap / near-duplication.
"""

import argparse
import os
import json
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_fasta(path):
    seqs = []
    if not path:
        return seqs
    with open(path, "r") as f:
        s = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if s:
                    seqs.append("".join(s).upper().replace("T", "U"))
                    s = []
            else:
                s.append(line)
        if s:
            seqs.append("".join(s).upper().replace("T", "U"))
    return seqs

def kmers(seq, k):
    if len(seq) < k:
        return set()
    return {seq[i:i+k] for i in range(len(seq)-k+1) if "N" not in seq[i:i+k]}

def build_index(train_seqs, k):
    train_kmers = []
    index = defaultdict(list)
    for i, s in enumerate(train_seqs):
        ks = kmers(s, k)
        train_kmers.append(ks)
        for x in ks:
            index[x].append(i)
    return train_kmers, index

def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a) + len(b) - inter
    return inter / union

def max_sim_for_test(test_seq, test_kmers, train_kmers, index, max_candidates=2000):
    # candidate train ids from shared kmers
    cand = set()
    for x in test_kmers:
        for tid in index.get(x, []):
            cand.add(tid)
            if len(cand) >= max_candidates:
                break
        if len(cand) >= max_candidates:
            break
    if not cand:
        return 0.0, None

    best = 0.0
    best_id = None
    for tid in cand:
        s = jaccard(test_kmers, train_kmers[tid])
        if s > best:
            best = s
            best_id = tid
    return best, best_id

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pos", required=True)
    ap.add_argument("--train_neg", required=True)
    ap.add_argument("--test_pos", required=True)
    ap.add_argument("--test_neg", required=True)
    ap.add_argument("--k", type=int, default=8, help="k-mer length (default: 8)")
    ap.add_argument("--max_candidates", type=int, default=2000)
    ap.add_argument("--out_json", default="results/leakage.json")
    ap.add_argument("--out_pdf", default="results/leakage_hist.pdf")
    args = ap.parse_args()

    train = read_fasta(args.train_pos) + read_fasta(args.train_neg)
    test = read_fasta(args.test_pos) + read_fasta(args.test_neg)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    train_kmers, index = build_index(train, args.k)

    sims = []
    for s in test:
        ks = kmers(s, args.k)
        sim, _ = max_sim_for_test(s, ks, train_kmers, index, max_candidates=args.max_candidates)
        sims.append(sim)

    # summary
    thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]
    summary = {
        "k": args.k,
        "n_train": len(train),
        "n_test": len(test),
        "max_candidates": args.max_candidates,
        "mean_max_jaccard": float(sum(sims) / max(1, len(sims))),
        "median_max_jaccard": float(sorted(sims)[len(sims)//2]) if sims else None,
        "counts_ge_threshold": {str(t): int(sum(1 for x in sims if x >= t)) for t in thresholds},
        "sims": sims,
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # plot
    plt.figure()
    plt.hist(sims, bins=50)
    plt.xlabel(f"Max Jaccard similarity to any train seq (k={args.k})")
    plt.ylabel("Count (test sequences)")
    plt.tight_layout()
    plt.savefig(args.out_pdf)

    print("Wrote:", args.out_json)
    print("Wrote:", args.out_pdf)
    print("Mean max Jaccard:", summary["mean_max_jaccard"])
    print("Counts >= thresholds:", summary["counts_ge_threshold"])

if __name__ == "__main__":
    main()
