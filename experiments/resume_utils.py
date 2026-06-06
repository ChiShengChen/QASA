"""Tiny helpers for resumable, incrementally-written experiment CSVs.

Each completed (config, seed) run is appended to a stable-named progress CSV
immediately, so a mid-stage interruption loses at most the single in-flight run;
re-launching the same command skips runs already present in the CSV.
"""

import os
import csv
import numpy as np


def load_done(path, key_cols):
    """Return the set of key tuples (as strings) already recorded in `path`."""
    done = set()
    if os.path.exists(path):
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                try:
                    done.add(tuple(str(row[k]) for k in key_cols))
                except KeyError:
                    continue
    return done


def append_row(path, header, row):
    """Append one row, writing the header first if the file is new/empty."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def aggregate(path, group_col, val_cols, extra_cols=()):
    """Group rows of `path` by `group_col`; return per-group mean/std of val_cols
    (+ the first value of each extra_col, + count N)."""
    groups = {}
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            g = row[group_col]
            groups.setdefault(g, []).append(row)
    out = []
    for g, rows in groups.items():
        rec = {'group': g, 'N': len(rows)}
        for vc in val_cols:
            vals = [float(r[vc]) for r in rows if r.get(vc) not in (None, '')]
            rec[f'{vc}_mean'] = float(np.mean(vals)) if vals else float('nan')
            rec[f'{vc}_std'] = float(np.std(vals)) if len(vals) > 1 else 0.0
        for ec in extra_cols:
            rec[ec] = rows[0].get(ec, '')
        out.append(rec)
    return out
