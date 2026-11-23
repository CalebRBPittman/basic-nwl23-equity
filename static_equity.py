"""
Static equity of Scrabble tiles, using the NWL23 word list.

How many points is each tile worth compared with an average tile?

  1. deal a lot of random 7-tile racks
  2. for each rack, find every play the lexicon allows and score it
     (face values, +50 for using all 7 tiles)
  3. rack equity = best (score + value of the tiles left over)
  4. the leftover ("leave") values are what we want, so solve for them
     by iterating until they stop moving

Run from the repo root:
    python static_equity.py
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

# the lexicon lives in the repo, next to this file
REPO_DIR = Path(__file__).resolve().parent
LEXICON_PATH = REPO_DIR / "data" / "NWL23.txt"
RESULTS_DIR = REPO_DIR / "results"

N_RACKS = 100_000
N_BOOT = 200
SEED = 2023

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = 26                    # tile index for the blank, so it sorts last
N_TILES = 27
RACK_SIZE = 7
BINGO_BONUS = 50

#        A  B  C  D   E  F  G  H  I  J  K  L  M  N  O  P  Q  R  S  T  U  V  W  X  Y  Z  ?
COPIES = [9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1, 2]
VALUES = [1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10, 0]
TILE_LABELS = list(LETTERS) + ["?"]
P_BAG = np.array(COPIES) / 100  # chance a random draw is each tile

# every way to pick 2+ tiles out of a 7 tile rack: (tiles played, tiles kept)
SUBSETS = []
for mask in range(1, 1 << RACK_SIZE):
    played = tuple(i for i in range(RACK_SIZE) if mask >> i & 1)
    if len(played) >= 2:
        kept = tuple(i for i in range(RACK_SIZE) if not mask >> i & 1)
        SUBSETS.append((played, kept))


def load_words(path=LEXICON_PATH):
    """One word per line. Only 2-7 letter words matter for a single rack."""
    words = set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if parts:
                w = parts[0].upper()
                if 2 <= len(w) <= RACK_SIZE and w.isascii() and w.isalpha():
                    words.add(w)
    return words


def build_pattern_index(words, max_blanks=2):
    """
    Set of strings like 'ACT', 'AC?', 'A??' meaning "these real tiles, plus
    this many blanks, can make some word". Letters are sorted so the order
    tiles come off the rack doesn't matter. A blank scores zero, so all we
    need to know is whether a play exists, not which word it is.
    """
    keys = {"".join(sorted(w)) for w in words}
    patterns = set(keys)
    for k in keys:
        n = len(k)
        for b in range(1, min(max_blanks, n) + 1):
            for drop in itertools.combinations(range(n), b):
                d = set(drop)
                patterns.add("".join(k[i] for i in range(n) if i not in d) + "?" * b)
    return patterns


def sample_racks(n, rng):
    bag = np.repeat(np.arange(N_TILES), COPIES)
    out = np.empty((n, RACK_SIZE), dtype=np.int8)
    for s in range(0, n, 10_000):
        m = min(10_000, n - s)
        idx = np.argsort(rng.random((m, len(bag))), axis=1)[:, :RACK_SIZE]
        out[s:s + m] = np.sort(bag[idx], axis=1)
    return out


def build_candidates(racks, patterns, chunk=5000):
    """
    For every rack, list each distinct (score, leave) it can produce.
    Returns the scores, the leave tile counts, and where each rack starts.
    """
    all_scores, all_leave, per_rack = [], [], []

    for s in range(0, len(racks), chunk):
        scores, rows, tiles_left, counts = [], [], [], []
        k = 0
        for rack in racks[s:s + chunk].tolist():
            seen = {(0.0, ())}          # exchanging the whole rack is always an option
            for played, kept in SUBSETS:
                tiles = [rack[i] for i in played]
                n_blank = tiles.count(BLANK)
                key = "".join(LETTERS[t] for t in tiles if t != BLANK) + "?" * n_blank
                if key in patterns:
                    score = sum(VALUES[t] for t in tiles)
                    if len(played) == RACK_SIZE:
                        score += BINGO_BONUS
                    seen.add((score, tuple(rack[i] for i in kept)))
            for score, leave in seen:
                scores.append(score)
                for t in leave:
                    rows.append(k)
                    tiles_left.append(t)
                k += 1
            counts.append(len(seen))

        leave_counts = np.zeros((k, N_TILES), dtype=np.float32)
        np.add.at(leave_counts, (np.array(rows, dtype=np.int64),
                                 np.array(tiles_left, dtype=np.int64)), 1.0)
        all_scores.append(np.array(scores, dtype=np.float32))
        all_leave.append(leave_counts)
        per_rack.extend(counts)

    starts = np.concatenate([[0], np.cumsum(per_rack)[:-1]])
    return np.concatenate(all_scores), np.vstack(all_leave), starts


def rack_counts(racks):
    return (racks[:, :, None] == np.arange(N_TILES)).sum(axis=1).astype(np.float64)


def centred_coefs(X, y):
    """
    Regress y on tile counts and subtract the average tile's coefficient.
    No intercept - every rack has exactly 7 tiles so it would be collinear.
    Centering is there because a tile you keep gets swapped for an average
    draw, so what matters is how much better/worse it is than average.
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta - P_BAG @ beta


def solve_leave_values(scores, leave, starts, X, tol=0.005, damping=0.5, max_iter=60):
    """
    Leave values depend on rack equity and rack equity depends on leave
    values, so start from zero and keep updating until they settle.
    Damping at 0.5 stops it from bouncing back and forth.
    """
    v = np.zeros(N_TILES)
    for it in range(1, max_iter + 1):
        equity = scores + leave @ v.astype(np.float32)
        best = np.maximum.reduceat(equity, starts)       # best play for each rack
        v_new = centred_coefs(X, best)
        v_next = damping * v_new + (1 - damping) * v
        change = np.abs(v_next - v).max()
        print(f"  iter {it:2d}   biggest change {change:7.4f}")
        v = v_next
        if change < tol:
            break
    equity = scores + leave @ v.astype(np.float32)
    return v, np.maximum.reduceat(equity, starts)


def bootstrap_ci(X, best, rng):
    """95% interval from resampling racks and refitting."""
    draws = np.empty((N_BOOT, N_TILES))
    for b in range(N_BOOT):
        pick = rng.integers(0, len(best), len(best))
        draws[b] = centred_coefs(X[pick], best[pick])
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def plot_tiles(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.sort_values("static_equity")
    err_lo = (d.static_equity - d.ci95_lo).clip(lower=0)
    err_hi = (d.ci95_hi - d.static_equity).clip(lower=0)
    colors = ["#2a7ab0" if x >= 0 else "#c0504d" for x in d.static_equity]

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.barh(d.tile, d.static_equity, color=colors, xerr=[err_lo, err_hi],
            error_kw={"elinewidth": 0.8, "capsize": 2})
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Static equity vs. average tile (points, 95% bootstrap CI)")
    ax.set_title("Tile static equity, NWL23")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)

    words = load_words()
    print(f"{len(words):,} words of 2-{RACK_SIZE} letters in the lexicon")
    patterns = build_pattern_index(words)

    racks = sample_racks(N_RACKS, rng)
    X = rack_counts(racks)
    scores, leave, starts = build_candidates(racks, patterns)
    print(f"{N_RACKS:,} racks, {len(scores):,} candidate plays")

    print("solving for leave values:")
    v, best = solve_leave_values(scores, leave, starts, X)
    lo, hi = bootstrap_ci(X, best, rng)

    df = pd.DataFrame({"tile": TILE_LABELS, "face_value": VALUES, "copies": COPIES,
                       "static_equity": v, "ci95_lo": lo, "ci95_hi": hi})
    df = df.sort_values("static_equity", ascending=False).reset_index(drop=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    df.round(3).to_csv(RESULTS_DIR / "tile_static_equity.csv", index=False)
    plot_tiles(df, RESULTS_DIR / "tile_static_equity.png")
    print()
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
