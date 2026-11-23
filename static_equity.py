#!/usr/bin/env python3
"""
Static equity of Scrabble tiles under the NWL23 lexicon.

Research question
-----------------
Ignoring the board, how many points is each tile worth, on average, relative to
a randomly drawn tile?  "Worth" = the tile's marginal effect on the expected
equity of a 7-tile rack, where equity = points scored now + value of the tiles
kept for the next turn (the "leave").

Method (see README.md for the full write-up)
--------------------------------------------
1. Monte-Carlo sample N racks from the standard 100-tile bag.
2. For every rack, enumerate every legal sub-multiset of tiles that spells a
   NWL23 word (blanks are wildcards), scoring each one.
3. Equity of a play = score + leave value; rack equity = max over plays
   (an "exchange everything" option worth 0 is always available).
4. Leave values are unknown, so solve for them by fixed-point iteration:
   regress rack equity on tile counts -> centre against the average tile ->
   use those as leave values -> recompute best plays -> repeat to convergence.
5. Bootstrap racks for uncertainty; run a sensitivity analysis on the scoring
   assumptions.

Usage
-----
    python static_equity.py --lexicon NWL23.txt --n-racks 50000 --outdir results
    python static_equity.py --selftest
"""
from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Game constants (standard English Scrabble)
# --------------------------------------------------------------------------
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = 26
N_TILES = 27
#         A  B  C  D   E  F  G  H  I  J  K  L  M  N  O  P  Q  R  S  T  U  V  W  X  Y  Z  ?
DIST = [9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1, 2]
VALUES = [1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10, 0]
RACK_SIZE = 7
BINGO_BONUS = 50
TILE_LABELS = list(LETTERS) + ["?"]
assert sum(DIST) == 100 and len(DIST) == len(VALUES) == N_TILES

# All sub-selections of a 7-tile rack that form a play of >= 2 tiles.
SUBSETS = []
for _mask in range(1, 1 << RACK_SIZE):
    _inc = tuple(i for i in range(RACK_SIZE) if _mask >> i & 1)
    if len(_inc) >= 2:
        SUBSETS.append((_inc, tuple(i for i in range(RACK_SIZE) if not _mask >> i & 1)))


# --------------------------------------------------------------------------
# Lexicon
# --------------------------------------------------------------------------
def load_lexicon(path: str, max_len: int = RACK_SIZE) -> set[str]:
    """Read one word per line (case/line-ending agnostic); keep 2..max_len letters."""
    words = set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            w = parts[0].upper()
            if 2 <= len(w) <= max_len and w.isascii() and w.isalpha():
                words.add(w)
    return words


def build_pattern_index(words: set[str], max_blanks: int = 2) -> set[str]:
    """
    Every string 'sorted letters + ?'*b' that can be completed to a valid word
    by filling the b blanks.  Because a blank scores 0, a play's score depends
    only on the multiset of real tiles used, so existence is all we need.
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


# --------------------------------------------------------------------------
# Sampling and play enumeration
# --------------------------------------------------------------------------
def sample_racks(n: int, rng: np.random.Generator) -> np.ndarray:
    bag = np.repeat(np.arange(N_TILES), DIST)
    out = np.empty((n, RACK_SIZE), dtype=np.int8)
    for s in range(0, n, 10_000):
        m = min(10_000, n - s)
        idx = np.argsort(rng.random((m, len(bag))), axis=1)[:, :RACK_SIZE]
        out[s:s + m] = np.sort(bag[idx], axis=1)  # sorted: blanks (26) last
    return out


def play_score(tiles: list[int], letter_mult: float, word_mult: float) -> float:
    vals = [VALUES[t] for t in tiles]
    base = sum(vals) + (letter_mult - 1) * max(vals)  # one premium letter square
    return base * word_mult + (BINGO_BONUS if len(tiles) == RACK_SIZE else 0)


@dataclass
class Candidates:
    scores: np.ndarray      # (n_cand,)  points for each candidate play
    leave: np.ndarray       # (n_cand, 27) tile counts left on the rack
    starts: np.ndarray      # (n_racks,) first candidate index of each rack
    has_bingo: np.ndarray   # (n_racks,) bool


def build_candidates(racks: np.ndarray, patterns: set[str],
                     letter_mult: float = 1.0, word_mult: float = 1.0,
                     chunk: int = 5000) -> Candidates:
    all_scores, all_leave, n_per_rack, bingo = [], [], [], []
    for s in range(0, len(racks), chunk):
        scores, li, lt, counts = [], [], [], []
        k = 0
        for rack in racks[s:s + chunk].tolist():
            seen = {(0.0, ())}             # exchange everything: 0 points, empty leave
            has_b = False
            for inc, exc in SUBSETS:
                tiles = [rack[i] for i in inc]
                nb = tiles.count(BLANK)
                key = "".join(LETTERS[t] for t in tiles if t != BLANK) + "?" * nb
                if key in patterns:
                    seen.add((play_score(tiles, letter_mult, word_mult),
                              tuple(rack[i] for i in exc)))
                    has_b |= len(inc) == RACK_SIZE
            for sc, lv in seen:
                scores.append(sc)
                for t in lv:
                    li.append(k)
                    lt.append(t)
                k += 1
            counts.append(len(seen))
            bingo.append(has_b)
        M = np.zeros((k, N_TILES), dtype=np.float32)
        np.add.at(M, (np.array(li, dtype=np.int64), np.array(lt, dtype=np.int64)), 1.0)
        all_scores.append(np.array(scores, dtype=np.float32))
        all_leave.append(M)
        n_per_rack.extend(counts)
        # lt entries are tile ids; leave rows built per candidate above
    n_per_rack = np.array(n_per_rack)
    starts = np.concatenate([[0], np.cumsum(n_per_rack)[:-1]])
    return Candidates(np.concatenate(all_scores), np.vstack(all_leave),
                      starts, np.array(bingo))


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------
P_BAG = np.array(DIST) / 100.0


def rack_counts(racks: np.ndarray) -> np.ndarray:
    return (racks[:, :, None] == np.arange(N_TILES)).sum(axis=1).astype(np.float64)


def centred_coefs(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS of y on tile counts (no intercept: counts sum to 7), centred on the average tile."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta - P_BAG @ beta


def solve_static_equity(cands: Candidates, X: np.ndarray, tol: float = 0.005,
                        damping: float = 0.5, max_iter: int = 60, verbose: bool = True):
    """Fixed point: v = centred_coefs(X, max_play(score + leave @ v))."""
    v = np.zeros(N_TILES)
    history = []
    for it in range(1, max_iter + 1):
        eq = cands.scores + cands.leave @ v.astype(np.float32)
        best = np.maximum.reduceat(eq, cands.starts)
        v_new = centred_coefs(X, best)
        v_next = damping * v_new + (1 - damping) * v
        delta = np.abs(v_next - v).max()
        history.append((it, delta, best.mean()))
        if verbose:
            print(f"  iter {it:2d}  max|dv|={delta:7.4f}  mean rack equity={best.mean():6.2f}")
        v = v_next
        if delta < tol:
            break
    eq = cands.scores + cands.leave @ v.astype(np.float32)
    return v, np.maximum.reduceat(eq, cands.starts), history


def bootstrap_ci(X: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator):
    if n_boot <= 0:
        nan = np.full(N_TILES, np.nan)
        return nan, nan, nan
    n = len(y)
    draws = np.empty((n_boot, N_TILES))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = centred_coefs(X[idx], y[idx])
    return draws.std(axis=0), np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def run(lexicon: str, n_racks: int, seed: int, letter_mult: float, word_mult: float,
        n_boot: int, patterns: set[str] | None = None, verbose: bool = True):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    if patterns is None:
        words = load_lexicon(lexicon)
        if verbose:
            print(f"Lexicon: {len(words):,} words of 2-{RACK_SIZE} letters")
        patterns = build_pattern_index(words)
        if verbose:
            print(f"Pattern index: {len(patterns):,} entries ({time.time()-t0:.0f}s)")
    racks = sample_racks(n_racks, rng)
    X = rack_counts(racks)
    cands = build_candidates(racks, patterns, letter_mult, word_mult)
    if verbose:
        print(f"{n_racks:,} racks -> {len(cands.scores):,} candidate plays ({time.time()-t0:.0f}s)")

    # Immediate-score value: no leave term at all (v = 0)
    best0 = np.maximum.reduceat(cands.scores, cands.starts)
    immediate = centred_coefs(X, best0)

    if verbose:
        print("Solving for leave values:")
    v, best, hist = solve_static_equity(cands, X, verbose=verbose)

    se, lo, hi = bootstrap_ci(X, best, n_boot, rng)
    bingo_pp = 100 * centred_coefs(X, cands.has_bingo.astype(float))

    # Diagnostic: duplication penalty (extra copies of a letter beyond the first)
    dup = np.clip(X - 1, 0, None).sum(axis=1)
    coef, *_ = np.linalg.lstsq(np.column_stack([X, dup]), best, rcond=None)

    df = pd.DataFrame({
        "tile": TILE_LABELS, "face_value": VALUES, "copies": DIST,
        "static_equity": v, "ci95_lo": lo, "ci95_hi": hi, "boot_se": se,
        "immediate_value": immediate, "bingo_prob_pp": bingo_pp,
    }).sort_values("static_equity", ascending=False).reset_index(drop=True)
    meta = {"bingo_rate": cands.has_bingo.mean(), "mean_best_equity": best.mean(),
            "mean_best_score": best0.mean(), "dup_penalty": coef[-1],
            "iterations": len(hist), "n_racks": n_racks}
    return df, meta, patterns


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def plot_results(df: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.sort_values("static_equity")
    fig, ax = plt.subplots(figsize=(7, 9))
    colors = ["#2a7ab0" if x >= 0 else "#c0504d" for x in d.static_equity]
    ax.barh(d.tile, d.static_equity, color=colors,
            xerr=[d.static_equity - d.ci95_lo, d.ci95_hi - d.static_equity],
            error_kw={"elinewidth": 0.8, "capsize": 2})
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Static equity vs. average tile (points, 95% bootstrap CI)")
    ax.set_title("Scrabble tile static equity, NWL23 (board-agnostic)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def selftest():
    words = {"AT", "TA", "CAT", "ACT", "TACO"}
    pats = build_pattern_index(words)
    assert "ACT" in pats and "AC?" in pats and "A??" in pats and "??" in pats
    assert "ACZ" not in pats and "Z?" not in pats
    # rack A C T + 4 junk tiles: best play must be able to use CAT/ACT
    rack = np.array([[0, 2, 19, 25, 25, 25, 25]], dtype=np.int8)  # A C T Z Z Z Z
    c = build_candidates(rack, pats)
    assert c.scores.max() == 1 + 3 + 1, c.scores  # ACT = 5 pts, not a bingo
    assert not c.has_bingo[0]
    # a blank can stand in for a missing letter: ?-C-T can make CAT
    rack = np.array([[2, 19, 26, 25, 25, 25, 25]], dtype=np.int8)
    c = build_candidates(rack, pats)
    assert c.scores.max() == 3 + 1, c.scores
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", help="NWL23 word list, one word per line")
    ap.add_argument("--n-racks", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--letter-mult", type=float, default=1.0,
                    help="multiplier on the best letter in each play (premium-square proxy)")
    ap.add_argument("--word-mult", type=float, default=1.0, help="word multiplier on each play")
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.lexicon:
        ap.error("--lexicon is required")

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    df, meta, patterns = run(a.lexicon, a.n_racks, a.seed, a.letter_mult, a.word_mult, a.n_boot)
    df.round(3).to_csv(out / "tile_static_equity.csv", index=False)
    plot_results(df, out / "tile_static_equity.png")
    print("\n", df.round(2).to_string(index=False))
    print(f"\nP(7-tile bingo available) = {meta['bingo_rate']:.1%}; "
          f"mean best raw score = {meta['mean_best_score']:.1f}; "
          f"duplicate-tile penalty = {meta['dup_penalty']:.2f} pts/extra copy")

    if not a.no_sensitivity:
        print("\nSensitivity of static equity to scoring assumptions:")
        cols = {}
        for lm, wm in [(1, 1), (2, 1), (2, 2), (3, 2)]:
            d, _, _ = run(a.lexicon, a.n_racks, a.seed, lm, wm, 0, patterns=patterns, verbose=False)
            cols[f"L{lm}xW{wm}"] = d.set_index("tile").static_equity
        sens = pd.DataFrame(cols).loc[df.tile]
        sens.round(2).to_csv(out / "sensitivity.csv")
        print(sens.round(2).to_string())


if __name__ == "__main__":
    main()
