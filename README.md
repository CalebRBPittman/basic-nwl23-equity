# Static equity of Scrabble tiles (NWL23)

**Question:** ignoring the board, how many points is each tile worth relative to
an average tile drawn from the bag?

This is the same structure as many baseball questions: estimating the marginal
value of a component (a tile / a base-out state / a pitch) when its value
depends on a future state that you also have to estimate (the leave / run
expectancy), and where the outcome is noisy enough to need uncertainty
estimates.

## Method

1. **Simulate** 100,000 racks (7 tiles, drawn without replacement from the
   standard 100-tile bag).
2. **Enumerate plays.** For each rack, every sub-multiset of >= 2 tiles that
   anagrams to an NWL23 word of up to 7 letters (blanks = wildcards). Because a
   blank scores 0, a play's score depends only on the multiset of real tiles
   used, so the lexicon is pre-indexed as "sorted letters + `?`s" patterns for
   O(1) lookup. +50 for using all 7 tiles. Exchanging everything (0 points,
   empty leave) is always available.
3. **Equity of a rack** = max over plays of (score + leave value).
4. **Leave values are circular** (they depend on equity, which depends on leave
   values), so solve the fixed point by iteration: regress rack equity on
   tile counts (no intercept, since counts sum to 7), centre the coefficients
   against the bag-weighted average tile, feed them back as leave values,
   and repeat with damping until max change < 0.005 pts. Centering matters
   because tiles that are kept are replaced by *average* draws.
5. **Uncertainty:** 200-draw bootstrap over racks (95% percentile CI).
6. **Sensitivity:** repeat under stylized premium-square scoring.

Outputs per tile: `static_equity`, `immediate_value` (effect on best raw score
only, no leave), and `bingo_prob_pp` (effect on probability a 7-tile play
exists, in percentage points).

## Run

    pip install numpy pandas matplotlib
    python static_equity.py --selftest
    python static_equity.py --lexicon NWL23.txt --n-racks 100000 --outdir results

Runs in about a minute on one core. The lexicon is not included in this repo;
obtain NWL23 from NASPA.

## Results (100,000 racks, seed 2023)

| Tile | Static equity (pts) | 95% CI |
|---|---|---|
| ? (blank) | +33.8 | 33.6, 33.9 |
| S | +7.7 | 7.6, 7.9 |
| R | +1.2 | 1.1, 1.3 |
| N / L / D / E / T / C | about 0 | |
| Q | -6.8 | -7.0, -6.6 |
| V | -4.3 | -4.5, -4.2 |
| W | -3.8 | -3.9, -3.6 |
| U | -3.5 | -3.7, -3.4 |

Full table: `results/tile_static_equity.csv`; chart: `results/tile_static_equity.png`.
Other findings: a 7-tile play exists on 13.4% of racks; each duplicated tile
costs about 5 points of rack equity.

## Limitations (important)

* **No board.** There are no premium squares, hooks, parallel plays, or
  defensive considerations. Heavy tiles (J, X, Z, Q) are therefore penalised for
  the lack of premiums (see `sensitivity.csv`: with a stylized premium on the
  best letter, Z/X/J flip from negative to strongly positive). The stylized
  premiums are guaranteed and so are too generous at high settings; the
  baseline and the premium runs bracket the truth rather than pin it down.
* **Linear leaves.** Leave value is additive across tiles; real leaves have
  synergies (e.g. QU, vowel/consonant balance). Only duplication is
  diagnosed, not modelled.
* **Blank is probably overstated** relative to values from full-board
  simulation, because a 50-point bingo is always available to be played out.
* **Plays up to 7 tiles only** (no through-tile 8-letter bingos).
* Bootstrap CI reflects rack-sampling noise only, not model misspecification.

## Possible extensions

Pairwise leave synergies; a first-move board model with real premium squares;
validating against full-game self-play simulation.
