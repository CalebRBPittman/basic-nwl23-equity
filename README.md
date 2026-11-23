# Scrabble tile static equity

How many points is each Scrabble tile worth, on average, compared with a
random tile from the bag? This estimates that from the NWL23 word list using
the standard tile distribution, face point values, and the 50-point bonus for playing
all seven tiles.

## Method

1. Deal 100,000 random 7-tile racks from the standard 100-tile bag.
2. For each rack, find every play the lexicon allows and score it. 
3. A rack's equity is defined as its best play's score plus the value of the tiles kept
   for next turn.
4. Those leave values are what we're after, so they're solved by iteration:
   regress rack equity on tile counts, compare each tile to the average
   tile, feed those numbers back in as leave values, and repeat until stable.
5. Bootstrap the racks to get 95% confidence intervals.

## Layout

    data/NWL23.txt    word list (one word per line)
    static_equity.py  the whole analysis
    results/          CSV and chart written by the script

## Running it

    pip install -r requirements.txt
    python static_equity.py

The script reads the lexicon from `data/NWL23.txt` (set by `LEXICON_PATH`
at the top of `static_equity.py`) and writes to `results/`. The random seed
is fixed, so reruns give the same numbers.

## Limitations of this approach

The approach was purposefully simple, and had several limitations that I was well aware of. This includes:
- **No 8-letter bingos.** Since we only included the 7 tiles on a tile rack, we failed to account for the large amount of 8-letter bingoes, which are integral to Scrabble.
- **No board positioning or bonus squares.** Every play is scored at face value, with the 50 point bingo bonus added. High-point tiles like the Z and the J can't use double or triple letter and word squares here, so they are potentially under-valued.
- **Additive leave values.** Each tile's value is estimated separately and summed, so tile synergies are not modelled.
- **No game context.** The values are averages over random racks. Score, tiles remaining in the bag, the opponent and the endgame play no part.

## Basic results

We can compare the results of this basic model to that of [Magpie](https://github.com/jvc56/MAGPIE), a much more robust engine created by 2-time North American champion Joshua Castellano.

Source for the MAGPIE values: [[cross-tables.com](https://www.cross-tables.com/leaves.php)].

| Tile | This model | 95% CI | MAGPIE | Difference | Rank (this model / MAGPIE) |
|:----:|-----------:|:------:|-------:|-----------:|:--------------------------:|
| ? | +33.75 | +33.59 to +33.93 | +28.7 | +5.0 | 1 / 1 |
| S | +7.74 | +7.61 to +7.90 | +8.5 | −0.8 | 2 / 2 |
| R | +1.16 | +1.05 to +1.29 | +0.8 | +0.4 | 3 / 6 |
| N | +0.42 | +0.31 to +0.56 | −0.2 | +0.6 | 4 / 12 |
| L | +0.22 | +0.05 to +0.38 | −0.6 | +0.8 | 5 / 17 |
| D | +0.20 | +0.05 to +0.33 | +0.3 | −0.1 | 6 / 10 |
| C | +0.12 | −0.10 to +0.35 | +0.4 | −0.3 | 7 / 9 |
| E | +0.08 | +0.00 to +0.16 | +0.6 | −0.5 | 8 / 7 |
| T | −0.03 | −0.18 to +0.10 | −0.3 | +0.3 | 9 / 13 |
| Z | −0.35 | −0.58 to −0.13 | +5.8 | −6.2 | 10 / 3 |
| M | −0.60 | −0.80 to −0.41 | +0.6 | −1.2 | 11 / 7 |
| A | −0.97 | −1.07 to −0.90 | −0.5 | −0.5 | 12 / 15 |
| P | −1.32 | −1.50 to −1.12 | −0.5 | −0.8 | 13 / 15 |
| H | −1.40 | −1.61 to −1.24 | +1.1 | −2.5 | 14 / 5 |
| I | −1.48 | −1.56 to −1.40 | −2.3 | +0.8 | 15 / 21 |
| G | −1.54 | −1.71 to −1.36 | −3.1 | +1.6 | 16 / 23 |
| B | −2.15 | −2.33 to −1.96 | −2.1 | −0.1 | 17 / 20 |
| O | −2.19 | −2.28 to −2.09 | −2.5 | +0.3 | 18 / 22 |
| X | −2.21 | −2.43 to −2.03 | +3.7 | −5.9 | 19 / 4 |
| K | −2.25 | −2.49 to −2.03 | +0.1 | −2.3 | 20 / 11 |
| Y | −2.42 | −2.60 to −2.26 | −0.4 | −2.0 | 21 / 14 |
| F | −2.86 | −3.01 to −2.69 | −2.0 | −0.9 | 22 / 19 |
| J | −2.89 | −3.10 to −2.73 | −1.3 | −1.6 | 23 / 18 |
| U | −3.53 | −3.66 to −3.40 | −5.5 | +2.0 | 24 / 25 |
| W | −3.77 | −3.90 to −3.60 | −3.4 | −0.4 | 25 / 24 |
| V | −4.34 | −4.49 to −4.16 | −5.7 | +1.4 | 26 / 26 |
| Q | −6.80 | −7.00 to −6.62 | −6.3 | −0.5 | 27 / 27 |

**Where the two agree**
- The blank and S are first and second in both. Q is last in both, and U, V and W are the other three of the bottom four in both.
- Excluding the blank, the correlation between the two sets is 0.80 (Pearson) and 0.69 (Spearman). The mean absolute difference is 1.3 points, and 16 of the 26 remaining tiles are within 1 point.

**Where they disagree**
- The largest gaps are Z (−6.2), X (−5.9), H (−2.5), K (−2.3) and Y (−2.0), all lower in this model. Z and X rank 10th and 19th here but 3rd and 4th in MAGPIE. This is expected; my model under-values high-point consonants that will otherwise be useful in Scrabble.
- The blank is 5.0 points higher here than in MAGPIE, which once again makes sense. My model fails to account for bonus squares, which is where high-point tiles are most valuable. In my model, the best way to score is to create 7-tile bingoes, and the blank tile is the most conducive to that.
- Agreement improves once the high-point tiles are set aside. Excluding the blank, Z and X, the correlation rises to 0.92 and the mean absolute difference falls to 0.9 points. Also excluding J and K, it reaches 0.94 and 0.8.

## Word list

NWL23 is the NASPA Word List 2023. It solely belongs to NASPA games, and its inclusion in this private repository is so that the script can execute fully.
