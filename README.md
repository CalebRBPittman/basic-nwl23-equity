# Scrabble tile static equity

How many points is each Scrabble tile worth, on average, compared with a
random tile from the bag? This estimates that from the NWL23 word list using
the standard tile distribution, face values, and a 50-point bonus for playing
all seven tiles.

## Method

1. Deal 100,000 random 7-tile racks from the standard 100-tile bag.
2. For each rack, find every play the lexicon allows (blanks are wildcards)
   and score it: face values, plus 50 for using all seven tiles.
3. A rack's equity is its best play's score plus the value of the tiles kept
   for next turn.
4. Those leave values are what we're after, so they're solved by iteration:
   regress rack equity on tile counts, compare each tile to the average
   tile, feed those numbers back in as leave values, repeat until stable.
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

## Word list

NWL23 is the NASPA Word List 2023. It's included so the code runs as is;
the list belongs to its publishers.
