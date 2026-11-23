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

## Word list

NWL23 is the NASPA Word List 2023. It solely belongs to NASPA games, and its inclusion in this private repository is so that the script can execute fully.
