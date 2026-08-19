# A stochastic simulation model for baseball outcomes

A Monte Carlo model that estimates the full distribution of outcomes for a
baseball game from the individual players in it, together with the validation
work done to find out whether it was any good.

It quoted seven families of game-level contracts on a regulated binary options
exchange, continuously, for two months. That setting is what makes the project
worth reporting. The exchange price is an independent forecast of the same event
at the same moment, and every contract eventually settles against what happened,
so the model could be scored against a competing forecast rather than against its
own residuals.

**The forecasts match what happened on the field, and do not beat the price they
were quoting against.** Calibration tracks observed frequency closely and the
predicted distributions come out the right shape. But across 35,159 scored
contracts covering 764 games, the model's Brier score is 0.1905 against the
market's 0.1896, and the confidence interval around that gap contains zero. One
family is significantly worse, six are indistinguishable.
[RESULTS.md](RESULTS.md) gives the breakdown.

**Short version:** a simulation model with hierarchically shrunk parameter
estimates, validated out-of-sample over 764 games against both realized outcomes
and a live benchmark, with per-segment analysis on game-clustered intervals and a
direct account of what it does badly. A
[one-page summary](https://eliwitz34-dev.github.io/baseball-simulation-model/)
covers the same ground in less space.

Repository: <https://github.com/eliwitz34-dev/baseball-simulation-model>

---

## Why simulate rather than fit a regression

Every contract asks about a tail, not an average. Whether two teams combine for
more than 8.5 runs is a question about crossing a threshold. A margin contract
requires the distribution of the difference between two correlated quantities. A
regression predicting expected runs supplies neither, because getting from an
average to the probability of exceeding a threshold requires the shape of the
distribution around that average, and the regression never estimated it.

So the model simulates. It plays a half-inning one plate appearance at a time,
tracking two things: which bases are occupied, and how many are out. Eight
arrangements of runners crossed with three out counts gives 24 states, and those
24 describe the situation completely. What happens next depends on the current
state and the batter, not on how the inning arrived there. That property is what
makes the approach practical, since the consequences of every outcome in every
state can be worked out once and stored in a table.

Each plate appearance draws one of nine outcomes from probabilities specific to
that batter facing that pitcher. Running a matchup 50,000 times gives the joint
distribution of home and away runs. Every price is read off that single
distribution, which is what keeps the totals price and the margin price from
implying different things about the same game.

## Where the parameters come from

Each batter needs nine outcome probabilities against each pitcher, and counting
what a player has actually done does not supply them. A hitter with 150 plate
appearances has a strikeout rate whose standard error is larger than most of the
differences the model is trying to detect. Taken at face value, those rates would
have the simulation reproducing a great deal of noise.

Each player's rate is therefore pulled toward what a model of his observable
characteristics predicts for a player like him, by an amount that depends on how
much is known about him. The weight on his own record is his playing time over
playing time plus a variance ratio, which compares how much players genuinely
differ from one another against how noisy each individual estimate is. A hitter
with a handful of games moves most of the way to the prediction. One with a full
season barely moves.

Those variance terms are estimated alongside the rates rather than fixed
beforehand, so how much to shrink is part of the answer instead of an assumption.
The fit uses a Fay–Herriot specification sampled with Hamiltonian Monte Carlo. The
simulation can then run on the average of those estimates, or at higher cost on
draws from their full distribution, which carries parameter uncertainty into the
output rather than discarding it.

## What is in this repository

```
sim/core.py              the simulation engine, 458 readable lines, runs with no data
sim/tables.py            base-out transition tables, with a data-free fallback
sim/production/          the actual production kernel (3,811 lines), for reference
scripts/build_tables.py  builds transition tables from published advancement rates
scripts/validate.py      the checks that must pass before any number here is quoted
METHODOLOGY.md           how the model was validated, and why that way
RESULTS.md               what the validation found, including everything that failed
```

Start with `sim/core.py`. It runs the same algorithm as the production kernel over
the same state space and the same tables, minus the eleven effect layers, the
environment-variable switches, and the fitted artifacts those layers need. The
production kernel is included unmodified so the reduction can be checked rather
than taken on trust, but it is not where to start and it will not run without
inputs that are not published.

### What is deliberately not here

- **The execution layer.** Order placement, fill modeling and position sizing are
  omitted. That is the part with any commercial value and the least
  methodological interest.
- **The data.** Play-by-play and pitch-level inputs, and the exchange's order
  book. None of it is included. The transition tables are rebuilt locally from
  published constants instead.
- **The trained sub-models.** [METHODOLOGY.md](METHODOLOGY.md) describes the
  gradient-boosted starter-removal hazard and the conditional-logit reliever
  choice, but their fitted parameters are not included.

## Running it

```bash
pip install -r requirements.txt
python scripts/build_tables.py     # builds the transition tables; no download
python scripts/validate.py         # 7 checks, all should pass
python sim/core.py                 # simulate 50,000 league-average games
```

No data files and no network access are required.

### The three generations of transition table

How runners advance is the largest single lever on simulated scoring. Rather than
assert that a simple treatment is good enough, the repository builds it three ways
and measures each against real league scoring. Mean total runs for a
league-average matchup, where the true league average is about 8.8:

| tables | mean total runs | what they represent |
|---|---:|---|
| deterministic fallback | 6.98 | textbook advancement, one outcome per cell, no data at all |
| literature-parameterized | 8.28 | eight published advancement rates, out-dependent |
| fully empirical | 8.88 | all 216 cells estimated from play-by-play |

The fallback exists so that a fresh clone runs immediately. It understates scoring
by 21%, because it cannot represent a double play, a sacrifice fly, or a runner
scoring from second on a single. `scripts/build_tables.py` closes 68% of that gap
with eight constants that are printed and sourced in the file. Most of the value
comes from conditioning them on the out count: a runner scores from second on a
single 39% of the time with nobody out and 80% with two out. The remaining 6% is
rare events such as errors, wild pitches and dropped third strikes, which no
compact set of constants will capture.

Whichever tables are loaded, their provenance travels with them in
`data/provenance.json` and every script reports it, so no number can be quietly
attributed to the wrong generation.

## Data

`scripts/build_tables.py` downloads nothing. It builds the transition tables from
eight published league-average advancement rates, each cited in the file. Those
are eight constants summarizing hundreds of thousands of plays, not a dataset.
The built tables land in `data/` and are excluded from version control, so what
the repository carries is the code that produces them.

The play-by-play and pitch-level inputs used by the production model, and the
record of market prices behind the results, are not included.

## Status

Built and operated as a personal research project between May and August 2026.

Everything in the repository reproduces from the code in it. The market
comparison in [RESULTS.md](RESULTS.md) does not: that requires the price and
outcome record described above.
