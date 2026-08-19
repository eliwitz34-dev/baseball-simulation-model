# A stochastic simulation model for baseball outcomes

A Monte Carlo model that prices the distribution of outcomes for a baseball game
from the individual players involved, together with the validation apparatus
built to find out whether it was any good.

It was built to price sixteen families of contracts on a regulated binary options
exchange, six of which it quoted continuously for two months. The exchange price
supplies an independent benchmark forecast for every contract, and every contract
settles against an observed outcome, so the model can be scored against a
competing forecast rather than against its own residuals.

**The headline result is that the model does not beat that benchmark overall.**
Across 163,777 scored contracts on 786 games, its Brier score is 0.1891 against
the market's 0.1883 on game-level markets — slightly worse, by a margin whose
game-clustered confidence interval contains zero. It is significantly better on
exactly one of the eleven families large enough to test, and significantly worse
on four.
[RESULTS.md](RESULTS.md) gives the breakdown by family.

**Short version, if you are skimming:** a stochastic simulation model with
hierarchically shrunk parameter estimates, validated out-of-sample against a live
benchmark forecast over 786 games, with per-segment calibration analysis on
game-clustered intervals, and an explicit account of what it does badly. There is a [one-page summary](https://eliwitz34-dev.github.io/baseball-simulation-model/) if you would
rather not read three documents.

Repository: <https://github.com/eliwitz34-dev/baseball-simulation-model>

---

## Why a simulation rather than a regression

The quantity of interest is never a mean. Pricing a contract on total runs
scored requires **P(total > 8.5)**; pricing a margin contract requires the
distribution of the difference between two correlated random variables. A
regression fitted to expected runs answers neither, because the mapping
from a mean to a tail probability depends on a shape the regression never
estimated.

So the model simulates. A baseball game is a Markov chain over 24 base-out
states — the eight arrangements of runners on base crossed with three out
counts. Each plate appearance draws one of nine outcomes from a distribution
that depends on the batter and the pitcher he is facing, moves the chain, and
possibly scores runs. Simulating the chain many times produces the joint
distribution of (home runs, away runs), and every price is a functional of that
one object, which guarantees the prices are mutually consistent.

The object being built is a compound distribution: a random number of events,
each contributing a random amount, aggregated into a total whose tail is the
quantity of interest. Simulation is used rather than analytic convolution because
the summands are neither independent nor identically distributed — the base-out
state couples consecutive plate appearances, and the batter changes every time.

## Where the parameters come from

Each batter needs nine outcome probabilities against each pitcher. Estimating
those from a single player's season is hopeless: the sample is small, and the
raw rates are dominated by noise. The model instead estimates them
hierarchically, shrinking each player's observed rates toward a structured prior
mean using a Fay–Herriot small-area specification fitted by Hamiltonian Monte
Carlo. The simulation can be run on posterior means or, at higher cost, on
posterior draws, so that parameter uncertainty is carried into the output rather
than discarded.

The weight on a player's own record is his playing time divided by playing time
plus a variance ratio, so a hitter with a handful of games is pulled most of the
way toward what the covariate model predicts for him, while one with a full
season barely moves. The variance components are estimated jointly with the rates
rather than fixed in advance, so the uncertainty in that weight is itself carried
rather than assumed away.

## What is in this repository

```
sim/core.py              the simulation engine, 458 readable lines, runs with no data
sim/reference_numpy.py   an independent implementation, used to cross-check core.py
sim/tables.py            base-out transition tables, with a data-free fallback
sim/production/          the actual production kernel (3,811 lines), for reference
scripts/build_tables.py  builds transition tables from published advancement rates
scripts/validate.py      the checks that must pass before any number here is quoted
scripts/benchmark.py     simulation throughput, for anyone who wants it
METHODOLOGY.md           how the model was validated, and why it was validated that way
RESULTS.md               what the validation found, including everything that failed
```

`sim/core.py` is the file to read. It implements the same algorithm as the
production kernel over the same state space and the same transition tables, but
without the eleven further effect layers, the environment-variable switches, or
the fitted artifacts those layers need. The production kernel is included
unmodified so that the reduction can be checked rather than taken on trust, but
it is not the file to start with, and it will not run without proprietary
inputs.

### What is deliberately not here

- **The execution layer.** Order placement, fill modeling, and position sizing
  are omitted. They are the part with any commercial value and the part with the
  least methodological interest.
- **The data.** Play-by-play and pitch-level inputs come from sources whose terms
  do not permit redistribution, and the exchange's order-book data likewise. No
  data of either kind is included; the transition tables are rebuilt locally from
  published constants instead. See the data note below.
- **The trained sub-models.** The gradient-boosted starter-removal hazard and the
  conditional-logit reliever-choice model are described in
  [METHODOLOGY.md](METHODOLOGY.md) but their fitted parameters are not included.

## Running it

```bash
pip install -r requirements.txt
python scripts/build_tables.py     # builds the transition tables; no download
python scripts/validate.py         # 10 checks, all should pass
python sim/core.py                 # simulate 50,000 league-average games
```

Nothing here needs data files or network access.

### The three generations of transition table

How runners advance is the single largest lever on simulated scoring, and the
repository makes the cost of approximating it explicit rather than asserting it
is small. Mean total runs for a league-average matchup, against a true league
average near 8.8:

| tables | mean total runs | what they represent |
|---|---:|---|
| deterministic fallback | 6.98 | textbook advancement, one outcome per cell, no data at all |
| literature-parameterized | 8.28 | eight published advancement rates, out-dependent |
| fully empirical | 8.88 | all 216 cells estimated from play-by-play |

The fallback exists so a fresh clone runs immediately; it understates scoring by
21%, because it cannot represent a double play, a sacrifice fly, or a runner
scoring from second on a single. `scripts/build_tables.py` closes 68% of that gap
using eight constants that are printed, sourced, and out-dependent — and the
out-dependence is most of the value, since a runner scores from second on a
single 39% of the time with nobody out and 80% with two out. The remaining 6%
shortfall is rare events (errors, wild pitches, dropped third strikes) that no
compact parameterization captures, and it is the honest price of not shipping
data.

Whichever tables are loaded, their provenance travels with them in
`data/provenance.json` and is reported by every script, so no number can be
quietly attributed to the wrong generation.

## Checking the implementation

The model is implemented twice. `sim/core.py` is the compiled kernel used for
everything; `sim/reference_numpy.py` is an independent implementation written
separately from the same specification.

That redundancy is the point. A transcription error — an off-by-one in a lookup
index, a state written back to the wrong variable — produces output that still
looks like baseball, and inside a compiled parallel loop there is no way to
inspect the intermediate steps. Two implementations agreeing on a distribution to
within Monte Carlo error is the cheapest available evidence that neither contains
one. They agree on mean total runs to 0.02 against a four-standard-error
tolerance of 0.12, and on the home win probability to 0.001.
`scripts/validate.py` runs the comparison and takes its tolerances from Monte
Carlo error rather than from whatever number happens to pass.

The kernel is compiled with Numba, which matters here for one reason only: it
sets how many simulations are affordable, and that sets the precision floor. At
50,000 simulations the Monte Carlo standard error near a probability of one half
is about 0.22 percentage points, so two forecasts from this engine closer
together than that are not meaningfully different. `scripts/benchmark.py`
measures throughput on your own machine if that is of interest.

## Data note

No data from the exchange or from any commercial provider is included or
redistributed. `scripts/build_tables.py` needs no download at all: it builds the
transition tables from eight published league-average advancement rates, each
cited in the file. Those are constants summarizing hundreds of thousands of plays, not a
dataset. The built tables are written to `data/` and are excluded from version
control, so what the repository distributes is always the code that produces
them.

## Status and provenance

Built and operated as a personal research project between May and August 2026.
It is not running now, and this repository is a writeup of completed work rather
than a live system. Nothing here constitutes advice of any kind.

Numbers in this repository are reproducible from the code in it, with the
exception of the live-market results in [RESULTS.md](RESULTS.md), which are
computed from a private tape and are labeled as such wherever they appear.
