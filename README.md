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

**Short version, if you are skimming:** a stochastic simulation model built with
credibility-weighted parameter estimates, validated out-of-sample against a live
benchmark forecast over 786 games, with per-segment actual-versus-expected
analysis on game-clustered intervals — and an explicit account of what it does
badly. There is a
[one-page summary](https://claude.ai/code/artifact/da09aef2-5329-4a53-a756-489e793e5d7f)
if you would rather not read three documents.

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

This is structurally a **collective risk model**: a random number of events, each
contributing a random severity, aggregated to a distribution whose tail is the
thing that matters. The reason to simulate rather than convolve analytically is
that the summands are neither independent nor identically distributed — the
base-out state couples consecutive plate appearances, and the batter changes
every time.

## Where the parameters come from

Each batter needs nine outcome probabilities against each pitcher. Estimating
those from a single player's season is hopeless: the sample is small, and the
raw rates are dominated by noise. The model instead estimates them
hierarchically, shrinking each player's observed rates toward a structured prior
mean using a Fay–Herriot small-area specification fitted by Hamiltonian Monte
Carlo. The simulation can be run on posterior means or, at higher cost, on
posterior draws, so that parameter uncertainty is carried into the output rather
than discarded.

An actuary will recognize this as **credibility theory**. The Fay–Herriot
estimator is the Bühlmann–Straub credibility estimator with an explicit
regression component for the prior mean: the weight placed on a player's own
record is his exposure divided by exposure plus a variance ratio, which is
exactly the credibility factor Z. The difference from textbook credibility is
that the hyperparameters are estimated jointly with everything else rather than
plugged in, and that the resulting parameter uncertainty flows through to the
simulated distribution instead of being discarded.

## What is in this repository

```
sim/core.py              the simulation engine, 458 readable lines, runs with no data
sim/reference_numpy.py   an independent vectorized implementation, used to cross-check core.py
sim/tables.py            base-out transition tables, with a data-free fallback
sim/production/          the actual production kernel (3,811 lines), for reference
scripts/build_tables.py  builds transition tables from published advancement rates
scripts/validate.py      the checks that must pass before any number here is quoted
scripts/benchmark.py     measures throughput on your machine
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

## Performance

The inner loop is compiled with Numba and parallelized across simulations.
Measured on an Apple M5 Pro (15 cores) with `scripts/benchmark.py`:

| simulations | compiled | vectorized NumPy | speedup |
|---:|---:|---:|---:|
| 10,000 | 6.2 ms | 101.0 ms | 16.3× |
| 50,000 | 25.5 ms | 534.6 ms | 21.0× |
| 200,000 | 91.5 ms | 2,089 ms | 22.8× |

That is about 2 million simulated games per second at the 50,000 mark.

The baseline is a genuinely vectorized NumPy implementation
(`sim/reference_numpy.py`), not interpreted Python loops. Benchmarking against
interpreted loops would report a much larger multiple and would mean nothing,
because nobody would write the reference that way; this comparison is the one
that answers a useful question. Run the script yourself rather than trusting the
table — it reports your machine and separates one-time compilation from
steady-state throughput.

The second implementation exists mainly as a cross-check. Two independently
written versions of the same model agreeing to within Monte Carlo error is
meaningful evidence that neither contains a transcription bug, which is
otherwise very hard to establish inside a compiled parallel loop. They agree on
mean total runs to 0.02 runs against a four-standard-error tolerance of 0.12, and
on the home win probability to 0.001 — `scripts/validate.py` runs the comparison
and sets its tolerances from Monte Carlo error rather than by choosing a number
that passes.

## Correspondence with actuarial practice

The methods here are standard actuarial machinery wearing different names. For a
property and casualty reader in particular:

| in this project | actuarial equivalent |
|---|---|
| Fay–Herriot hierarchical shrinkage of player rates | Bühlmann–Straub **credibility**, with a regression prior mean |
| Monte Carlo over base-out states → run distribution | **collective risk model**; aggregate loss distribution |
| tail measures on the simulated position portfolio | **TVaR / Conditional Tail Expectation** |
| leak-safe walk-forward backtesting | out-of-sample validation under **ASOP 56** |
| Brier decomposition, reliability curves, PIT histograms | **actual-versus-expected** monitoring |
| the record of rejected model changes | model change control and documented limitations |

The correspondence is not decorative. The reason the model shrinks player rates
is the same reason a rating plan credibility-weights a small class: the observed
mean of a low-exposure cell is mostly noise, and the optimal estimator trades
bias for variance in a way that depends on the ratio of process variance to
hypothetical mean variance. The reason the validation reports reliability rather
than accuracy is the same reason a reserving actuary runs actual-versus-expected
by segment: an aggregate in which two offsetting errors cancel is
indistinguishable from an aggregate with no errors at all.

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
