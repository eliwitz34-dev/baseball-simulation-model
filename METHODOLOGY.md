# Methodology

How the model was built, and at more length, how it was validated. Getting a
simulation to produce plausible numbers is the easy half. Working out whether
those numbers beat a benchmark you could have had for free is the hard one, and
it's where most of this project's mistakes happened.

---

## 1. Model structure

### 1.1 The state space

A half-inning is a Markov chain on 24 states: three out counts crossed with eight
arrangements of runners, plus an absorbing state for the third out. Each plate
appearance draws one of nine outcomes (strikeout, walk, hit by pitch, ground out,
fly out, single, double, triple, home run), and the pair of state and outcome
gives a distribution over the next state and any runs scored.

That last distribution is stochastic, and it matters more than you'd think. A
single with a runner on second scores him 39% of the time with nobody out and 80%
with two out, because with two out he's running on contact. Collapse that to one
averaged rate and you lose the out-dependence and bias the run distribution.
Collapse it to deterministic advancement and you lose about 21% of league-average
scoring. So the production model estimates the tables cell by cell instead of
parameterizing them: 216 cells, a median of four distinct outcomes each. The
public repository ships a parameterized reconstruction and measures how far short
it falls.

### 1.2 Layers above the core

Eleven further effects sit on top of the core chain. Each is estimated separately
and each sits behind a switch so it can be turned off for testing.

- **Starter removal.** A discrete-time hazard over the outing, fitted as a
  gradient-boosted classifier on pitch count, runs allowed, score margin,
  base-out state and accrued baserunners, then isotonically recalibrated. It gets
  baked into a per-start lookup table so the inner loop can index it instead of
  calling a model.
- **Reliever selection.** A McFadden conditional logit over whoever is available,
  so the choice responds to inning, leverage, handedness and recent usage instead
  of being drawn at random from a pool.
- **Times through the order**, park effects, umpire tendency, ball-carry
  conditions, platoon splits, the running game, and several smaller terms.

Each of these earned its place by improving out-of-sample score. Several
candidates didn't, and those are in [RESULTS.md](RESULTS.md).

### 1.3 Parameter estimation

Every batter needs nine outcome probabilities against every pitcher, and raw
observed rates can't supply them. A hitter with 150 plate appearances has a
strikeout rate with a standard error around 3.5 percentage points, bigger than
most of the differences the model has to resolve.

So the rates get estimated hierarchically. Each player's rate is shrunk toward a
prior mean that is itself a regression on observable covariates, in a
Fay–Herriot specification fitted by Hamiltonian Monte Carlo with the No-U-Turn
Sampler. The serving layer uses posterior means by default. It can also push
coherent posterior draws through the simulation, which gives a predictive
distribution that accounts for parameter uncertainty instead of conditioning on a
single point.

The estimator has the form

  θ̂ᵢ = Zᵢ · yᵢ + (1 − Zᵢ) · x'ᵢβ,  Zᵢ = σ²_between ⁄ (σ²_between + σ²_within,ᵢ)

where yᵢ is the player's observed rate, x'ᵢβ is what the covariate model predicts
for him, and Zᵢ is the weight between them. Zᵢ rises with the player's playing
time and falls as his own record becomes noisier relative to the spread between
players, so a thin sample is pulled most of the way toward the prediction and a
full season is barely moved.

Two things about this are worth flagging. The variance components are estimated
jointly with the rates rather than plugged in, so the uncertainty about how much
to shrink is itself represented. And propagating the posterior rather than
collapsing it matters because the output is a tail probability: the tail
probability at the posterior mean is not the posterior mean of the tail
probability. That gap is Jensen's inequality. Ignore it and the forecasts come
out systematically overconfident, which is a failure this project ran into
directly.

---

## 2. Validation

### 2.1 Walk-forward, and what makes it leak-safe

The model is refitted at successive points in time and evaluated only on games
after each fit, so no evaluation ever uses a parameter estimated with knowledge
of the outcome being predicted.

That isn't sufficient, which is the part people miss. A frozen posterior
evaluated on later games is still contaminated if any *input* carries later
information. A park factor recomputed over the full season. A bullpen roster
reflecting a trade that hadn't happened yet. A player's season-long rate used to
predict a game inside that season. With this many inputs, getting the training
split right doesn't prevent leakage. Auditing every input for its as-of date
does. Several of the layers above leaked on first implementation and had to be
rebuilt with explicit as-of semantics.

### 2.2 Scoring rules

Forecasts are scored with the Brier score, the mean squared difference between
the probability quoted and what happened. It's a **proper** scoring rule, meaning
a forecaster gets its best expected score only by reporting what it actually
believes. Improper measures like accuracy, hit rate or profit can all be improved
by shading your answers, so they can't be used to compare two forecasters. Log
loss is reported alongside as a second proper rule with different tail
sensitivity.

The Brier score decomposes as

  Brier = uncertainty − resolution + reliability

**Reliability** is calibration: do events assigned 30% happen 30% of the time.
**Resolution** is discrimination: does the forecaster actually separate the events
that happen from the ones that don't. Splitting the score this way is what makes
it diagnostic instead of just comparative, because a model can be perfectly
calibrated and useless by forecasting the base rate every time. The central
finding in [RESULTS.md](RESULTS.md), comparable calibration with weak
discrimination, only shows up once you decompose.

Calibration also gets checked directly, with reliability curves and with
probability integral transform histograms. A correctly specified predictive
distribution gives a uniform PIT histogram. The U-shape you get from an
overconfident model, with too much mass piled in the extreme bins, is the single
most useful diagnostic picture in the project.

### 2.3 Clustered inference

The several hundred contracts priced on one game all share a single outcome.
Their errors move together, so the effective sample size is the number of *games*,
not the number of contracts.

Every interval reported here is therefore bootstrapped by resampling games and
carrying each game's contracts along with it. Resampling contracts instead
narrows the intervals by somewhere between 1.3 and 2.5 times depending on the
segment. On this sample that doesn't flip any conclusion, since the same single
family comes out significant either way. But the narrower intervals would misstate
the precision of everything reported, and there's no way to know in advance that a
sample will be forgiving.

### 2.4 The measurement basis

The most persistent error in this project lay not in any model but in the
choice of sample used to evaluate one: conditioning an evaluation on an event
correlated with the outcome.

The clearest case was calibration measured only on the contracts where an order
actually filled. An order fills when a counterparty takes the other side, and that
happens preferentially when the price is wrong. So conditioning on fills selects
for the model's own errors and reports worse calibration than the model has. The
measurement isn't noisy, it's biased, and more data won't fix it.

The general forms this took, all of which recurred:

- **Selected slices.** Conditioning on an outcome-correlated event, as above.
- **Circular conditioning.** Evaluating a model against a benchmark that was
  itself derived from that model's output — for instance comparing against
  cached prices that the model had generated on an earlier pass.
- **Confounded second-order comparisons.** Splitting results by a variable that
  is a collider between the treatment and the outcome, which induces an
  association where none exists.
- **Canceling aggregates.** Two large errors of opposite sign in different
  segments, netting to a small aggregate error that looks like success. Only a
  segment-level breakdown finds these.

What I settled on was a written pre-flight check before believing any result:
what defines this sample, what could have selected it, and what would this
measurement show if the model had no skill at all. That last one is a placebo
test, and it killed more than one promising result.

### 2.5 Pre-committing to the reading

For each model change I fixed the criterion and the interpretation before running
the test, including the conditions under which a *favorable* result would be
disbelieved.

Here's why that matters. A multiplier meant to correct apparent over-shrinkage in
the pitcher strikeout estimates did improve the average predicted count, which was
exactly the number the work had set out to move. But the criterion agreed
beforehand was probabilistic accuracy, not mean accuracy, and on that criterion
the multiplier made things worse at every level. It was rejected, and later turned
out to be correcting a defect that was an artifact of measurement in the first
place. Without the prior commitment the natural move would have been to read the
improvement in the means as success.

---

## 3. Risk measurement

Position sizing maximized expected log wealth, the Kelly criterion at a fraction,
over jointly simulated outcomes rather than treating positions as independent.
The joint part matters because contracts on the same game are strongly dependent,
and a sizing rule that assumes independence will keep taking too much correlated
exposure.

The optimization is constrained by a tail-risk limit expressed as a **Conditional
Value at Risk** bound, implemented in the Rockafellar–Uryasev auxiliary-variable
form that makes the constraint convex and therefore tractable alongside the
exposure caps. It is preferred over value at risk for two reasons: VaR is not
subadditive, so it can report that a diversified portfolio is riskier than its
parts, and it says nothing about how bad losses are once the quantile is
breached.

Maximizing expected log wealth is a utility-theoretic statement rather than a
trading heuristic. Log utility is the one utility function under which the optimal
fraction doesn't depend on wealth, which is why it turns up in both the ruin
literature and the growth-optimal literature.

One empirical finding is worth recording. Across all the sizing work, the binding
constraint was almost always a hard per-game exposure cap, not the objective
function. On the full evaluation tape that cap bound on 67 of 67 days. Any
refinement of the objective tested under those caps is being measured in their
null space: the caps were doing the sizing, not the optimizer. Working out which
constraint actually binds before tuning what it constrains generalizes well beyond
this project.

---

## 4. Known limitations

- **One partial season.** About two months of live operation and 764 games.
  Nothing here says anything about stability across seasons, rule changes or
  roster turnover.
- **The benchmark knows things the model doesn't.** Market prices pick up late
  lineup and weather news the model never saw. Some of the measured deficit is
  that information gap rather than a modeling problem, and this analysis can't
  separate the two.
- **Layer interactions aren't fully explored.** All eleven layers were validated
  against a baseline individually, but I never searched the full interaction
  space, so some of them may be partly redundant with each other.
- **The public reimplementation isn't the production model.** `sim/core.py` drops
  all eleven layers. Its output shows the mechanism working; it isn't what the
  production model would forecast.
- **One person.** Every modeling choice and every interpretation was mine.
  Pre-commitment and placebo testing substitute for independent review only
  partially.
