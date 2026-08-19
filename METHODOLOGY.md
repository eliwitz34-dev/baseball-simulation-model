# Methodology

How the model was built, and at greater length, how it was validated. Getting a
simulation to produce plausible numbers is the easy half. Establishing whether
those numbers beat a benchmark already available for free is the harder one, and
it is where most of this project's mistakes occurred.

---

## 1. Model structure

### 1.1 The state space

A half-inning is a Markov chain on 24 states: three out counts crossed with eight
arrangements of runners, plus an absorbing state for the third out. Each plate
appearance draws one of nine outcomes (strikeout, walk, hit by pitch, ground out,
fly out, single, double, triple, home run), and the pair of state and outcome
gives a distribution over the next state and any runs scored.

That last distribution is stochastic, and it matters more than it appears. A
single with a runner on second scores him 39% of the time with nobody out and 80%
with two out, because with two out the runner is moving on contact. Collapsing
that to one averaged rate discards the out-dependence and biases the run
distribution. Collapsing it to deterministic advancement costs about 21% of
league-average scoring. The production model therefore estimates the tables cell
by cell rather than parameterizing them: 216 cells, a median of four distinct
outcomes each. The public repository ships a parameterized reconstruction and
measures how far short it falls.

### 1.2 Layers above the core

Eleven further effects sit on top of the core chain, each estimated separately and
each behind a switch so that it can be turned off for testing.

- **Starter removal.** A discrete-time hazard over the outing, fitted as a
  gradient-boosted classifier on pitch count, runs allowed, score margin, base-out
  state and accrued baserunners, then isotonically recalibrated. It is baked into a
  per-start lookup table so that the inner loop can index it rather than call a
  model.
- **Reliever selection.** A McFadden conditional logit over the available
  bullpen, so that the choice responds to inning, leverage, handedness and recent
  usage rather than being drawn at random from a pool.
- **Times through the order**, park effects, umpire tendency, ball-carry
  conditions, platoon splits, the running game, and several smaller terms.

Each of these earned its place by improving out-of-sample score. Several
candidates did not, and those are recorded in [RESULTS.md](RESULTS.md).

### 1.3 Parameter estimation

Every batter needs nine outcome probabilities against every pitcher, and raw
observed rates can't supply them. A hitter with 150 plate appearances has a
strikeout rate with a standard error around 3.5 percentage points, bigger than
most of the differences the model has to resolve.

The rates are therefore estimated hierarchically. Each player's rate is shrunk
toward a prior mean that is itself a regression on observable covariates, in a
Fay–Herriot specification fitted by Hamiltonian Monte Carlo with the No-U-Turn
Sampler. The serving layer uses posterior means by default, and can instead
propagate coherent posterior draws through the simulation, which gives a
predictive distribution accounting for parameter uncertainty rather than
conditioning on a single point.

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

That is not sufficient, which is the part most often missed. A frozen posterior
evaluated on later games is still contaminated if any *input* carries later
information: a park factor recomputed over the full season, a bullpen roster
reflecting a trade that had not yet happened, a player's season-long rate used to
predict a game inside that season. With this many inputs, getting the training
split right does not prevent leakage; auditing every input for its as-of date
does. Several of the layers above leaked on first implementation and had to be
rebuilt with explicit as-of semantics.

### 2.2 Scoring rules

Forecasts are scored with the Brier score, the mean squared difference between
the probability quoted and what happened. It is a **proper** scoring rule, meaning
a forecaster achieves its best expected score only by reporting what it actually
believes. Improper measures such as accuracy, hit rate or profit can be improved
by misreporting, so none of them can be used to compare two forecasters. Log loss
is reported alongside as a second proper rule with different tail sensitivity.

The Brier score decomposes as

  Brier = uncertainty − resolution + reliability

**Reliability** is calibration: whether events assigned 30% happen 30% of the
time. **Resolution** is discrimination: whether the forecaster separates the
events that happen from those that do not. Splitting the score this way is what
makes it diagnostic rather than merely comparative, since a model can be perfectly
calibrated and useless by forecasting the base rate every time. The central
finding in [RESULTS.md](RESULTS.md), comparable calibration with weak
discrimination, is visible only through the decomposition.

Calibration is also checked directly, with reliability curves and with
probability integral transform histograms. A correctly specified predictive
distribution gives a uniform PIT histogram. The U-shape produced by an
overconfident model, with too much mass in the extreme bins, is the single most
useful diagnostic picture in the project.

### 2.3 Clustered inference

The several hundred contracts priced on one game share a single outcome. Their
errors move together, so the effective sample size is the number of *games*, not
the number of contracts.

Every interval reported here is therefore bootstrapped by resampling games and
carrying each game's contracts along with it. Resampling contracts instead narrows
the intervals by between about 1.3 and 2.5 times depending on the segment. On this
sample that does not flip any conclusion, since the same single family comes out
significant either way, but the narrower intervals would misstate the precision of
everything reported, and there is no way to know in advance that a sample will be
forgiving.

### 2.4 The measurement basis

The most persistent error in this project lay not in any model but in the
choice of sample used to evaluate one: conditioning an evaluation on an event
correlated with the outcome.

The clearest case was calibration measured only on the contracts where an order
actually filled. An order fills when a counterparty takes the other side, and that
happens preferentially when the price is wrong. Conditioning on fills therefore
selects for the model's own errors and reports worse calibration than the model
has. The resulting estimate is biased rather than merely imprecise, so more data
does not correct it.

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

The defense adopted was a written pre-flight check before any result was
believed: what defines this sample, what could have selected it, and what would
this measurement show if the model had no skill at all. That last question is a
placebo test, and it killed more than one promising result.

### 2.5 Pre-committing to the reading

For each model change, the criterion and the interpretation were fixed before the
test ran, including the conditions under which a *favorable* result would be
disbelieved.

One instance shows why that matters. A multiplier intended to correct apparent
over-shrinkage in the pitcher strikeout estimates did improve the average
predicted count, which was precisely the number the work had set out to move. The
criterion agreed beforehand was probabilistic accuracy rather than mean accuracy,
and on that criterion the multiplier made things worse at every level. It was
rejected, and later found to be correcting a defect that was itself an artifact of
measurement. Without the prior commitment, the natural move would have been to
read the improvement in the means as success.

---

## 3. Risk measurement

Position sizing maximized expected log wealth, the Kelly criterion at a fraction,
over jointly simulated outcomes rather than treating positions as independent.
Joint simulation matters because contracts on the same game are strongly
dependent, and a sizing rule assuming independence will systematically take too
much correlated exposure.

The optimization is constrained by a tail-risk limit expressed as a **Conditional
Value at Risk** bound, implemented in the Rockafellar–Uryasev auxiliary-variable
form that makes the constraint convex and therefore tractable alongside the
exposure caps. It is preferred over value at risk for two reasons: VaR is not
subadditive, so it can report that a diversified portfolio is riskier than its
parts, and it says nothing about how bad losses are once the quantile is
breached.

Maximizing expected log wealth is a utility-theoretic statement rather than a
trading heuristic. Log utility is the unique utility function under which the
optimal fraction is independent of wealth, which is why it appears in both the
ruin literature and the growth-optimal literature.

One empirical finding is worth recording. Across the sizing work, the binding
constraint was almost always a hard per-game exposure cap rather than the
objective function. On the full evaluation tape that cap bound on 67 of 67 days.
Any refinement of the objective tested under those caps is being measured in their
null space: the caps were doing the sizing, not the optimizer. Establishing which
constraint actually binds before tuning what it constrains generalizes well beyond
this project.

---

## 4. Known limitations

- **One partial season.** Roughly two months of live operation and 764 games.
  Nothing here establishes stability across seasons, rule changes or roster
  turnover.
- **The benchmark holds information the model does not.** Market prices
  incorporate late lineup and weather news the model never saw. Part of the
  measured deficit is that information gap rather than a modeling deficiency, and
  this analysis cannot separate the two.
- **Layer interactions are not fully explored.** All eleven layers were validated
  against a baseline individually, but the full interaction space was never
  searched, so some of them may be partly redundant with each other.
- **The public reimplementation is not the production model.** `sim/core.py`
  omits all eleven layers. Its output demonstrates the mechanism and should not be
  read as the production model's forecasts.
- **One person.** Every modeling choice and every interpretation came from the
  same person. Pre-commitment and placebo testing substitute for independent
  review only partially.
