# Methodology

This document describes how the model was built and, at greater length, how it
was validated. Producing a simulation that returns plausible numbers is
straightforward; establishing whether those numbers improve on a benchmark
already available is the larger task, and is where this project's errors were
concentrated.

---

## 1. Model structure

### 1.1 The state space

A half-inning is a Markov chain on 24 states — three out counts crossed with
eight arrangements of runners — plus an absorbing state for the third out. Each
plate appearance draws one of nine outcomes (strikeout, walk, hit by pitch,
ground out, fly out, single, double, triple, home run) and the pair (state,
outcome) determines a distribution over (next state, runs scored).

That last distribution is genuinely stochastic and this matters more than it
appears. A single with a runner on second scores him 39% of the time with nobody
out and 80% with two out, because with two out the runner is moving on contact.
Representing this as a single averaged rate discards the out-dependence and
biases the run distribution; representing it as deterministic advancement
discards it entirely and costs about 21% of league-average scoring. The
transition tables in the production model are therefore estimated per cell
rather than parameterized — 216 cells, a median of four distinct outcomes each.
The public repository ships a parameterized reconstruction instead, and measures
how far that falls short.

### 1.2 Layers above the core

The core chain is modulated by eleven further effects, each estimated separately
and each gated behind a switch so it can be turned off for testing:

- **Starter removal.** A discrete-time hazard over the outing, fitted as a
  gradient-boosted classifier on pitch count, runs allowed, score margin,
  base-out state and accrued baserunners, then isotonically recalibrated. It is
  baked into a per-start lookup table so the simulation's inner loop can index
  it rather than call a model.
- **Reliever selection.** A McFadden conditional logit over the available
  bullpen, so the choice of reliever responds to inning, leverage, handedness
  and prior usage rather than being drawn from a pool.
- **Times through the order**, park effects, umpire tendency, ball-carry
  conditions, platoon splits, the running game, and several smaller terms.

Each layer earned its place by improving out-of-sample score, and several
candidate layers did not — see [RESULTS.md](RESULTS.md).

### 1.3 Parameter estimation

Every batter needs nine outcome probabilities against every pitcher. Raw
observed rates cannot supply them: a hitter with 150 plate appearances has a
strikeout rate whose standard error is around 3.5 percentage points, larger than
most of the differences the model must resolve.

The rates are therefore estimated hierarchically. Each player's rate is shrunk
toward a prior mean that is itself a regression on observable covariates,
in a Fay–Herriot small-area specification fitted by Hamiltonian Monte Carlo with
the No-U-Turn Sampler. The serving layer uses posterior means by default and can
propagate coherent posterior draws through the simulation instead, which
produces a predictive distribution that accounts for parameter uncertainty
rather than conditioning on a point estimate.

The estimator has the form

  θ̂ᵢ = Zᵢ · yᵢ + (1 − Zᵢ) · x'ᵢβ,  Zᵢ = σ²_between ⁄ (σ²_between + σ²_within,ᵢ)

where yᵢ is the player's observed rate, x'ᵢβ is what the covariate model predicts
for him, and Zᵢ is the weight between them. Zᵢ rises with the player's playing
time and falls as his own record becomes noisier relative to the spread between
players, so a thin sample is pulled most of the way toward the prediction and a
full season is barely moved.

Two points are worth naming. The variance components are estimated jointly with
the rates rather than plugged in, so the uncertainty in that weight is itself
represented. And the
posterior is propagated rather than collapsed, which matters because the model's
output is a tail probability, and a tail probability computed at the posterior
mean is not the posterior mean of the tail probability. That gap is Jensen's
inequality, and ignoring it produces forecasts that are systematically
overconfident — a failure mode this project encountered directly.

---

## 2. Validation

### 2.1 Walk-forward, and what makes it leak-safe

The model is refitted at successive points in time and evaluated only on games
after each fit, so no evaluation ever uses a parameter estimated with knowledge
of the outcome being predicted.

The subtle part is that this is not sufficient. A frozen posterior evaluated on
later games is still contaminated if any *input* to the simulation embeds later
information — a park factor recomputed over the full season, a bullpen roster
reflecting a trade that had not happened, a player's season-long rate used to
predict a game inside that season. Leakage in a system with this many inputs is
not prevented by getting the training split right; it is prevented by auditing
every input for its as-of date. Several of the layers listed above leaked on
first implementation and had to be rebuilt with explicit as-of semantics.

### 2.2 Scoring rules

Forecasts are scored with the Brier score, the mean squared difference between
the forecast probability and the realized indicator. It is a **proper** scoring
rule: a forecaster minimizes expected loss by reporting their true belief. An
improper rule — accuracy, or a hit rate, or profit — can be improved by
misreporting, and so cannot be used to compare forecasters. Log loss is reported
alongside as a second proper rule with different tail sensitivity.

The Brier score decomposes as

  Brier = uncertainty − resolution + reliability

where **reliability** measures calibration (do events assigned 30% happen 30% of
the time) and **resolution** measures discrimination (does the forecaster
separate events that happen from those that do not). The decomposition is what
makes the score diagnostic rather than merely comparative: a model can be
perfectly calibrated and useless, by forecasting the base rate every time. The
central finding in [RESULTS.md](RESULTS.md) — comparable calibration, deficient
discrimination — is only visible through the decomposition.

Calibration is also inspected directly, through reliability curves and through
probability integral transform histograms for the continuous quantities. A
correctly specified predictive distribution produces a uniform PIT histogram;
the characteristic U-shape of an overconfident model, with too much mass in the
extreme bins, is the single most useful diagnostic picture in the project.

### 2.3 Clustered inference

The several hundred contracts priced on one game share a single realized
outcome. Their errors are massively correlated, and the effective sample size is
the number of *games*, not the number of contracts.

Every confidence interval reported is therefore bootstrapped by resampling
games, carrying all of a game's contracts together. Resampling contracts instead
narrows the intervals by a factor of between about 1.3 and 2.5 depending on the
segment. On this particular sample that does not flip any conclusion — the same
single family is significant either way — but the narrower intervals would
misstate the precision of every estimate, and there is no way to know in advance
that a sample will be forgiving.

### 2.4 The measurement basis

The most persistent error in this project lay not in any model but in the
choice of sample used to evaluate one: conditioning an evaluation on an event
correlated with the outcome.

The clearest instance: calibration measured on the subset of quoted contracts
where an order actually filled. An order fills when a counterparty takes the
other side, which happens preferentially when the price is wrong — so
conditioning on fills selects for the model's errors and reports worse
calibration than the model has. The measurement is not noisy, it is biased, and
more data does not fix it.

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

The defense adopted was a written pre-flight check on the measurement basis
before any result was believed: what defines the sample, what could have
selected it, and what would this measurement show if the model had no skill at
all. The last question is a placebo test and it killed more than one promising
result.

### 2.5 Pre-committing to the reading

For each model change, the criterion and the interpretation were fixed before
the test ran,
including the conditions under which a *favorable* result would be
disbelieved.

One instance: a test of loosening the minimum edge threshold returned point estimates favoring the change. It was rejected anyway,
because the fill simulator is known to flatter marginal orders and the result
fell on exactly the side that bias would push it. The pre-registered reading was
that a win in the direction of a known bias is not evidence. Without the prior
commitment, the natural move would have been to accept a result that agreed with
what the change was hoped to do.

---

## 3. Risk measurement

Position sizing maximized expected log wealth — the Kelly criterion, fractionally
scaled — over jointly simulated outcomes rather than treating positions as
independent. Joint simulation matters because contracts on the same game are
strongly dependent, and a sizing rule that assumes independence will
systematically take too much correlated exposure.

The optimization is constrained by a tail-risk limit expressed as a **Conditional
Value at Risk** bound, implemented in the Rockafellar–Uryasev auxiliary-variable
form that makes the constraint convex and therefore tractable alongside the
exposure caps. It is preferred over value at risk for two reasons: VaR is not
subadditive, so it can report that a diversified portfolio is riskier than its
parts, and it says nothing about how bad losses are once the quantile is
breached.

Maximizing expected log wealth is a utility-theoretic statement, not a trading
heuristic: log utility is the unique utility function under which the optimal
fraction is independent of wealth, which is why it appears in both the ruin
literature and the growth-optimal literature.

An empirical finding worth recording: across the sizing work, the binding
constraint was almost always a hard per-game exposure cap rather than the
objective function. On the full evaluation tape the per-game cap bound on 67 of
67 days. Any refinement of the objective tested under those caps is being
measured in their null space — the caps, not the optimizer, were doing the
sizing. Establishing which constraint actually binds before tuning what it
constrains is a lesson that generalizes well beyond this application.

---

## 4. Known limitations

- **One partial season.** Roughly two months of live operation and 760 games in
  the evaluation window. Nothing here establishes stability across seasons,
  rule changes, or roster turnover.
- **The benchmark holds information the model does not.** Market prices
  incorporate late lineup and weather news the model never saw. Part of the
  measured deficit is an information gap rather than a modeling deficiency, and
  this analysis does not separate the two.
- **Layer interactions are not fully explored.** Eleven modulating layers were
  each validated against a baseline, but the full interaction space was not
  searched, so some layers may be partly redundant with one another.
- **The public reimplementation is not the production model.** `sim/core.py`
  omits all eleven layers. Its outputs demonstrate the mechanism and should not
  be read as the production model's forecasts.
- **Single operator.** Every modeling choice and every interpretation was made
  by one person. Pre-commitment and placebo testing are partial substitutes for
  independent review, and only partial.
