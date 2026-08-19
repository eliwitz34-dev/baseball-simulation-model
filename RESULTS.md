# Results

## Headline result

The model quoted seven families of game-level contracts. Across 35,159 of them,
covering 764 games between 29 May and 30 July 2026, every forecast was compared
against the market's midpoint price for the same contract at the same moment.
Both get scored with the Brier score, which is the mean squared error of a
probability forecast. It's a proper scoring rule, so neither forecaster can
improve its score by reporting something other than what it believes.

| | Brier |
|---|---|
| model | 0.1905 |
| market | 0.1896 |
| base-rate forecast | 0.2490 |
| difference, model − market | −0.00095, against the model |
| 95% CI (game-clustered) | [−0.00270, +0.00068] |

Both are well clear of the base rate, so the model does carry real information.
But the point estimate favors the market and the interval contains zero.
**On this sample the two are indistinguishable, with the point estimate slightly
against the model.**

Intervals are bootstrapped over **games, not contracts**. All the contracts on one
game share a single outcome and move together, so the game is the real sampling
unit. Resampling contracts instead would narrow these intervals by somewhere
between 1.3 and 2.5 times depending on the family, and every estimate would look
more precise than it is.

## Does the model match real baseball?

A separate question from whether it beats the market, and the one to ask first.

**Calibration.** Grouping every forecast by the probability it assigned, observed
frequencies track predicted ones closely from about 0.25 upward. Below that the
model is mildly overconfident: events it assigns 16% occur about 18% of the time.
Binned calibration error is a poor way to put a number on this, because the answer
moves with the choice of bins. The decomposition below does it without binning.

**Distribution shape.** A forecaster that quotes the base rate every time is
perfectly calibrated and useless, so the sharper test is whether the whole
predicted distribution has the right shape. For each game the model's predicted
distribution is evaluated at what actually happened, giving the fraction of the
distribution lying below the realized outcome. If the distribution is right,
those fractions are uniform on [0, 1]. All seven families come out close to flat.
None shows the U shape produced by a predicted distribution that is too narrow,
and the mean values run from 0.497 to 0.510 against an ideal of 0.500.

## Results by contract family

ΔBrier in units of 10⁻³, positive meaning the model is better:

| family | n | Brier model | Brier market | ΔBrier ×10³ [95% CI] | |
|---|---:|---:|---:|---|---|
| moneyline | 1,510 | 0.2440 | 0.2452 | +1.21 [−2.11, +4.56] | |
| run line | 4,500 | 0.1895 | 0.1901 | +0.60 [−1.26, +2.55] | |
| first five, run line | 3,004 | 0.1832 | 0.1838 | +0.59 [−1.28, +2.42] | |
| first five, moneyline | 2,250 | 0.2046 | 0.2045 | −0.01 [−1.82, +1.85] | |
| team total | 10,500 | 0.1952 | 0.1944 | −0.73 [−2.84, +1.25] | |
| first five, total | 5,238 | 0.1778 | 0.1769 | −0.92 [−3.25, +1.53] | |
| game total | 8,157 | 0.1821 | 0.1788 | −3.33 [−6.47, −0.50] | **worse** |

One family is significantly worse, six are indistinguishable, none is better.

### Where the gap comes from

A Brier score decomposes into miscalibration, discrimination, and the irreducible
uncertainty of the events. Estimated by isotonic regression rather than by
binning, on all seven families, in units of 10⁻³:

| | model | market | difference [95% CI] |
|---|---:|---:|---|
| miscalibration | 0.22 | 0.24 | −0.01 [−0.16, +0.16] |
| discrimination | 59.04 | 59.91 | −0.87 [−2.51, +0.76] |
| uncertainty | 248.98 | 248.98 | — |

Brier = miscalibration − discrimination + uncertainty, so better discrimination
lowers the score.

The calibration difference is a hundredth of a point with a third of a point of
interval around it. The two forecasters are calibrated equally well, and that is a
tight finding rather than a noisy one. Nearly the whole 0.95 gap is discrimination,
though the interval on that component also contains zero, so its direction is clear
and its magnitude is not established.

Both are well calibrated in absolute terms: miscalibration costs the model 0.22
against an irreducible 249, under a tenth of a percent of its score.

The game-total weakness is discrimination rather than calibration. Its
discrimination component sits 2.67 below the market's, while its miscalibration
difference is +0.33 with an interval of [−0.16, +0.93] that contains zero.

The game-total weakness concentrates at high lines rather than spreading evenly
across the family, which is itself informative. A failure confined to an
identifiable region usually indicates a specific modeling defect, where one spread
uniformly across a family more often indicates that the family is efficiently
priced.

The most direct test is what happens when the two disagree. On the 1,692 contracts
where the model and the market took opposite sides, the model's side won
**48.8%** of the time. Its stated confidence is honest and its distributions have
the right shape. What it lacks is information the price did not already
contain.

## A caution about the measurement basis

An earlier internal write-up reported the model as *worse calibrated* than the
market. That measurement used only the contracts where an order actually filled,
and it is wrong.

Filled contracts are not a random sample of quoted ones. An order fills when a
counterparty takes the other side, which happens preferentially when the price is
wrong; a resting order is likeliest to be lifted precisely when it is mispriced.
Conditioning on fills therefore selects for the model's own errors and will
report worse calibration than the model has, whether or not the model is any
good. The resulting estimate is biased rather than merely imprecise, so
collecting more data does not correct it.

Measured on everything quoted, the miscalibration components are 0.22 and 0.24
×10⁻³, a difference of −0.01 with an interval of [−0.16, +0.16]. The two are
calibrated equally well, and neither loses anything worth measuring to
calibration. What the model lacks is *discrimination*.

This was the most common failure mode in the project, and it warrants more
attention than any single result, because a conclusion drawn from a badly chosen
sample is wrong without leaving any trace of being wrong.

## Model changes that were tested and rejected

Each of these was built, measured against a criterion set before the test ran,
and then dropped. A list of changes that worked would say little, having been
selected on the basis of having worked.

**Correcting for compression in the strikeout estimates.** Pitchers' estimated
strikeout rates looked shrunk too far toward the league average, so a multiplier
was fitted to push them back out, and it was shipped. An on/off comparison then
showed it made the probabilistic forecasts worse at every level, starter through
game, while improving the average predicted count. That combination is the
signature of an over-correction.

The compression turned out not to exist. How much shrinkage the estimates appear
to carry depends entirely on how it is measured. Logits of noisy single-game rates
give a slope of 1.21. Per-game rates give 0.89. Proper aggregation by pitcher gives
1.04, which is calibrated. The multiplier was switched off and the de-shrinkage
work stopped.

**Per-batter platoon splits inside the model fit.** Every batter handles
left-handed and right-handed pitching somewhat differently. That was moved from a
post-hoc adjustment into the likelihood itself, with a per-player term sampled
alongside everything else. Over 531 walk-forward games it came out at parity or
slightly worse: correlation with realized starter strikeouts fell from 0.496 to
0.489, and outcome separation was lower at every strikeout line. The per-player
signal is real and correctly measured. It is simply too small, once properly
shrunk, to matter across a whole game.

Reviewing the build surfaced a defect that mattered regardless. Pure switch
hitters, 92 of 145 of them, were being silently dropped. The per-player artifact
was assembled only from players with a usable platoon split, and a switch hitter
never bats same-handed, so he never has one. The lesson generalizes: a per-player
artifact built from the players who have enough data will quietly exclude the
exact group a downstream feature is aimed at. Membership should be emitted from
the full classification rather than from the survivors.

**Weighting recent starts more heavily for pitchers who have changed.** The
premise was that a pitcher whose velocity or swing-and-miss rate has recently
moved should be forecast mostly from his recent starts. A gate fitted to do that
lost to a fixed middling weight, not only overall but in every third of every
measure of change, including pitchers whose stuff had demonstrably shifted.
Leaning on recent data over-weights noise even when the change is real.

The premise itself survived. Recent change does predict forward error, at about
+0.14 correlation for velocity. It belongs in the model of what a pitcher's stuff
implies about his strikeout rate rather than in how his history is weighted, and
that is where the work went next.

**Position-player pitching.** Position players sometimes pitch in blowouts, and
the model treated them as ordinary relievers. Before anything was built, the
largest effect a fix could possibly have was worked out: the share of plate
appearances involved times the largest plausible rate distortion, which came to
0.0121 runs per game. The discrepancy under investigation was 2.8 percentage
points. The ceiling sat below the target, so the work stopped there. Bounding a
mechanism before building it proved the cheapest screening step available.

## Limitations

- **Two months is a short window.** The model was validated over a single
  partial season. Nothing here speaks to stability across seasons, rule changes,
  or roster turnover.
- **The benchmark is a moving target.** The market price incorporates
  information the model does not have, including late lineup and weather news.
  Some of the model's apparent deficit is an information gap rather than a
  modeling one, and this analysis does not separate the two.
- **Selection into the quoted set.** These seven are the families the model
  actually quoted, and they were picked partly on earlier performance. Results
  from a set chosen that way are optimistically biased, which cuts in an
  interesting direction here: it makes the absence of any family beating the
  market a stronger finding, not a weaker one.
- **One person, no independent review.** Every modeling choice and every call on
  how to read a result came from the same person. Fixing each test's criterion
  beforehand substitutes for an outside check only partially.

## Reproducing these numbers

The simulation results are reproducible from this repository:

```bash
python scripts/build_tables.py
python scripts/validate.py
```

The market comparison needs the record of quoted prices and outcomes, which isn't
in the repository. The analysis runs off a table of `(model probability, market
probability, realized outcome, game id)` and is short to rebuild against any such
table. The game-clustered bootstrap is the only part that needs care.
