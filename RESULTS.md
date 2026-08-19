# Results

## Headline result

The model quoted seven families of game-level contracts. Across 35,159 scored
contracts covering 764 games between 29 May and 30 July 2026, its forecasts were
compared against the market's midpoint price for the same contract at the same
moment. Both were scored with the Brier score — the mean squared error of a
probability forecast, and a proper scoring rule, so neither forecaster can improve
its score by misreporting what it believes.

| | Brier |
|---|---|
| model | 0.1905 |
| market | 0.1896 |
| base-rate forecast | 0.2490 |
| difference, model − market | −0.00095, against the model |
| 95% CI (game-clustered) | [−0.00270, +0.00068] |

Both forecasters are far better than the base rate, so the model is genuinely
informative. The point estimate favors the market and the interval contains zero,
so the correct statement is that **the model and the market are indistinguishable
on this sample, with the point estimate slightly against the model.**

Intervals are bootstrapped over **games, not contracts**. The contracts on a
single game share one realized outcome and are strongly correlated, so games are
the effective sampling unit. Resampling contracts instead would narrow the
intervals by a factor of between about 1.3 and 2.5, depending on the family,
which would overstate the precision of every estimate reported.

## Does the model match real baseball?

A separate question from whether it beats the market, and the one to ask first.

**Calibration.** Grouping every forecast by the probability it assigned, observed
frequencies track predicted ones closely from about 0.25 upward. Below that the
model is mildly overconfident: events it assigns 16% occur about 18% of the time.
Expected calibration error over the full range is 0.0078, against the market's
0.0082.

**Distribution shape.** Calibration alone can be achieved by a forecaster that
quotes the base rate every time, so the sharper test is whether the whole
predicted distribution has the right shape. For each game the model's predicted
distribution is evaluated at what actually happened, giving a probability integral
transform value that should be uniform on [0, 1] if the distribution is right.
Across all seven families the resulting histograms are close to flat, with no
family showing the U shape that indicates a predicted distribution that is too
narrow. Mean transform values range from 0.497 to 0.510 against an ideal of 0.500.

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

One family is significantly worse and six are indistinguishable. None is
significantly better.

The weakness in game totals concentrates at high lines rather than spreading
evenly across the family. A failure concentrated in an identifiable region
indicates a specific modeling defect, whereas one spread uniformly more often
indicates that the family is efficiently priced.

The most direct test is what happens when the two forecasters disagree. On the
1,692 contracts where the model and the market took opposite sides, the model's
side won **48.8%** of the time. Its calibration is sound and its predicted
distributions have the right shape; what it lacks is any information the price did
not already contain.

## A caution about the measurement basis

An earlier internal analysis reported the model as *worse calibrated* than the
market, measured on the subset of contracts where an order actually filled. This
document does not repeat that claim, because the two analyses disagree and the
present basis is the sounder one.

Contracts that fill are not a random sample of contracts quoted. An order fills
preferentially when the counterparty was willing to take the other side, which
correlates with the model being wrong — a resting order is most likely to be
lifted precisely when it is mispriced. Conditioning a calibration measurement on
fills therefore selects for the model's errors, and will report worse calibration
than the model actually has, whether or not the model is any good.

On the full quoted tape, which does not condition on fills, the model's expected
calibration error is 0.0078 against the market's 0.0082 — the model is if
anything marginally the better calibrated, and calibration is certainly not where
it loses. What it lacks is *discrimination*. The
two analyses were measuring different things, and the fills-conditioned
measurement does not answer the question it was taken to answer.

This kind of error — a conclusion that is an artifact of the slice it was
measured on — was the single most recurrent failure mode in this project. It is
worth more attention than any individual result, because it invalidates a
conclusion without leaving any sign that it has done so.

## Model changes that were tested and rejected

The following were each built, measured against a criterion fixed in advance,
and abandoned. They are listed because a record of rejected changes is a better
indicator of process than a list of accepted ones — accepted changes are
selected on having worked, and are therefore contaminated by the selection.

**Correcting for compression in the strikeout estimates.** Pitchers' estimated
strikeout rates appeared shrunk too far toward the league average, so a
multiplier was fitted to push them back out, and it was shipped. A direct on/off
comparison then showed it worsened the probabilistic forecasts at every level —
starter, team, batter and game — while improving the average predicted count,
which is the signature of an over-correction. The compression being corrected
turned out not to exist. How much shrinkage the estimates appear to carry depends
entirely on how it is measured: logits of noisy single-game rates give a slope of
1.21, per-game rates 0.89, and proper aggregation by pitcher 1.04, which is
calibrated. The multiplier was switched off and the de-shrinkage work stopped.

**Per-batter platoon splits inside the model fit.** Moved from a post-hoc
adjustment into the likelihood itself, with a per-player term sampled alongside
everything else. Graded over 531 walk-forward games it came out at parity or
slightly worse: correlation with realized starter strikeouts fell from 0.496 to
0.489, and the ability to separate outcomes was lower at every strikeout line.
The per-player signal is real and correctly measured; it is too small, once
properly shrunk, to matter at the level of a whole game.

Reviewing the rejected build surfaced two defects that mattered anyway. Pure
switch hitters — 92 of 145 — were being silently dropped, because the per-player
artifact was assembled only from players who had a usable split, and a switch
hitter never bats same-handed. The lesson generalizes: when a per-player artifact
is built from the players with enough data, it silently excludes the exact
population a downstream feature is aimed at, so membership should be emitted from
the full classification rather than from the survivors.

**Weighting recent performance more heavily for pitchers who have changed.** The
premise was that a pitcher whose velocity or swing-and-miss rate has recently
moved should be forecast with more weight on his recent starts. A gate fitted to
do that lost to a fixed middling weight — not only overall but in every third of
every measure of change, including for pitchers whose stuff had demonstrably
shifted. The premise itself survived: recent change does predict forward error,
with a correlation of about +0.14 for velocity change. It belongs in the model of
what a pitcher's stuff implies about his strikeout rate rather than in how his
history is weighted, and that is where the work went.

**Position-player pitching.** Position players occasionally pitch in lopsided
games, and the model treated them as ordinary relievers. Before building the
correction, the largest attainable effect was computed as the share of plate
appearances affected multiplied by the maximum rate distortion: +0.0121 runs per
game, against a discrepancy of 2.8 percentage points under investigation. The
ceiling was below the target, so the work stopped there. Bounding a mechanism's
effect before implementing it was the cheapest screening step available.

## Limitations

- **Two months is a short window.** The model was validated over a single
  partial season. Nothing here speaks to stability across seasons, rule changes,
  or roster turnover.
- **The benchmark is a moving target.** The market price incorporates
  information the model does not have, including late lineup and weather news.
  Some of the model's apparent deficit is an information gap rather than a
  modeling one, and this analysis does not separate the two.
- **Selection into the quoted set.** The seven families reported here were the
  ones the model quoted, and they were chosen partly on the basis of earlier
  performance. Results restricted to a set chosen that way are optimistically
  biased, which makes the absence of any family beating the market a stronger
  finding than it would otherwise be, not a weaker one.
- **Single operator, no independent review.** Every modeling choice, and every
  decision about how to read a result, was made by one person. The
  pre-commitment discipline described above is a partial substitute for
  independent review, and only a partial one.

## Reproducing these numbers

The simulation results are reproducible from this repository:

```bash
python scripts/build_tables.py
python scripts/validate.py
```

The market comparison needs the record of quoted prices and realized outcomes,
which is not in the repository. The analysis takes a table of `(model
probability, market probability, realized outcome, game id)` and is short to
reimplement against any such table; the game-clustered bootstrap is the only part
that needs care.
