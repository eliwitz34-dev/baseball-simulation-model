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
Expected calibration error over the full range is 0.0078, against the market's
0.0082.

**Distribution shape.** A forecaster that quotes the base rate every time is
perfectly calibrated and useless, so the sharper test is whether the whole
predicted distribution has the right shape. For each game, take the model's
predicted distribution, see what fraction of it sits below what actually
happened, and collect that fraction across games. If the distribution is right,
those values are uniform on [0, 1]. All seven families come out close to flat.
None shows the U shape you get when a predicted distribution is too narrow, and
the mean values run from 0.497 to 0.510 against an ideal of 0.500.

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

The game-total weakness concentrates at high lines rather than spreading evenly
across the family. That's worth knowing. A failure with an address is usually a
specific modeling defect, where one smeared uniformly across a family more often
means the family is just efficiently priced.

The most direct test is what happens when the two disagree. On the 1,692 contracts
where the model and the market took opposite sides, the model's side won
**48.8%** of the time. So its confidence is honest and its distributions have the
right shape. What it doesn't have is information the price didn't already
contain.

## A caution about the measurement basis

An earlier internal write-up reported the model as *worse calibrated* than the
market. That measurement used only the contracts where an order actually filled,
and it's wrong.

Filled contracts aren't a random sample of quoted ones. An order fills when a
counterparty takes the other side, which happens preferentially when the price is
wrong. A resting order is likeliest to get lifted exactly when it's mispriced. So
conditioning on fills selects for the model's own errors and will report worse
calibration than the model has, whether or not the model is any good. More data
doesn't help. The estimate isn't noisy, it's biased.

Measured on everything quoted, expected calibration error is 0.0078 against the
market's 0.0082. If anything the model is marginally the better calibrated of the
two, and calibration certainly isn't where it loses. What it lacks is
*discrimination*.

This was the most common failure mode in the whole project, and it's worth more
attention than any single result. A conclusion drawn from a badly chosen sample
is wrong without leaving any trace that it's wrong.

## Model changes that were tested and rejected

Each of these was built, measured against a criterion set before the test ran,
and then dropped. A list of changes that worked wouldn't tell you much, since it
was selected on having worked.

**Correcting for compression in the strikeout estimates.** Pitchers' estimated
strikeout rates looked shrunk too far toward the league average, so I fitted a
multiplier to push them back out and shipped it. An on/off comparison then showed
it made the probabilistic forecasts worse at every level, starter through game,
while improving the average predicted count. That combination is what an
over-correction looks like.

The compression turned out not to exist. How much shrinkage the estimates appear
to carry depends entirely on how you measure it. Take logits of noisy single-game
rates and the slope is 1.21. Use per-game rates and it's 0.89. Aggregate properly
by pitcher and it's 1.04, which is calibrated. I switched the multiplier off and
stopped the de-shrinkage work.

**Per-batter platoon splits inside the model fit.** Every batter handles
left-handed and right-handed pitching a bit differently. I moved that from a
post-hoc adjustment into the likelihood itself, with a per-player term sampled
alongside everything else. Over 531 walk-forward games it landed at parity or
slightly worse: correlation with realized starter strikeouts dropped from 0.496 to
0.489, and outcome separation was worse at every strikeout line. The per-player
signal is real and measured correctly. It's just too small, once properly shrunk,
to matter across a whole game.

Reviewing the build turned up a bug that mattered anyway. Pure switch hitters, 92
of 145 of them, were being silently dropped. The per-player artifact was assembled
only from players with a usable platoon split, and a switch hitter never bats
same-handed, so he never has one. That generalizes: build a per-player artifact
from the players who have enough data and you quietly exclude the exact group a
downstream feature is aimed at. Emit membership from the full classification, not
from the survivors.

**Weighting recent starts more heavily for pitchers who have changed.** The idea
was that a pitcher whose velocity or swing-and-miss rate has just moved should be
forecast mostly off his recent starts. I fitted a gate to do that and it lost to a
fixed middling weight. Not just overall, but in every third of every measure of
change, including pitchers whose stuff had demonstrably shifted. Leaning on recent
data over-weights noise even when the change is real.

The premise survived, though. Recent change does predict forward error, at about
+0.14 correlation for velocity. It just belongs in the model of what a pitcher's
stuff implies about his strikeout rate, not in how his history gets weighted, and
that's where the work went next.

**Position-player pitching.** Position players sometimes pitch in blowouts, and
the model treated them as ordinary relievers. Before building anything I worked
out the biggest effect a fix could possibly have: the share of plate appearances
involved times the largest plausible rate distortion, which came to 0.0121 runs
per game. The discrepancy I was chasing was 2.8 percentage points. The ceiling was
below the target, so I stopped there. Bounding a mechanism before building it was
the cheapest screening step I had.

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
  how to read a result was mine. Fixing each test's criterion beforehand is a
  partial substitute for someone else checking, and only partial.

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
