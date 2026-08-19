# Results

## Headline result

Across 163,777 scored contracts covering 786 games between 29 May and 30 July
2026, the model's forecasts were compared against the market's midpoint price
for the same contract at the same moment. Both were scored with the Brier score
— the mean squared error of a probability forecast, and a proper scoring rule,
so neither forecaster can improve its score by misreporting what it believes.

On game-level markets:

| | Brier |
|---|---|
| model | 0.1891 |
| market | 0.1883 |
| difference | −0.00088, against the model |
| 95% CI (game-clustered) | [−0.00249, +0.00078] |

The point estimate favors the market and the interval contains zero, so the
correct statement is that **the model and the market are indistinguishable on this
sample, with the point estimate slightly against the model.**

Both are far better than a base-rate forecast, so the model is genuinely
informative — it simply is not more informative than the price it was quoting
against. Since the model's purpose was to disagree with that price, this is the
result that matters most, and it is reported first for that reason.

The confidence interval is bootstrapped over **games, not contracts**. The
several hundred contracts on a single game share one realized outcome and are
strongly correlated, so games are the effective sampling unit. Resampling
contracts instead would narrow the intervals in this document by a factor of
between about 1.1 and 2.5, depending on the family. On this sample that does not
change which families come out significant — the same five do either way — but it
would overstate the precision of every estimate reported. Every interval here is
clustered the same way.

## Results by contract family

Aggregate agreement conceals a real split. Broken out by contract family, with
ΔBrier in units of 10⁻³ and positive meaning the model is better:

| family | n | Brier model | Brier market | ΔBrier ×10³ [95% CI] | |
|---|---:|---:|---:|---|---|
| moneyline | 1,510 | 0.2440 | 0.2452 | +1.21 [−2.25, +4.56] | |
| game total | 8,157 | 0.1821 | 0.1788 | −3.33 [−6.39, −0.48] | **worse** |
| run line | 4,500 | 0.1895 | 0.1901 | +0.60 [−1.40, +2.48] | |
| first 5 innings, moneyline | 2,250 | 0.2046 | 0.2045 | −0.01 [−1.85, +1.90] | |
| first 5 innings, total | 5,238 | 0.1778 | 0.1769 | −0.92 [−3.22, +1.33] | |
| first 5 innings, run line | 3,004 | 0.1832 | 0.1838 | +0.59 [−1.09, +2.44] | |
| strikeouts | 8,843 | 0.1584 | 0.1560 | −2.41 [−4.65, −0.26] | **worse** |
| home runs | 9,106 | 0.1056 | 0.1047 | −0.89 [−1.48, −0.31] | **worse** |
| hits | 24,725 | 0.1591 | 0.1584 | −0.70 [−1.28, −0.11] | **worse** |
| total bases | 34,964 | 0.1508 | 0.1508 | +0.03 [−0.58, +0.67] | |
| home runs recorded | 41,814 | 0.1849 | 0.1873 | **+2.42 [+1.29, +3.65]** | **better** |

Eleven of the sixteen priced families appear here; the other five carried too
few scored contracts in this window for a clustered interval to say anything, and
are omitted rather than reported at a precision the sample cannot support. Of the
eleven, four are significantly worse, one significantly better, and six
indistinguishable. The single genuine win is the largest family by volume and
survives the clustered interval comfortably. Its independent confirmations: a
discrimination advantage (area under the ROC curve 0.762 against the market's
0.756), a calibration-error advantage (0.012 against 0.025), and a direct
comparison in which the model's side wins **55.1% of the 2,962 contracts** where
model and market take opposite positions.

The same head-to-head test applied to game-level markets gives **49.2% of 1,933
disagreements**: when this model disagreed with the price on a game market, it
was wrong slightly more often than right.

The game-total weakness is consistent with a defect found independently in the
model's own diagnostics, where the error concentrates at high total lines rather
than being spread evenly. A failure concentrated in an identifiable region
indicates a specific modeling defect, whereas one spread uniformly across a
family more often indicates that the family is efficiently priced.

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
calibration error is 0.007 against the market's 0.007 on game markets, and 0.006
against 0.014 on props. By that measure calibration is comparable or better,
while *discrimination* is what the model lacks. The two analyses were measuring
different things, and the fills-conditioned measurement does not answer the
question it was taken to answer.

This kind of error — a conclusion that is an artifact of the slice it was
measured on — was the single most recurrent failure mode in this project. It is
worth more attention than any individual result, because it invalidates a
conclusion without leaving any sign that it has done so.

## Exclusion of profit figures

There are realized trading results. They are omitted deliberately, for three
reasons.

**They were wrong when last computed.** The accounting code treated an exchange
settlement record as a position when it is in fact a lifetime activity summary.
Any position closed before expiry was therefore reported with both legs, booking
completed round trips as total losses; one row was wrong by a factor of 145. The
defect was found and fixed after the figures in question were produced, and they
have not been recomputed on the corrected basis.

**The sample is too small to support a claim.** Two months and a few hundred
games is nowhere near enough to distinguish a real return from noise at the
observed effect size. Reporting a percentage return from it would imply a
precision the data cannot carry.

**It is the least informative number available.** Realized profit over a short
window is a convolution of forecast quality, position sizing, execution, and
luck. The Brier decomposition above isolates forecast quality directly, which is
the only component this repository is about.

The general point is worth stating plainly: a metric that was produced by code
with a known defect should be withdrawn rather than caveated, and re-derived
before it is quoted again.

## Model changes that were tested and rejected

The following were each built, measured against a criterion fixed in advance,
and abandoned. They are listed because a record of rejected changes is a better
indicator of process than a list of accepted ones — accepted changes are
selected on having worked, and are therefore contaminated by the selection.

**Position-player pitching.** Position players occasionally pitch in lopsided
games, and the model treated them as ordinary relievers. Before building the
correction, the plausible effect ceiling was computed as the share of plate
appearances affected multiplied by the maximum rate distortion: +0.0121 runs per
game, against a defect being chased of 2.8 percentage points. The ceiling was
below the target, so the work stopped there. Computing a mechanism's maximum
possible effect before implementing it turned out to be the highest-return habit
in the project.

**Adverse-selection adjustments.** Fill-conditional price shifts, estimated per
contract family. The per-family effects reversed sign when aggregated, a
straightforward Simpson's paradox driven by the mix of sides quoted, and the
interval on the pooled estimate spanned 21 percentage points. The conclusion was
not that the effect was unproven but that it was not identifiable at the sample
size available.

**A reinforcement-learning run-line policy.** A fast screening harness endorsed
it. The full walk-forward harness reversed the sign and showed a worse maximum
drawdown. The screen was wrong because it scored on a metric that reads
leverage rather than skill. The rule adopted afterwards: a screen may generate
hypotheses and may never decide.

**Loosening the minimum edge threshold from 1¢ to 0¢.** A full two-arm
walk-forward test. Point estimates favored the looser threshold, but the fill
simulator is known to flatter marginal orders, and the result landed on exactly
the side that known bias would push it. The reading agreed in advance was that a
win in the direction of a known bias is not evidence. The threshold was kept at
1¢.

The last case is the most instructive. The test favored the change and was
rejected regardless, because the direction of the result coincided with that of a
bias identified before the test was run. Fixing the interpretation in advance,
including the conditions under which a favorable result will be rejected, is the
only reliable protection against selecting the reading after the fact.

## Limitations

- **Two months is a short window.** The model was validated over a single
  partial season. Nothing here speaks to stability across seasons, rule changes,
  or roster turnover.
- **The benchmark is a moving target.** The market price incorporates
  information the model does not have, including late lineup and weather news.
  Some of the model's apparent deficit is an information gap rather than a
  modeling one, and this analysis does not separate the two.
- **Survivorship in the family set.** Sixteen families were priced and six were
  quoted. The six were chosen partly on the basis of earlier performance, so
  results restricted to those six would be optimistically biased. The table above
  therefore covers every priced family with enough scored contracts to test, not
  only the traded ones, which is why families that were never quoted appear in
  it.
- **Single operator, no independent review.** Every modeling choice, and every
  decision about how to read a result, was made by one person. The
  pre-commitment discipline described above is a partial substitute for
  independent review, and only a partial one.

## Reproducing these numbers

The transition table comparison in [README.md](README.md) is reproducible from
this repository:

```bash
python scripts/build_tables.py
python scripts/validate.py
```

The market comparison in this document is not. It requires the private tape of
quoted prices and realized outcomes, which is not redistributable. The analysis
code that produced it takes a table of `(model probability, market probability,
realized outcome, game id)` and is straightforward to reimplement against any
such table; the clustered bootstrap is the only part that needs care.
