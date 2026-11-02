# Midterm Elections Model

The code and data behind **"Stocks Don't Care Who Wins. Sectors Do. Oil Cares About Neither."** (Research Desk, Kevin Sigey, November 2026) — an event study of how U.S. stocks and sectors actually trade around midterm elections, and a test of whether the vote has ever moved energy stocks.

Paper: https://kevinsigey.github.io/portfolio/stocks-dont-care-who-wins.html

## What the study does

For each of the eight midterm elections from 1994 to 2022, it measures adjusted-close total returns over fixed trading-day windows around election Tuesday (day 0):

- **runup** — the 5 sessions ending the Monday before the election
- **wed** — the Wednesday reaction session (close +1 vs close 0)
- **p5 / p20 / p60** — cumulative returns 5, 20, and 60 sessions out
- **drift20 / drift60** — the post-reaction drift after the Wednesday close

Assets: the index ETFs (SPY, QQQ, IWM, DIA), 17 sector and industry ETFs (the SPDRs plus KBE, XOP, ITA, IHF, XHB, TAN, ICLN), 25 large single names, and front-month WTI crude (CL=F). Each cycle is also classified by what actually happened versus what was expected (expected / surprise, and for which party), from the contemporary forecasting record.

## Headline findings

- **Who wins does not move the market; the election ending does.** The S&P 500 rose on the Wednesday after the vote in 7 of 8 midterms (mean +0.48%) and was up at 60 days in 7 of 8 (mean +4.55%), positive under every winner grouping.
- **Sectors move only through what a result actually changes.** The branches that survive the study's own evidence gates are narrow: banks repeat on expected outcomes; healthcare on a split government survives as a gated story; banks on a Republican sweep survives with a demerit. Several plausible branches were tested and failed, and one candidate (E-6) was not adopted. The sector grid is partial — sector vehicles only exist from 2002, so most sector cells rest on six cycles or fewer.
- **The election does not move oil.** Across the six SPDR-era cycles, XLE's Wednesday return correlates 0.78 with WTI's, and the five-session correlation is 0.95 counted from the Monday close (0.83 counted from the Tuesday close). Oil-beta residuals are mildly positive in all six cycles regardless of which party won — the vote explains almost none of energy's election-week variance once oil is in the regression. Energy trades the oil price, not the ballot.

## Sample limits — read these before citing the numbers

- **Eight elections.** n = 8 for index windows, n ≤ 6 for most sector windows. These are patterns in a small sample, stated as such in the paper; they are not laws, and the study's own falsification lines are in the notebook.
- Returns are yfinance adjusted closes pulled in October 2026; figures can differ trivially from other vendors' adjusted series.
- The expectation classifications are the author's, built from the cited contemporary sources listed per cycle in `compute.py`.

## How to read the files

| File | What it is |
|---|---|
| `compute.py` | The compute pass: downloads prices, builds every window, writes the CSVs below |
| `midterm_elections.ipynb` | The full study notebook, with outputs — the analysis, branch tests, and verdicts |
| `midterm_elections_stripped.ipynb` | The same notebook with outputs stripped |
| `build_notebook.py` | The script that assembles the notebook |
| `prices.csv` | The adjusted-close price matrix the study runs on (1993–2023) |
| `window_returns_all.csv` | Every asset × cycle × window return |
| `sector_windows.csv` | Sector windows, including relative-to-SPY columns |
| `cycles.csv` | The eight cycles: dates, results, expectation classifications |
| `movers.csv` | Single-name windows for the 25 large names |
| `spy_paths.npy`, `spy_paths_years.json` | The S&P 500 resolution-path series behind the paper's first chart |
| `energy/i_oil_correlation.py` | The oil correlation lab: correlations, betas, and capture ratios of energy names and ETFs against WTI across windows |
| `energy/oil_stats_all_windows.csv` | The lab's full output table |

## Credit

Study, code, and paper by **Kevin Sigey**. Published so readers can check the raw evidence behind the paper's claims.
