"""Midterm elections event study — compute pass.
Windows (trading days, election Tuesday = day 0):
  runup  = close(-1)/close(-6)-1      (5 sessions ending Mon pre-election)
  wed    = close(+1)/close(0)-1       (Wednesday reaction session)
  wedcum = close(+1)/close(-1)-1
  p5/p20/p60 = close(+k)/close(-1)-1
  drift20/60 = close(+k)/close(+1)-1  (post-reaction drift)
All returns adjusted-close total returns via yfinance.
"""
import warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

OUT = "/home/hatch/workspace/ibkr-comp/models/midterm-elections"

CYCLES = [
    # year, date, president, house_result, senate_result, govt, house_exp, senate_exp, exp_class, exp_note
    (1994, "1994-11-08", "D", "R take (+54)", "R take (+8)", "R Congress", "SURPRISE-R", "SURPRISE-R", "SURPRISE",
     "R gains expected on a flipped generic ballot, but eve consensus was R falls short of majorities in BOTH chambers; only 3 of 14 WaPo-polled experts predicted the House flip; +54 seats far exceeded expectations (EBSCO/History; Am Rev of Politics 37(1))."),
    (1998, "1998-11-03", "D", "R hold (D +5)", "R hold (no net change)", "R Congress", "SURPRISE-D", "EXPECTED", "SURPRISE",
     "Six-year itch + impeachment: R gains expected (Gingrich predicted pickups); Democrats GAINED 5 House seats, first president's-party House gain in a midterm since 1934; Gingrich resigned (Wikipedia; Abramowitz 2001)."),
    (2002, "2002-11-05", "R", "R hold (+8)", "R take (+2 net)", "Unified R", "EXPECTED", "SURPRISE-R", "SURPRISE",
     "Post-9/11 rally, Bush approval ~63%: House hold/gain expected and delivered; Senate control a genuine toss-up after the Jeffords switch — R retake a mild surprise (RCP final generic R+1.7)."),
    (2006, "2006-11-07", "R", "D take (+31)", "D take (+6)", "D Congress", "EXPECTED", "EXPECTED", "EXPECTED",
     "Generic ballot D+11 avg; D wave the consensus call all fall; models if anything UNDERestimated (+31 actual vs +15-20 projected) (Gallup; uspollingdata model history)."),
    (2010, "2010-11-02", "D", "R take (+63)", "D hold", "Split", "EXPECTED", "EXPECTED", "EXPECTED",
     "Generic ballot R+9; R House take the consensus (models R+40-50, actual +63); D Senate hold was the modal forecast (Sabato/538 leaned D hold)."),
    (2014, "2014-11-04", "D", "R hold (+13)", "R take (+9)", "R Congress", "EXPECTED", "EXPECTED", "EXPECTED",
     "Forecast models favored R Senate take (~2/3-3/4 probability); generic ballot R+3; best-calibrated modern midterm (uspollingdata)."),
    (2018, "2018-11-06", "R", "D take (+41)", "R hold (+2)", "Split", "EXPECTED", "EXPECTED", "EXPECTED",
     "Generic ballot D+7-8; D House take and R Senate hold/gain both the consensus forecast (brutal D Senate map, 26 D seats defended)."),
    (2022, "2022-11-08", "D", "R take (+9)", "D hold (+1)", "Split", "EXPECTED", "SURPRISE-D", "SURPRISE",
     "The red wave that failed: 538 final gave R 59% Senate / 84% House; prediction markets ~33% D Senate hold. D held Senate and GAINED a seat; R House gain +9 vs +20-30 projected (davemanuel polling-miss database; Split Ticket verdict)."),
]

INDEX = ["SPY", "QQQ", "IWM", "DIA"]
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
           "KBE", "XOP", "ITA", "IHF", "XHB", "TAN", "ICLN"]
MOVERS = ["XOM", "CVX", "SLB", "OXY", "HAL", "COP",
          "UNH", "HUM", "CI", "LLY", "JNJ", "PFE", "CVS",
          "LMT", "RTX", "NOC", "GD", "BA",
          "JPM", "BAC", "GS", "WFC", "C",
          "FSLR", "ENPH", "NEE"]
EXTRA = ["CL=F"]
ALL = sorted(set(INDEX + SECTORS + MOVERS + EXTRA))

print("downloading", len(ALL), "tickers...")
raw = yf.download(ALL, start="1993-09-01", end="2023-05-01", auto_adjust=True,
                  progress=False, threads=True)
px = raw["Close"].copy()
px.to_csv(f"{OUT}/prices.csv")
print("price matrix:", px.shape)

EDATES = {c[0]: pd.Timestamp(c[1]) for c in CYCLES}

def windows(series, edate):
    s = series.dropna()
    pos = s.index.get_indexer([edate], method="ffill")[0]
    if pos < 0 or s.index[pos] != edate:
        # election day must be an actual bar for the anchor asset; for others allow nearest prior
        if pos < 6 or pos + 61 >= len(s):
            return None
    if pos < 6 or pos + 61 >= len(s):
        return None
    c = s.iloc
    base = c[pos - 1]
    return {
        "runup": c[pos - 1] / c[pos - 6] - 1,
        "wed": c[pos + 1] / c[pos] - 1,
        "wedcum": c[pos + 1] / base - 1,
        "p5": c[pos + 5] / base - 1,
        "p20": c[pos + 20] / base - 1,
        "p60": c[pos + 60] / base - 1,
        "drift20": c[pos + 20] / c[pos + 1] - 1,
        "drift60": c[pos + 60] / c[pos + 1] - 1,
    }

# ---- cycles.csv
cyc_rows = []
for (yr, dt, pres, hr, sr, govt, hx, sx, xc, note) in CYCLES:
    cyc_rows.append(dict(cycle=yr, election_date=dt, president_party=pres, house_result=hr,
                         senate_result=sr, government=govt, house_expectation=hx,
                         senate_expectation=sx, expectation_class=xc, expectation_note=note))
cyc = pd.DataFrame(cyc_rows)
cyc.to_csv(f"{OUT}/cycles.csv", index=False)

# ---- window returns for every ticker x cycle
recs = []
for tk in ALL:
    if tk not in px.columns:
        continue
    for (yr, dt, *_rest) in CYCLES:
        w = windows(px[tk], EDATES[yr])
        if w is None:
            continue
        recs.append(dict(ticker=tk, cycle=yr, **{k: round(v * 100, 2) for k, v in w.items()}))
W = pd.DataFrame(recs)
meta = cyc.set_index("cycle")[["government", "expectation_class"]]
W = W.join(meta, on="cycle")
W.to_csv(f"{OUT}/window_returns_all.csv", index=False)
print("window rows:", len(W))

def stats(df, cols=("wed", "p5", "p20", "p60")):
    out = {}
    for c in cols:
        s = df[c].dropna()
        out[c] = dict(n=len(s), mean=round(s.mean(), 2), med=round(s.median(), 2),
                      hit=round((s > 0).mean() * 100) if len(s) else np.nan)
    return out

print("\n=== SPY by cycle ===")
spy = W[W.ticker == "SPY"].set_index("cycle")
print(spy[["runup", "wed", "wedcum", "p5", "p20", "p60", "drift60", "government", "expectation_class"]].to_string())

print("\n=== SPY grouped ===")
for label, grp in [("government", spy.groupby("government")), ("expectation", spy.groupby("expectation_class"))]:
    for k, g in grp:
        print(label, k, stats(g))

print("\nSPY all-cycle:", stats(spy))
print("SPY drift60 all:", stats(spy, ("drift60",)))

print("\n=== IWM vs SPY Wednesday (rel) ===")
iwm = W[W.ticker == "IWM"].set_index("cycle")
rel = (iwm["wed"] - spy["wed"]).dropna()
print(rel.round(2).to_string(), "| IWM lagged Wed in", int((rel < 0).sum()), "of", len(rel))

# eta-squared: outcome vs expectation grouping on SPY wed & p20
def eta2(s, groups):
    s = s.dropna(); groups = groups.loc[s.index]
    tot = ((s - s.mean()) ** 2).sum()
    bet = sum(len(g) * (g.mean() - s.mean()) ** 2 for _, g in s.groupby(groups))
    return round(bet / tot, 3) if tot > 0 else np.nan
print("\neta2 SPY wed: outcome", eta2(spy["wed"], spy["government"]), "| expectation", eta2(spy["wed"], spy["expectation_class"]))
print("eta2 SPY p20: outcome", eta2(spy["p20"], spy["government"]), "| expectation", eta2(spy["p20"], spy["expectation_class"]))
print("mean |wed| surprise cycles:", round(spy[spy.expectation_class == "SURPRISE"]["wed"].abs().mean(), 2),
      "| expected:", round(spy[spy.expectation_class == "EXPECTED"]["wed"].abs().mean(), 2))

# ---- sector windows file (sector x cycle long form with classes + SPY-relative)
sec = W[W.ticker.isin(SECTORS)].copy()
spy_w = spy[["wed", "p5", "p20"]].rename(columns={"wed": "spy_wed", "p5": "spy_p5", "p20": "spy_p20"})
sec = sec.join(spy_w, on="cycle")
for a, b in [("wed", "spy_wed"), ("p5", "spy_p5"), ("p20", "spy_p20")]:
    sec[a + "_rel"] = (sec[a] - sec[b]).round(2)
sec.to_csv(f"{OUT}/sector_windows.csv", index=False)

print("\n=== SECTOR Wednesday reaction by cycle (abs %) ===")
piv = sec.pivot_table(index="ticker", columns="cycle", values="wed")
print(piv.round(2).to_string())

print("\n=== SECTOR grouped means (Wed abs / Wed rel / p20 rel) ===")
for tk, g in sec.groupby("ticker"):
    line = [tk]
    for k, gg in [("ALL", g), ("R-Cong/UnifR", g[g.government.isin(["R Congress", "Unified R"])]),
                  ("D-Cong", g[g.government == "D Congress"]), ("Split", g[g.government == "Split"]),
                  ("EXPECTED", g[g.expectation_class == "EXPECTED"]), ("SURPRISE", g[g.expectation_class == "SURPRISE"])]:
        if len(gg):
            line.append(f"{k}: wed {gg.wed.mean():+.2f} (n{len(gg)}, hit {(gg.wed>0).mean()*100:.0f}%), wedrel {gg.wed_rel.mean():+.2f}, p20rel {gg.p20_rel.mean():+.2f}")
    print(" | ".join(line))

# ---- movers file
mov = W[W.ticker.isin(MOVERS)].copy()
mov = mov.join(spy_w, on="cycle")
mov["wed_rel"] = (mov["wed"] - mov["spy_wed"]).round(2)
mov["p5_rel"] = (mov["p5"] - mov["spy_p5"]).round(2)
mov.to_csv(f"{OUT}/movers.csv", index=False)
print("\n=== Biggest Wednesday movers per cycle (rel to SPY) ===")
for yr, g in mov.groupby("cycle"):
    g = g.dropna(subset=["wed_rel"])
    top = g.nlargest(3, "wed_rel"); bot = g.nsmallest(3, "wed_rel")
    f = lambda r: f"{r.ticker} {r.wed_rel:+.1f}"
    print(yr, g.government.iloc[0], "| up:", ", ".join(map(f, top.itertuples())), "| dn:", ", ".join(map(f, bot.itertuples())))

print("\n=== Narrative-trade audit (Wed rel / p5 rel, % ) ===")
NARR = {"XOM": "R-win energy", "CVX": "R-win energy", "LMT": "R-win defense", "RTX": "R-win defense",
        "FSLR": "D-win clean", "UNH": "managed care", "KBE": "banks", "JPM": "banks"}
for tk in ["XOM", "CVX", "LMT", "RTX", "FSLR", "UNH", "JPM"]:
    g = mov[mov.ticker == tk]
    for r in g.itertuples():
        print(f"{tk:5s} {r.cycle} {r.government:11s} {r.expectation_class:9s} runup {r.runup:+.2f} wed {r.wed:+.2f} wedrel {r.wed_rel:+.2f} p5rel {r.p5_rel:+.2f}")
    print()

# ---- energy vs oil decomposition
print("=== ENERGY vs OIL ===")
oil = W[W.ticker == "CL=F"].set_index("cycle")
xle = W[W.ticker == "XLE"].set_index("cycle")
both = pd.DataFrame({"xle_wed": xle["wed"], "oil_wed": oil["wed"], "xle_p5": xle["p5"], "oil_p5": oil["p5"]}).dropna()
print(both.round(2).to_string())
for a, b in [("xle_wed", "oil_wed"), ("xle_p5", "oil_p5")]:
    print(a, "vs", b, "corr:", round(both[a].corr(both[b]), 2))
# XLE weekly variance explained by oil, all weeks 2002-2022 (context benchmark)
wk = px[["XLE", "CL=F"]].dropna().resample("W-FRI").last().pct_change().dropna()
wk = wk[(wk.index >= "2002-01-01") & (wk.index <= "2022-12-31")]
r2 = wk["XLE"].corr(wk["CL=F"]) ** 2
print(f"all-weeks 2002-2022: weekly XLE~CL corr {wk['XLE'].corr(wk['CL=F']):.2f}, R2 {r2:.2f}, n={len(wk)}")
# trailing beta at each election for residual computation
for yr in both.index:
    ed = EDATES[yr]
    s = px[["XLE", "CL=F"]].dropna()
    pos = s.index.get_indexer([ed], method="ffill")[0]
    tr = s.iloc[max(0, pos - 260):pos - 5]
    trr = tr.pct_change().dropna()
    beta = trr["XLE"].cov(trr["CL=F"]) / trr["CL=F"].var()
    resid = both.loc[yr, "xle_p5"] - beta * both.loc[yr, "oil_p5"]
    print(f"{yr}: XLE/oil trailing beta {beta:.2f}; p5 XLE {both.loc[yr,'xle_p5']:+.2f}, oil {both.loc[yr,'oil_p5']:+.2f}, oil-explained {beta*both.loc[yr,'oil_p5']:+.2f}, resid {resid:+.2f}")

# ---- SPY normalized paths for chart
paths = {}
for yr, ed in EDATES.items():
    s = px["SPY"].dropna()
    pos = s.index.get_indexer([ed], method="ffill")[0]
    seg = s.iloc[pos - 10:pos + 66]
    paths[yr] = (seg / seg.iloc[10] * 100).values  # day0 (election) = 100
np.save(f"{OUT}/spy_paths.npy", np.array([paths[y] for y in sorted(paths)]))
json.dump(sorted(paths), open(f"{OUT}/spy_paths_years.json", "w"))
print("\npaths saved for", sorted(paths))
print("DONE")
