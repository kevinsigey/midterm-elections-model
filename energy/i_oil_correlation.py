# Oil correlation lab — Workstream I (paper-trading competition, research only)
# Question: which stocks actually track crude, and are XLE/XOM/CVX the right vehicles?
# Method: daily log returns on split/dividend-adjusted closes; single-index OLS of each
# name on the benchmark over trailing windows; capture = mean name return on benchmark
# days beyond +/-2%. Futures and stocks inner-joined on stock trading days.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

NAMES = ["XLE","XOM","CVX","EQT","FCG","PBF","OXY","COP","EOG","DVN","FANG",
         "SLB","HAL","MPC","VLO","XOP"]
SECTOR = {"XLE":"ETF (energy sector)","XOP":"ETF (E&P)","XOM":"Integrated","CVX":"Integrated",
          "OXY":"E&P / integrated-leaning","COP":"E&P","EOG":"E&P","DVN":"E&P","FANG":"E&P (shale)",
          "EQT":"Natgas E&P","FCG":"ETF (natgas E&P)","SLB":"Oilfield services","HAL":"Oilfield services",
          "MPC":"Refiner","VLO":"Refiner","PBF":"Refiner"}
BENCH = {"WTI":"CL=F", "Brent":"BZ=F", "NatGas":"NG=F"}
EXTRA = {"USO":"USO", "UNG":"UNG", "SPY":"SPY"}
WINDOWS = {"3mo": 63, "6mo": 126, "1y": 252, "2y": 504}

START, END = "2024-10-20", "2026-10-27"   # >2y of trading days, through Mon Oct 26 2026 close
all_tk = NAMES + list(BENCH.values()) + list(EXTRA.values())
px = yf.download(all_tk, start=START, end=END, auto_adjust=True, progress=False)["Close"]
px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
px = px.dropna(how="all")
ret = np.log(px / px.shift(1))

def window_stats(y, x, n):
    d = pd.concat([y, x], axis=1).dropna().tail(n)
    if len(d) < max(20, n // 2): return None
    yy, xx = d.iloc[:, 0], d.iloc[:, 1]
    corr = yy.corr(xx)
    beta = yy.cov(xx) / xx.var()
    resid = yy - (yy.mean() + beta * (xx - xx.mean()))
    up = xx > 0.02; dn = xx < -0.02
    return dict(n=len(d), corr=corr, beta=beta, r2=corr ** 2,
                vol=yy.std() * np.sqrt(252), resid_vol=resid.std() * np.sqrt(252),
                cap_up=yy[up].mean() * 100 if up.sum() >= 3 else np.nan, n_up=int(up.sum()),
                cap_dn=yy[dn].mean() * 100 if dn.sum() >= 3 else np.nan, n_dn=int(dn.sum()),
                bench_up=xx[up].mean() * 100 if up.sum() >= 3 else np.nan,
                bench_dn=xx[dn].mean() * 100 if dn.sum() >= 3 else np.nan)

rows = []
for tk in NAMES:
    for bname, bt in BENCH.items():
        for wname, n in WINDOWS.items():
            s = window_stats(ret[tk], ret[bt], n)
            if s: rows.append(dict(ticker=tk, bench=bname, window=wname, **s))
    # cross-check benchmarks
    for bname, bt in [("USO", "USO")]:
        s = window_stats(ret[tk], ret[bt], WINDOWS["6mo"])
        if s: rows.append(dict(ticker=tk, bench=bname, window="6mo", **s))
for tk in ["EQT", "FCG"]:
    s = window_stats(ret[tk], ret["UNG"], WINDOWS["6mo"])
    if s: rows.append(dict(ticker=tk, bench="UNG", window="6mo", **s))
stats = pd.DataFrame(rows)
stats.to_csv("oil_stats_all_windows.csv", index=False)

# benchmark cross-checks (do the futures and ETFs agree?)
cc = {}
for a, b in [("CL=F", "USO"), ("NG=F", "UNG"), ("CL=F", "BZ=F")]:
    d = pd.concat([ret[a], ret[b]], axis=1).dropna().tail(252)
    cc[f"{a}~{b}"] = round(float(d.iloc[:, 0].corr(d.iloc[:, 1])), 3)
print("Benchmark cross-checks (1y daily corr):", cc)

# ---------- main WTI table ----------
def show(wname, bench="WTI"):
    t = stats[(stats.window == wname) & (stats.bench == bench)].copy()
    t["sector"] = t.ticker.map(SECTOR)
    t = t.sort_values("corr", ascending=False)
    cols = ["ticker", "sector", "corr", "beta", "r2", "vol", "resid_vol", "cap_up", "cap_dn"]
    print(f"\n===== vs {bench}, {wname} (n={int(t.n.iloc[0])} days) =====")
    print(t[cols].round(3).to_string(index=False))
    return t

for w in ["3mo", "6mo", "1y", "2y"]:
    show(w)

t6 = stats[(stats.window == "6mo") & (stats.bench == "WTI")].set_index("ticker")
t1 = stats[(stats.window == "1y") & (stats.bench == "WTI")].set_index("ticker")

# capture efficiency (stock move per unit of oil move, big days only), 6mo
cap = stats[(stats.window == "6mo") & (stats.bench == "WTI")].copy()
cap["up_eff"] = cap.cap_up / cap.bench_up
cap["dn_eff"] = cap.cap_dn / cap.bench_dn   # both negative; ratio >0 = follows down
cap = cap.sort_values("up_eff", ascending=False)
print("\n===== Capture efficiency 6mo (avg stock % on WTI >+2% days / avg WTI % those days) =====")
print(cap[["ticker", "cap_up", "n_up", "up_eff", "cap_dn", "n_dn", "dn_eff"]].round(3).to_string(index=False))

# ---------- natgas ----------
g = stats[(stats.bench == "NatGas") & (stats.window == "6mo")].sort_values("corr", ascending=False)
print("\n===== vs NatGas (NG=F), 6mo — gas names only matter =====")
print(g[g.ticker.isin(["EQT", "FCG", "DVN", "COP", "XLE"])][["ticker", "corr", "beta", "r2", "cap_up", "cap_dn"]].round(3).to_string(index=False))
g2 = stats[(stats.bench == "UNG") & (stats.window == "6mo")]
print(g2[["ticker", "corr", "beta", "r2"]].round(3).to_string(index=False))

# ---------- Brent for top names + current picks ----------
top_names = list(t6.sort_values("r2", ascending=False).head(6).index)
for tk in ["XLE", "XOM", "CVX"]:
    if tk not in top_names: top_names.append(tk)
b = stats[(stats.bench == "Brent") & (stats.window.isin(["6mo", "1y"])) & (stats.ticker.isin(top_names))]
print("\n===== vs Brent, top names + current picks =====")
print(b[["ticker", "window", "corr", "beta", "r2"]].round(3).to_string(index=False))

# ---------- rolling correlation stability ----------
roll = {}
for tk in ["XLE", "XOM", "CVX"]:
    d = pd.concat([ret[tk], ret["CL=F"]], axis=1).dropna()
    roll[tk] = d.iloc[:, 0].rolling(60).corr(d.iloc[:, 1])
roll = pd.DataFrame(roll).dropna()
roll.to_csv("rolling_corr_wti.csv")
print("\n===== 60d rolling corr to WTI (2y) =====")
print(roll.describe().round(3).to_string())
oct26 = roll["2026-10-01":"2026-10-26"]
pre = roll["2026-07-01":"2026-09-30"]
print("Oct 2026 mean:", oct26.mean().round(3).to_dict(), "| Jul-Sep 2026 mean:", pre.mean().round(3).to_dict())
print("Latest (Oct 26):", roll.iloc[-1].round(3).to_dict())
print("Date of max corr (2y):", {t: str(roll[t].idxmax().date()) for t in roll.columns})

# ---------- purity ranking (6mo primary, 1y confirm) ----------
pur = pd.DataFrame({"beta6": t6.beta, "r2_6": t6.r2, "resid6": t6.resid_vol, "vol6": t6.vol,
                    "beta1": t1.beta, "r2_1": t1.r2, "resid1": t1.resid_vol})
pur["purity"] = pur.r2_6 * pur.beta6.abs()          # oil signal per unit of position
pur["noise_share"] = 1 - pur.r2_6
pur = pur.sort_values("purity", ascending=False)
print("\n===== PURITY RANK (6mo: purity = R2 x |beta|; residual vol = idiosyncratic noise) =====")
print(pur.round(3).to_string())

# ---------- book math ----------
CAPITAL = 1_000_000
book = {"XLE": 0.11, "XOM": 0.09, "CVX": 0.08}
def sleeve_beta(weights, beta_src):
    return sum(w * beta_src.loc[t] for t, w in weights.items())
for label, src in [("6mo betas", t6.beta), ("1y betas", t1.beta)]:
    pb = sleeve_beta(book, src)
    print(f"\nBook oil beta ({label}): sleeve-only contribution = {pb:.3f} "
          f"-> WTI +10% = ${CAPITAL * pb * 0.10:,.0f} | WTI -10% = ${-CAPITAL * pb * 0.10:,.0f}")
# swap scenario: XOM+CVX dollars ($170k) -> two purest single-stock trackers (ex-ETF), XLE kept
singles = [t for t in pur.index if t not in ("XLE", "XOP", "FCG")]
repl = list(pur.loc[singles].head(2).index)
swap = {"XLE": 0.11, repl[0]: 0.09, repl[1]: 0.08}
print("Swap candidates (purest single stocks):", repl)
for label, src in [("6mo betas", t6.beta), ("1y betas", t1.beta)]:
    pb = sleeve_beta(swap, src)
    print(f"Swapped sleeve oil beta ({label}) = {pb:.3f} -> WTI +10% = ${CAPITAL * pb * 0.10:,.0f} | -10% = ${-CAPITAL * pb * 0.10:,.0f}")
# residual risk of sleeve, both versions (independence approx across names given oil factor)
def sleeve_resid(weights, resid_src):
    return float(np.sqrt(sum((w * resid_src.loc[t]) ** 2 for t, w in weights.items())))
print(f"Sleeve idiosyncratic vol (ann., indep approx): current {sleeve_resid(book, t6.resid_vol):.3f} vs swapped {sleeve_resid(swap, t6.resid_vol):.3f}")
# FCG if triggered ($30k sleeve): its WTI beta adds almost nothing; its gas beta is the point
print("FCG 6mo: beta to WTI", round(float(t6.loc['FCG', 'beta']), 3), "| beta to NatGas",
      round(float(stats[(stats.ticker=='FCG')&(stats.bench=='NatGas')&(stats.window=='6mo')].beta.iloc[0]), 3))
# tech sleeve sanity: QQQ/XLK oil beta (small negative?)
for tk in ["SPY"]:
    s = window_stats(ret[tk], ret["CL=F"], 126)
    print(f"{tk} vs WTI 6mo: corr {s['corr']:.3f} beta {s['beta']:.3f}")
print("\nDONE — CSVs: oil_stats_all_windows.csv, rolling_corr_wti.csv")
