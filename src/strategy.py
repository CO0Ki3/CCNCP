"""
NCP 전략 v2: TailEcho + BearAbsorption + TripleRise 조합 (Long only)
청산: 트레일링 스탑 10%, max_hold=10일
목표: 월 평균 수익률 30% (지속 개발 중)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from eda import load_data
from patterns import pattern_tail_echo, pattern_bear_absorption, pattern_triple_rise

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TEST_START = "2021-01-01"
TRAIL_PCT  = 0.10
MAX_HOLD   = 10
FEE        = 0.001


def combined_signals(df: pd.DataFrame) -> pd.Series:
    s1 = pattern_tail_echo(df)
    s2 = pattern_bear_absorption(df)
    s3 = pattern_triple_rise(df)
    combo = s1.copy()
    for s in [s2, s3]:
        combo[(combo == "none") & (s == "long")] = "long"
    return combo


def backtest_trailing(
    df: pd.DataFrame,
    signals: pd.Series,
    trail_pct: float = TRAIL_PCT,
    max_hold: int = MAX_HOLD,
    fee: float = FEE,
) -> pd.DataFrame:
    trades = []
    for i, (date, sig) in enumerate(signals.items()):
        if sig != "long":
            continue
        entry_i = i + 1
        if entry_i >= len(df):
            continue

        entry_price = df["open"].iloc[entry_i]
        entry_date  = df.index[entry_i]
        peak = entry_price
        exit_price = exit_date = None

        for d in range(1, max_hold + 1):
            day_i = entry_i + d
            if day_i >= len(df):
                exit_price = df["close"].iloc[day_i - 1]
                exit_date  = df.index[day_i - 1]
                break
            peak  = max(peak, df["high"].iloc[day_i])
            stop  = peak * (1 - trail_pct)
            if df["low"].iloc[day_i] <= stop:
                exit_price = stop
                exit_date  = df.index[day_i]
                break
            if d == max_hold:
                exit_price = df["close"].iloc[day_i]
                exit_date  = df.index[day_i]

        if exit_price is None:
            continue

        ret = (exit_price / entry_price - 1) - 2 * fee
        trades.append({
            "signal_date": date,
            "entry_date":  entry_date,
            "exit_date":   exit_date,
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "return":      ret,
        })

    return pd.DataFrame(trades)


def compute_metrics(trades: pd.DataFrame, label: str = "") -> dict:
    if len(trades) == 0:
        return {"label": label, "trades": 0}
    rets = trades["return"]
    cum  = (1 + rets).cumprod()
    peak = cum.cummax()
    mdd  = ((cum - peak) / peak).min()
    annual_factor = 365 / MAX_HOLD
    sharpe = (rets.mean() / rets.std()) * np.sqrt(annual_factor) if rets.std() > 0 else 0
    wins   = (rets > 0).sum()
    losses = (rets <= 0).sum()
    avg_w  = rets[rets > 0].mean() if wins > 0 else 0
    avg_l  = rets[rets <= 0].mean() if losses > 0 else 0
    pf = (wins * avg_w) / (-losses * avg_l) if losses > 0 and avg_l != 0 else np.inf
    return {
        "label":         label,
        "trades":        len(trades),
        "win_rate":      f"{wins/len(trades)*100:.1f}%",
        "avg_return":    f"{rets.mean()*100:.2f}%",
        "total_return":  f"{(cum.iloc[-1]-1)*100:.1f}%",
        "sharpe":        f"{sharpe:.2f}",
        "mdd":           f"{mdd*100:.1f}%",
        "profit_factor": f"{pf:.2f}",
    }


def monthly_report(trades: pd.DataFrame, start: str) -> pd.Series:
    oos = trades[trades["entry_date"] >= start].copy()
    oos["month"] = oos["entry_date"].dt.to_period("M")
    return oos.groupby("month")["return"].apply(lambda x: (1 + x).prod() - 1) * 100


def plot_equity(trades: pd.DataFrame, label: str, fname: str):
    curve = (1 + trades.set_index("entry_date")["return"]).cumprod()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(curve.index, curve.values, label=label, linewidth=1.5)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(label)
    ax.set_ylabel("Equity (start=1)")
    ax.legend()
    fig.tight_layout()
    plt.savefig(RESULTS_DIR / fname, dpi=150)
    plt.close()
    print(f"  -> results/{fname}")


def run():
    print("=" * 60)
    print("NCP 전략 v2 (TailEcho + BearAbsorption + TripleRise)")
    print(f"트레일링 스탑 {TRAIL_PCT*100:.0f}%, max_hold={MAX_HOLD}일")
    print("=" * 60)

    df = load_data()
    signals = combined_signals(df)
    trades  = backtest_trailing(df, signals)
    oos     = trades[trades["entry_date"] >= TEST_START].copy()
    monthly = monthly_report(trades, TEST_START)

    print(f"\n전체 거래: {len(trades)}건 ({len(trades)/13:.1f}건/년)")
    print(f"\n[OOS 성과 2021~2026]")
    m = compute_metrics(oos)
    for k, v in m.items():
        print(f"  {k}: {v}")

    print(f"\n[월별 수익률]")
    print(f"  월 평균:    {monthly.mean():.2f}%")
    print(f"  월 최대:    {monthly.max():.2f}%")
    print(f"  월 최소:    {monthly.min():.2f}%")
    print(f"  수익 월 %:  {(monthly>0).mean()*100:.1f}%")
    print(f"  30% 이상:   {(monthly>=30).sum()}회 / {len(monthly)}개월")

    print("\n[패턴별 기여]")
    for name, fn in [("TailEcho", pattern_tail_echo),
                     ("BearAbsorption", pattern_bear_absorption),
                     ("TripleRise", pattern_triple_rise)]:
        n = (fn(df) == "long").sum()
        print(f"  {name}: {n}건/전체기간")

    print("\n[차트]")
    train_trades = trades[trades["entry_date"] < TEST_START].copy()
    plot_equity(train_trades, "v2 Train",     "v2_train.png")
    plot_equity(oos,          "v2 OOS",       "v2_oos.png")

    print("\n[완료]")
    return trades, oos, monthly


if __name__ == "__main__":
    run()
