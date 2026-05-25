"""
NCP 전략 v3: TailEcho + BearAbsorption + TripleRise 조합 (Long only)
청산: 트레일링 스탑 20%, max_hold=30일
포지션 사이징: Kelly Criterion (반Kelly)
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
TRAIL_PCT  = 0.20   # v3: 20% (v2: 10%)
MAX_HOLD   = 30     # v3: 30일 (v2: 10일)
FEE        = 0.001
KELLY_FRACTION = 0.5  # 반Kelly (보수적)


def combined_signals(df: pd.DataFrame) -> pd.Series:
    s1 = pattern_tail_echo(df)
    s2 = pattern_bear_absorption(df)
    s3 = pattern_triple_rise(df)
    combo = s1.copy()
    for s in [s2, s3]:
        combo[(combo == "none") & (s == "long")] = "long"
    return combo


def kelly_size(win_rate: float, avg_win: float, avg_loss: float, fraction: float = KELLY_FRACTION) -> float:
    """Kelly Criterion으로 포지션 비율 계산 (반Kelly 적용)"""
    if avg_loss == 0 or win_rate <= 0:
        return 0.0
    b = avg_win / abs(avg_loss)
    q = 1 - win_rate
    k = (b * win_rate - q) / b
    return max(0.0, min(k * fraction, 1.0))  # 0~1 클램프


def backtest_trailing(
    df: pd.DataFrame,
    signals: pd.Series,
    trail_pct: float = TRAIL_PCT,
    max_hold: int = MAX_HOLD,
    fee: float = FEE,
    use_kelly: bool = False,
) -> pd.DataFrame:
    trades = []
    # Kelly를 위한 롤링 win/loss 통계 (최근 20건)
    recent_rets = []

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

        # Kelly 포지션 비율 계산
        position_size = 1.0
        if use_kelly and len(recent_rets) >= 10:
            wins = [r for r in recent_rets if r > 0]
            losses = [r for r in recent_rets if r <= 0]
            if wins and losses:
                wr = len(wins) / len(recent_rets)
                aw = np.mean(wins)
                al = np.mean(losses)
                position_size = kelly_size(wr, aw, al)

        recent_rets.append(ret)
        if len(recent_rets) > 20:
            recent_rets.pop(0)

        trades.append({
            "signal_date":   date,
            "entry_date":    entry_date,
            "exit_date":     exit_date,
            "entry_price":   entry_price,
            "exit_price":    exit_price,
            "return":        ret,
            "position_size": position_size,
            "adj_return":    ret * position_size,
        })

    return pd.DataFrame(trades)


def compute_metrics(trades: pd.DataFrame, label: str = "", use_adj: bool = False) -> dict:
    if len(trades) == 0:
        return {"label": label, "trades": 0}
    col = "adj_return" if (use_adj and "adj_return" in trades.columns) else "return"
    rets = trades[col]
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


def monthly_report(trades: pd.DataFrame, start: str, use_adj: bool = False) -> pd.Series:
    oos = trades[trades["entry_date"] >= start].copy()
    oos["month"] = oos["entry_date"].dt.to_period("M")
    col = "adj_return" if (use_adj and "adj_return" in oos.columns) else "return"
    return oos.groupby("month")[col].apply(lambda x: (1 + x).prod() - 1) * 100


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
    print("NCP 전략 v3 (TailEcho + BearAbsorption + TripleRise)")
    print(f"트레일링 스탑 {TRAIL_PCT*100:.0f}%, max_hold={MAX_HOLD}일, 반Kelly")
    print("=" * 60)

    df = load_data()
    signals = combined_signals(df)

    # 기본 (포지션 사이징 없음)
    trades_base = backtest_trailing(df, signals, use_kelly=False)
    oos_base    = trades_base[trades_base["entry_date"] >= TEST_START].copy()
    monthly_base = monthly_report(trades_base, TEST_START)

    # Kelly 포지션 사이징
    trades_kelly = backtest_trailing(df, signals, use_kelly=True)
    oos_kelly    = trades_kelly[trades_kelly["entry_date"] >= TEST_START].copy()
    monthly_kelly = monthly_report(trades_kelly, TEST_START, use_adj=True)

    print(f"\n전체 거래: {len(trades_base)}건")

    print(f"\n[OOS 기본 성과 — trail={TRAIL_PCT*100:.0f}%, hold={MAX_HOLD}일]")
    m = compute_metrics(oos_base)
    for k, v in m.items():
        print(f"  {k}: {v}")

    print(f"\n[OOS 월별 수익률 — 기본]")
    print(f"  월 평균:    {monthly_base.mean():.2f}%")
    print(f"  월 최대:    {monthly_base.max():.2f}%")
    print(f"  월 최소:    {monthly_base.min():.2f}%")
    print(f"  수익 월 %:  {(monthly_base>0).mean()*100:.1f}%")
    print(f"  30% 이상:   {(monthly_base>=30).sum()}회 / {len(monthly_base)}개월")

    print(f"\n[OOS Kelly 포지션 사이징 적용]")
    m_k = compute_metrics(oos_kelly, use_adj=True)
    for k, v in m_k.items():
        print(f"  {k}: {v}")
    print(f"  월평균(Kelly): {monthly_kelly.mean():.2f}%")
    print(f"  30% 이상:      {(monthly_kelly>=30).sum()}회 / {len(monthly_kelly)}개월")

    print("\n[패턴별 기여]")
    for name, fn in [("TailEcho", pattern_tail_echo),
                     ("BearAbsorption", pattern_bear_absorption),
                     ("TripleRise", pattern_triple_rise)]:
        n = (fn(df) == "long").sum()
        print(f"  {name}: {n}건/전체기간")

    print("\n[차트]")
    train_trades = trades_base[trades_base["entry_date"] < TEST_START].copy()
    plot_equity(train_trades, "v3 Train (trail=20%, hold=30)",  "v3_train.png")
    plot_equity(oos_base,     "v3 OOS (trail=20%, hold=30)",    "v3_oos.png")

    print("\n[완료]")
    return trades_base, oos_base, monthly_base, trades_kelly, oos_kelly, monthly_kelly


if __name__ == "__main__":
    run()
