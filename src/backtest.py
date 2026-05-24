import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from eda import load_data
from patterns import pattern_tail_echo, pattern_dual_pressure

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_END = "2020-12-31"
TEST_START = "2021-01-01"


# ──────────────────────────────────────────────
# 백테스트 엔진
# ──────────────────────────────────────────────

def backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    hold_days: int = 3,
    fee: float = 0.001,        # 편도 0.1%
    direction: str = "long",   # "long" or "both"
) -> pd.DataFrame:
    """
    신호 발생 시 다음날 open 진입, hold_days 후 close 청산
    fee: 편도 거래 비용 (슬리피지 포함)
    """
    trades = []

    for i, (date, sig) in enumerate(signals.items()):
        if sig == "none":
            continue
        if direction == "long" and sig != "long":
            continue

        entry_i = i + 1
        exit_i = i + 1 + hold_days

        if entry_i >= len(df) or exit_i >= len(df):
            continue

        entry_price = df["open"].iloc[entry_i]
        exit_price = df["close"].iloc[exit_i]
        entry_date = df.index[entry_i]
        exit_date = df.index[exit_i]

        if sig == "long":
            ret = (exit_price / entry_price - 1) - 2 * fee
        else:
            ret = (entry_price / exit_price - 1) - 2 * fee

        trades.append({
            "signal_date": date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "signal": sig,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return": ret,
        })

    return pd.DataFrame(trades)


def compute_metrics(trades: pd.DataFrame, label: str = "") -> dict:
    if len(trades) == 0:
        return {"label": label, "trades": 0}

    rets = trades["return"]
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = dd.min()

    annual_factor = 365 / 3  # 평균 3일 보유
    sharpe = (rets.mean() / rets.std()) * np.sqrt(annual_factor) if rets.std() > 0 else 0

    wins = (rets > 0).sum()
    losses = (rets <= 0).sum()
    avg_win = rets[rets > 0].mean() if wins > 0 else 0
    avg_loss = rets[rets <= 0].mean() if losses > 0 else 0
    profit_factor = (wins * avg_win) / (-losses * avg_loss) if losses > 0 and avg_loss != 0 else np.inf

    return {
        "label":          label,
        "trades":         len(trades),
        "win_rate":       f"{wins/len(trades)*100:.1f}%",
        "avg_return":     f"{rets.mean()*100:.2f}%",
        "total_return":   f"{(cum.iloc[-1]-1)*100:.1f}%",
        "sharpe":         f"{sharpe:.2f}",
        "mdd":            f"{mdd*100:.1f}%",
        "profit_factor":  f"{profit_factor:.2f}",
    }


def equity_curve(trades: pd.DataFrame, label: str) -> pd.Series:
    if len(trades) == 0:
        return pd.Series()
    curve = (1 + trades.set_index("entry_date")["return"]).cumprod()
    curve.name = label
    return curve


def plot_equity(curves: list, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(14, 5))
    for curve in curves:
        if len(curve) > 0:
            ax.plot(curve.index, curve.values, label=curve.name, linewidth=1.5)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Equity (start=1)")
    ax.legend()
    fig.tight_layout()
    plt.savefig(RESULTS_DIR / fname, dpi=150)
    plt.close()
    print(f"  -> results/{fname} 저장")


def walk_forward(
    df: pd.DataFrame,
    signal_fn,
    hold_days: int = 3,
    train_years: int = 3,
    test_years: int = 1,
    direction: str = "long",
) -> pd.DataFrame:
    """롤링 Walk-forward 테스트"""
    all_trades = []
    start_year = df.index[0].year + train_years
    end_year = df.index[-1].year

    for year in range(start_year, end_year + 1):
        train_start = f"{year - train_years}-01-01"
        train_end = f"{year - 1}-12-31"
        test_s = f"{year}-01-01"
        test_e = f"{year}-12-31"

        df_test = df[test_s:test_e]
        if len(df_test) < 10:
            continue

        signals = signal_fn(df[train_start:test_e])
        signals_test = signals[test_s:test_e]

        trades = backtest(df[train_start:test_e], signals, hold_days, direction=direction)
        trades_test = trades[trades["entry_date"] >= test_s] if len(trades) > 0 else pd.DataFrame()
        if len(trades_test) > 0:
            trades_test = trades_test.copy()
            trades_test["wf_year"] = year
            all_trades.append(trades_test)

    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def run():
    print("=" * 60)
    print("Phase 4: 백테스트 시작")
    print("=" * 60)

    df = load_data()
    df_train = df[:TRAIN_END]
    df_test = df[TEST_START:]

    results = {}

    # ── TailEcho Long ──
    print("\n[1] TailEcho — 학습 구간")
    sig_te_train = pattern_tail_echo(df_train)
    trades_te_train = backtest(df_train, sig_te_train, direction="long")
    m = compute_metrics(trades_te_train, "TailEcho (Train)")
    results["TailEcho_train"] = m
    for k, v in m.items():
        print(f"  {k}: {v}")

    print("\n[1] TailEcho — 검증 구간")
    sig_te_all = pattern_tail_echo(df)
    sig_te_test = sig_te_all[TEST_START:]
    trades_te_test = backtest(df, sig_te_all, direction="long")
    trades_te_test = trades_te_test[trades_te_test["entry_date"] >= TEST_START] if len(trades_te_test) > 0 else pd.DataFrame()
    m = compute_metrics(trades_te_test, "TailEcho (Test)")
    results["TailEcho_test"] = m
    for k, v in m.items():
        print(f"  {k}: {v}")

    # ── DualPressure Long ──
    print("\n[2] DualPressure — 학습 구간")
    sig_dp_train = pattern_dual_pressure(df_train)
    trades_dp_train = backtest(df_train, sig_dp_train, direction="long")
    m = compute_metrics(trades_dp_train, "DualPressure (Train)")
    results["DualPressure_train"] = m
    for k, v in m.items():
        print(f"  {k}: {v}")

    print("\n[2] DualPressure — 검증 구간")
    sig_dp_all = pattern_dual_pressure(df)
    trades_dp_all = backtest(df, sig_dp_all, direction="long")
    trades_dp_test = trades_dp_all[trades_dp_all["entry_date"] >= TEST_START] if len(trades_dp_all) > 0 else pd.DataFrame()
    m = compute_metrics(trades_dp_test, "DualPressure (Test)")
    results["DualPressure_test"] = m
    for k, v in m.items():
        print(f"  {k}: {v}")

    # ── Walk-forward ──
    print("\n[3] Walk-forward 테스트 (TailEcho)")
    wf_te = walk_forward(df, pattern_tail_echo, direction="long")
    m_wf_te = compute_metrics(wf_te, "TailEcho WF")
    results["TailEcho_wf"] = m_wf_te
    for k, v in m_wf_te.items():
        print(f"  {k}: {v}")

    print("\n[3] Walk-forward 테스트 (DualPressure)")
    wf_dp = walk_forward(df, pattern_dual_pressure, direction="long")
    m_wf_dp = compute_metrics(wf_dp, "DualPressure WF")
    results["DualPressure_wf"] = m_wf_dp
    for k, v in m_wf_dp.items():
        print(f"  {k}: {v}")

    # ── 차트 ──
    print("\n[4] 차트 생성")
    te_train_curve = equity_curve(trades_te_train, "TailEcho Train")
    te_test_curve = equity_curve(trades_te_test, "TailEcho Test")
    dp_train_curve = equity_curve(trades_dp_train, "DualPressure Train")
    dp_test_curve = equity_curve(trades_dp_test, "DualPressure Test")
    wf_te_curve = equity_curve(wf_te, "TailEcho WF") if len(wf_te) > 0 else pd.Series()
    wf_dp_curve = equity_curve(wf_dp, "DualPressure WF") if len(wf_dp) > 0 else pd.Series()

    plot_equity([te_train_curve, te_test_curve], "TailEcho Equity", "bt_tail_echo.png")
    plot_equity([dp_train_curve, dp_test_curve], "DualPressure Equity", "bt_dual_pressure.png")
    plot_equity([wf_te_curve, wf_dp_curve], "Walk-forward Equity", "bt_walk_forward.png")

    print("\n[완료] 백테스트 종료")
    print("=" * 60)
    return results


if __name__ == "__main__":
    run()
