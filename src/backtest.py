import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from eda import load_data
from patterns import pattern_tail_echo, pattern_bear_absorption

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_END = "2020-12-31"
TEST_START = "2021-01-01"


# ──────────────────────────────────────────────
# 백테스트 엔진 (손절 포함)
# ──────────────────────────────────────────────

def backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    hold_days: int = 3,
    fee: float = 0.001,
    direction: str = "long",
    stop_loss: float = None,   # None=손절없음, 예: -0.07 = -7%
    regime_filter: list = None,  # None=전체, 예: ["Bull","Volatile_Bull"]
) -> pd.DataFrame:
    """
    신호 발생 시 다음날 open 진입, hold_days 후 close 청산.
    stop_loss: 진입가 대비 손절 비율 (long 기준 음수)
    regime_filter: 허용할 국면 목록
    """
    # 국면 계산
    log_ret = np.log(df["close"] / df["close"].shift(1))
    ma50 = df["close"].rolling(50).mean()
    vol20 = log_ret.rolling(20).std()
    vol_med = vol20.median()
    conditions = [
        (df["close"] >= ma50) & (vol20 <= vol_med),
        (df["close"] >= ma50) & (vol20 > vol_med),
        (df["close"] < ma50) & (vol20 <= vol_med),
        (df["close"] < ma50) & (vol20 > vol_med),
    ]
    regime = pd.Series(
        np.select(conditions, ["Bull", "Volatile_Bull", "Bear", "Volatile_Bear"], "Unknown"),
        index=df.index
    )

    trades = []
    for i, (date, sig) in enumerate(signals.items()):
        if sig == "none":
            continue
        if direction == "long" and sig != "long":
            continue

        # 국면 필터
        if regime_filter and regime.iloc[i] not in regime_filter:
            continue

        entry_i = i + 1
        if entry_i >= len(df):
            continue

        entry_price = df["open"].iloc[entry_i]
        entry_date = df.index[entry_i]

        # 손절 포함 일별 가격 추적
        exit_price = None
        exit_date = None
        hit_stop = False

        for d in range(1, hold_days + 1):
            day_i = entry_i + d
            if day_i >= len(df):
                day_i = len(df) - 1
                exit_price = df["close"].iloc[day_i]
                exit_date = df.index[day_i]
                break

            day_low = df["low"].iloc[day_i]
            day_close = df["close"].iloc[day_i]

            if stop_loss is not None and sig == "long":
                stop_price = entry_price * (1 + stop_loss)
                if day_low <= stop_price:
                    exit_price = stop_price
                    exit_date = df.index[day_i]
                    hit_stop = True
                    break

            if d == hold_days:
                exit_price = day_close
                exit_date = df.index[day_i]

        if exit_price is None:
            continue

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
            "hit_stop": hit_stop,
            "regime": regime.iloc[i],
        })

    return pd.DataFrame(trades)


def compute_metrics(trades: pd.DataFrame, label: str = "") -> dict:
    if len(trades) == 0:
        return {"label": label, "trades": 0}

    rets = trades["return"]
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    mdd = ((cum - peak) / peak).min()

    annual_factor = 365 / 3
    sharpe = (rets.mean() / rets.std()) * np.sqrt(annual_factor) if rets.std() > 0 else 0

    wins = (rets > 0).sum()
    losses = (rets <= 0).sum()
    avg_win = rets[rets > 0].mean() if wins > 0 else 0
    avg_loss = rets[rets <= 0].mean() if losses > 0 else 0
    pf = (wins * avg_win) / (-losses * avg_loss) if losses > 0 and avg_loss != 0 else np.inf

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


def equity_curve(trades: pd.DataFrame, label: str) -> pd.Series:
    if len(trades) == 0:
        return pd.Series(name=label)
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
    print(f"  -> results/{fname}")


def walk_forward(
    df: pd.DataFrame,
    signal_fn,
    hold_days: int = 3,
    train_years: int = 3,
    stop_loss: float = None,
    direction: str = "long",
) -> pd.DataFrame:
    all_trades = []
    start_year = df.index[0].year + train_years
    end_year = df.index[-1].year

    for year in range(start_year, end_year + 1):
        train_start = f"{year - train_years}-01-01"
        test_s = f"{year}-01-01"
        test_e = f"{year}-12-31"

        df_window = df[train_start:test_e]
        if len(df_window) < 30:
            continue

        signals = signal_fn(df_window)
        trades = backtest(df_window, signals, hold_days, stop_loss=stop_loss, direction=direction)
        trades_test = trades[trades["entry_date"] >= test_s].copy() if len(trades) > 0 else pd.DataFrame()
        if len(trades_test) > 0:
            trades_test["wf_year"] = year
            all_trades.append(trades_test)

    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()


# ──────────────────────────────────────────────
# 민감도 분석
# ──────────────────────────────────────────────

def sensitivity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """TailEcho 파라미터 ±10% 변화 시 OOS 성과 변화"""
    from patterns import pattern_tail_echo
    base_pct = 80.0
    base_window = 60

    rows = []
    for pct_delta in [-10, -5, 0, +5, +10]:
        for win_delta in [-10, 0, +10]:
            pct = base_pct + pct_delta
            win = base_window + win_delta
            if pct <= 0 or pct >= 100 or win < 10:
                continue
            sig = pattern_tail_echo(df, pct_threshold=pct, window=win)
            trades = backtest(df, sig, direction="long", stop_loss=-0.07)
            test_trades = trades[trades["entry_date"] >= TEST_START].copy() if len(trades) > 0 else pd.DataFrame()
            m = compute_metrics(test_trades, f"pct={pct:.0f} win={win}")
            m["pct_threshold"] = pct
            m["window"] = win
            rows.append(m)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def run():
    print("=" * 60)
    print("Phase 5: 최종 검증")
    print("=" * 60)

    df = load_data()

    # ── 1. 손절 없는 베이스라인 재확인 ──
    print("\n[1] TailEcho 베이스라인 (손절 없음)")
    sig_all = pattern_tail_echo(df)
    t_train = backtest(df[:TRAIN_END], pattern_tail_echo(df[:TRAIN_END]), direction="long")
    t_test_all = backtest(df, sig_all, direction="long")
    t_test = t_test_all[t_test_all["entry_date"] >= TEST_START].copy() if len(t_test_all) > 0 else pd.DataFrame()
    for k, v in compute_metrics(t_train, "Train (no SL)").items():
        print(f"  {k}: {v}")
    for k, v in compute_metrics(t_test, "Test (no SL)").items():
        print(f"  {k}: {v}")

    # ── 2. 손절 -7% 추가 ──
    print("\n[2] TailEcho + 손절 -7%")
    t_train_sl = backtest(df[:TRAIN_END], pattern_tail_echo(df[:TRAIN_END]),
                          direction="long", stop_loss=-0.07)
    t_test_sl_all = backtest(df, sig_all, direction="long", stop_loss=-0.07)
    t_test_sl = t_test_sl_all[t_test_sl_all["entry_date"] >= TEST_START].copy() if len(t_test_sl_all) > 0 else pd.DataFrame()
    m_train_sl = compute_metrics(t_train_sl, "Train (SL -7%)")
    m_test_sl = compute_metrics(t_test_sl, "Test (SL -7%)")
    for k, v in m_train_sl.items():
        print(f"  {k}: {v}")
    for k, v in m_test_sl.items():
        print(f"  {k}: {v}")

    # ── 3. Walk-forward ──
    print("\n[3] Walk-forward (손절 -7%)")
    wf = walk_forward(df, pattern_tail_echo, stop_loss=-0.07, direction="long")
    m_wf = compute_metrics(wf, "WF (SL -7%)")
    for k, v in m_wf.items():
        print(f"  {k}: {v}")

    # ── 4. 민감도 분석 ──
    print("\n[4] 민감도 분석 (OOS 기준)")
    sa = sensitivity_analysis(df)
    print(sa[["pct_threshold", "window", "trades", "avg_return", "sharpe", "mdd", "profit_factor"]].to_string(index=False))

    # ── 5. 차트 ──
    print("\n[5] 차트 생성")
    c1 = equity_curve(t_train, "Train (no SL)")
    c2 = equity_curve(t_test, "Test (no SL)")
    c3 = equity_curve(t_train_sl, "Train (SL -7%)")
    c4 = equity_curve(t_test_sl, "Test (SL -7%)")
    c5 = equity_curve(wf, "Walk-forward (SL -7%)")

    plot_equity([c1, c2], "TailEcho — No Stop Loss", "p5_no_sl.png")
    plot_equity([c3, c4], "TailEcho — Stop Loss -7%", "p5_sl.png")
    plot_equity([c5], "TailEcho — Walk-forward", "p5_wf.png")

    print("\n[완료] Phase 5 검증 종료")
    print("=" * 60)
    return {"train_sl": m_train_sl, "test_sl": m_test_sl, "wf": m_wf, "sensitivity": sa}


if __name__ == "__main__":
    run()
