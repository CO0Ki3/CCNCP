"""
국면 필터 테스트: Bull/Volatile_Bull 구간에서만 진입 허용
MDD 개선 목적
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eda import load_data
from patterns import (
    _body_ratio, _range, _lower_wick, _rolling_percentile,
    pattern_tail_echo, pattern_bear_absorption, pattern_triple_rise,
)
from pattern_filter_test import pattern_lower_tail_break, pattern_close_dominance_loose
from strategy_v3 import compute_metrics, monthly_report

TEST_START = "2021-01-01"
TRAIL_PCT  = 0.20
MAX_HOLD   = 30
FEE        = 0.001


def get_regime(df: pd.DataFrame) -> pd.Series:
    log_ret = np.log(df["close"] / df["close"].shift(1))
    ma50    = df["close"].rolling(50).mean()
    vol20   = log_ret.rolling(20).std()
    vol_med = vol20.median()
    conditions = [
        (df["close"] >= ma50) & (vol20 <= vol_med),
        (df["close"] >= ma50) & (vol20 > vol_med),
        (df["close"] < ma50) & (vol20 <= vol_med),
        (df["close"] < ma50) & (vol20 > vol_med),
    ]
    return pd.Series(
        np.select(conditions, ["Bull", "Volatile_Bull", "Bear", "Volatile_Bear"], "Unknown"),
        index=df.index
    )


def backtest_trailing_regime(
    df: pd.DataFrame,
    signals: pd.Series,
    allowed_regimes: list = None,
    trail_pct: float = TRAIL_PCT,
    max_hold: int = MAX_HOLD,
    fee: float = FEE,
) -> pd.DataFrame:
    regime = get_regime(df)
    trades = []

    for i, (date, sig) in enumerate(signals.items()):
        if sig != "long":
            continue
        # 국면 필터
        if allowed_regimes and regime.iloc[i] not in allowed_regimes:
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
            "regime":      regime.iloc[i],
        })

    return pd.DataFrame(trades)


def combined_v4(df: pd.DataFrame) -> pd.Series:
    """5패턴 조합: TailEcho + BearAbsorption + TripleRise + LowerTailBreak + CloseDominanceLoose"""
    fns = [
        pattern_tail_echo,
        pattern_bear_absorption,
        pattern_triple_rise,
        pattern_lower_tail_break,
        pattern_close_dominance_loose,
    ]
    sigs = [fn(df) for fn in fns]
    combo = sigs[0].copy()
    for s in sigs[1:]:
        combo[(combo == "none") & (s == "long")] = "long"
    return combo


def run():
    df = load_data()
    signals = combined_v4(df)

    regime = get_regime(df)
    print("=" * 70)
    print("국면 필터 테스트 (5패턴 조합, trail=20%, hold=30)")
    print("=" * 70)

    # 국면 분포 확인
    oos_dates = df[df.index >= TEST_START].index
    regime_oos = regime[oos_dates]
    print("\n[OOS 기간 국면 분포]")
    print(regime_oos.value_counts())

    configs = {
        "필터 없음 (전체)":                   None,
        "Bull만":                             ["Bull"],
        "Volatile_Bull만":                    ["Volatile_Bull"],
        "Bull + Volatile_Bull (상승장만)":    ["Bull", "Volatile_Bull"],
    }

    print()
    for label, allowed in configs.items():
        trades = backtest_trailing_regime(df, signals, allowed_regimes=allowed)
        oos = trades[trades["entry_date"] >= TEST_START].copy()
        monthly = monthly_report(trades, TEST_START)
        m = compute_metrics(oos, label=label)

        print(f"\n[{label}]")
        n_oos = len(oos)
        for k, v in m.items():
            print(f"  {k}: {v}")
        if len(monthly) > 0:
            print(f"  월평균: {monthly.mean():.2f}%")
            print(f"  30%이상: {(monthly>=30).sum()}회 / {len(monthly)}개월")

    # 베스트 설정 연도별 breakdown
    print("\n" + "=" * 70)
    print("상승장 필터 적용 OOS 연도별 수익률")
    print("=" * 70)
    trades_bull = backtest_trailing_regime(df, signals, allowed_regimes=["Bull", "Volatile_Bull"])
    oos_bull = trades_bull[trades_bull["entry_date"] >= TEST_START].copy()
    if len(oos_bull) > 0:
        oos_bull["year"] = oos_bull["entry_date"].dt.year
        yearly = oos_bull.groupby("year")["return"].apply(lambda x: (1+x).prod()-1) * 100
        for yr, ret in yearly.items():
            print(f"  {yr}: {ret:+.1f}%")


if __name__ == "__main__":
    run()
