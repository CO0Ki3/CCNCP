"""
LowerTailRebound 필터 강화 + CloseDominance 조건 완화 테스트
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eda import load_data
from patterns import (
    _body_ratio, _range, _upper_wick, _lower_wick, _rolling_percentile,
    pattern_tail_echo, pattern_bear_absorption, pattern_triple_rise,
)
from strategy_v3 import backtest_trailing, compute_metrics, monthly_report

TEST_START = "2021-01-01"


def pattern_lower_tail_break(
    df: pd.DataFrame,
    pct_threshold: float = 80.0,
    window: int = 60,
) -> pd.Series:
    """
    LowerTailRebound 강화판: 전일 고점 돌파까지 요구
    - 전일 아래꼬리 >= P80
    - 오늘 close > 전일 high (전고점 돌파)
    - 오늘 close > open (양봉)
    """
    lower  = _lower_wick(df)
    body   = (df["close"] - df["open"]).abs()
    lower_p    = _rolling_percentile(lower, window, pct_threshold).shift(1)
    prev_lower = lower.shift(1)
    prev_body  = body.shift(1)
    prev_high  = df["high"].shift(1)

    long_cond = (
        (prev_lower >= lower_p) &
        (prev_body > 0) &
        (df["close"] > prev_high) &
        (df["close"] > df["open"])
    )
    signals = pd.Series("none", index=df.index)
    signals[long_cond] = "long"
    return signals


def pattern_lower_tail_strong(
    df: pd.DataFrame,
    pct_threshold: float = 80.0,
    body_pct: float = 50.0,
    window: int = 60,
) -> pd.Series:
    """
    LowerTailRebound 강화판 B: 강한 양봉 조건 추가
    - 전일 아래꼬리 >= P80
    - 오늘 body_ratio >= P50 (보통 이상 강도 양봉)
    - 오늘 close > 전일 close
    - 오늘 close > open
    """
    lower  = _lower_wick(df)
    br     = _body_ratio(df)
    lower_p = _rolling_percentile(lower, window, pct_threshold).shift(1)
    br_p    = _rolling_percentile(br, window, body_pct)

    long_cond = (
        (lower.shift(1) >= lower_p) &
        (br >= br_p) &
        (df["close"] > df["close"].shift(1)) &
        (df["close"] > df["open"])
    )
    signals = pd.Series("none", index=df.index)
    signals[long_cond] = "long"
    return signals


def pattern_close_dominance_loose(
    df: pd.DataFrame,
    pos_threshold: float = 0.65,
    window: int = 60,
) -> pd.Series:
    """
    CloseDominance 완화판: threshold 0.70 → 0.65
    - 직전 2일 close_pos >= 0.65 (2일 연속으로 완화)
    - 직전 2일 양봉
    - 오늘 close > 직전 2일 최고 close
    """
    rng = _range(df).replace(0, np.nan)
    close_pos = (df["close"] - df["low"]) / rng
    ret = df["close"] - df["open"]

    streak = (
        (close_pos.shift(1) >= pos_threshold) &
        (close_pos.shift(2) >= pos_threshold) &
        (ret.shift(1) > 0) &
        (ret.shift(2) > 0)
    )
    new_close_high = df["close"] > df["close"].shift(1).rolling(2).max()

    signals = pd.Series("none", index=df.index)
    signals[streak & new_close_high] = "long"
    return signals


def run():
    df = load_data()
    fns = {
        "LowerTailBreak (close>prev_high)": pattern_lower_tail_break,
        "LowerTailStrong (body>=P50)":      pattern_lower_tail_strong,
        "CloseDominance Loose (2일/0.65)":  pattern_close_dominance_loose,
    }

    print("=" * 70)
    print("패턴 필터 강화/완화 테스트 (trail=20%, hold=30, OOS)")
    print("=" * 70)

    for name, fn in fns.items():
        sig = fn(df)
        n = (sig == "long").sum()
        trades = backtest_trailing(df, sig, trail_pct=0.20, max_hold=30)
        oos = trades[trades["entry_date"] >= TEST_START].copy()
        monthly = monthly_report(trades, TEST_START)
        m = compute_metrics(oos, label=name)
        print(f"\n[{name}] 전체신호={n}")
        for k, v in m.items():
            print(f"  {k}: {v}")
        if len(monthly) > 0:
            print(f"  월평균: {monthly.mean():.2f}%  / 30%이상: {(monthly>=30).sum()}회/{len(monthly)}개월")

    # v3 기존과 최상 후보 조합
    print("\n" + "=" * 70)
    print("v3 + 필터 강화 패턴 조합")
    print("=" * 70)

    base_fns = [pattern_tail_echo, pattern_bear_absorption, pattern_triple_rise]

    for name, fn in fns.items():
        def _combo(df, extra=fn):
            all_sigs = [f(df) for f in base_fns + [extra]]
            combo = all_sigs[0].copy()
            for s in all_sigs[1:]:
                combo[(combo == "none") & (s == "long")] = "long"
            return combo

        sig = _combo(df)
        n = (sig == "long").sum()
        trades = backtest_trailing(df, sig, trail_pct=0.20, max_hold=30)
        oos = trades[trades["entry_date"] >= TEST_START].copy()
        monthly = monthly_report(trades, TEST_START)
        m = compute_metrics(oos, label=f"v3+{name[:20]}")
        print(f"\n[v3 + {name}] 총신호={n}")
        for k, v in m.items():
            print(f"  {k}: {v}")
        if len(monthly) > 0:
            print(f"  월평균: {monthly.mean():.2f}%  / 30%이상: {(monthly>=30).sum()}회/{len(monthly)}개월")


if __name__ == "__main__":
    run()


# ─────────────────────────────────────────────
# 추가 테스트: v3 + LowerTailBreak + CloseDominance 3중 조합
# ─────────────────────────────────────────────
def test_triple_combo():
    df = load_data()
    base_fns = [pattern_tail_echo, pattern_bear_absorption, pattern_triple_rise]

    def _combo(df):
        extras = [pattern_lower_tail_break, pattern_close_dominance_loose]
        all_sigs = [fn(df) for fn in base_fns + extras]
        combo = all_sigs[0].copy()
        for s in all_sigs[1:]:
            combo[(combo == "none") & (s == "long")] = "long"
        return combo

    sig = _combo(df)
    n = (sig == "long").sum()
    trades = backtest_trailing(df, sig, trail_pct=0.20, max_hold=30)
    oos = trades[trades["entry_date"] >= TEST_START].copy()
    monthly = monthly_report(trades, TEST_START)
    m = compute_metrics(oos)

    print("\n[v3 + LowerTailBreak + CloseDominanceLoose 5중 조합]")
    print(f"  총신호={n}")
    for k, v in m.items():
        print(f"  {k}: {v}")
    if len(monthly) > 0:
        print(f"  월평균: {monthly.mean():.2f}%  / 30%이상: {(monthly>=30).sum()}회/{len(monthly)}개월")

