"""
NCP v3 신규 패턴 탐색 (자체 창안 원칙 준수)
모든 패턴은 데이터 특성에서 독자적으로 도출
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eda import load_data
from patterns import _body_ratio, _range, _upper_wick, _lower_wick, _rolling_percentile, FINAL_PATTERNS
from strategy_v3 import backtest_trailing, compute_metrics, monthly_report

TEST_START = "2021-01-01"


# ──────────────────────────────────────────────
# Pattern I — 압축 폭발 (Compression Burst)
# 아이디어: 변동폭이 수일간 좁아지다가 오늘 상방으로 급팽창
# 에너지 축적 후 방향 결정: 상방이면 강한 추세 시작 신호
# 조건:
#   - 직전 4일 daily range 모두 rolling P40 이하 (압축 구간)
#   - 오늘 range >= rolling P70 (팽창)
#   - 오늘 close > open (양봉, 상방 폭발)
#   - 오늘 close > 직전 4일 최고가 (새 고점 돌파)
# ──────────────────────────────────────────────
def pattern_compression_burst(
    df: pd.DataFrame,
    compress_pct: float = 40.0,
    expand_pct: float = 70.0,
    window: int = 60,
) -> pd.Series:
    rng = _range(df)
    rng_compress_p = _rolling_percentile(rng, window, compress_pct)
    rng_expand_p   = _rolling_percentile(rng, window, expand_pct)

    # 직전 4일 모두 압축 구간
    compressed = (
        (rng.shift(1) <= rng_compress_p.shift(1)) &
        (rng.shift(2) <= rng_compress_p.shift(2)) &
        (rng.shift(3) <= rng_compress_p.shift(3)) &
        (rng.shift(4) <= rng_compress_p.shift(4))
    )
    # 오늘 상방 폭발
    burst_up = (
        (rng >= rng_expand_p) &
        (df["close"] > df["open"]) &
        (df["close"] > df["high"].shift(1).rolling(4).max())
    )

    signals = pd.Series("none", index=df.index)
    signals[compressed & burst_up] = "long"
    return signals


# ──────────────────────────────────────────────
# Pattern J — 종가 지배력 연속 (Close Dominance Streak)
# 아이디어: 종가가 당일 범위 상단에 위치하는 날이 연속될수록
#           매수세가 지속 우위임을 나타냄
# Close Position = (close - low) / (high - low)
# 조건:
#   - 직전 3일 close_pos 모두 >= 0.70 (상단 30% 마감)
#   - 직전 3일 모두 양봉
#   - 오늘 close > 직전 3일 최고 close (신고점)
# ──────────────────────────────────────────────
def pattern_close_dominance(
    df: pd.DataFrame,
    pos_threshold: float = 0.70,
    window: int = 60,
) -> pd.Series:
    rng = _range(df).replace(0, np.nan)
    close_pos = (df["close"] - df["low"]) / rng
    ret = df["close"] - df["open"]

    streak = (
        (close_pos.shift(1) >= pos_threshold) &
        (close_pos.shift(2) >= pos_threshold) &
        (close_pos.shift(3) >= pos_threshold) &
        (ret.shift(1) > 0) &
        (ret.shift(2) > 0) &
        (ret.shift(3) > 0)
    )
    new_close_high = df["close"] > df["close"].shift(1).rolling(3).max()

    signals = pd.Series("none", index=df.index)
    signals[streak & new_close_high] = "long"
    return signals


# ──────────────────────────────────────────────
# Pattern K — 하락 꼬리 반전 가속 (Lower Tail Rebound)
# TailEcho와 반대 방향의 꼬리 신호
# 아이디어: 큰 아래꼬리(매도 세력 강한 테스트) 후 다음날 양봉 마감
#           → 매도 시도를 완전히 흡수한 매수 세력 우위
# 조건:
#   - 전일 아래꼬리 >= rolling P80
#   - 전일 몸통 > 0 (중립 아님)
#   - 오늘 close > 전일 close (전일 고점 미돌파, 완만한 회복)
#   - 오늘 close > open (양봉)
# ──────────────────────────────────────────────
def pattern_lower_tail_rebound(
    df: pd.DataFrame,
    pct_threshold: float = 80.0,
    window: int = 60,
) -> pd.Series:
    lower = _lower_wick(df)
    body  = (df["close"] - df["open"]).abs()

    lower_p    = _rolling_percentile(lower, window, pct_threshold).shift(1)
    prev_lower = lower.shift(1)
    prev_body  = body.shift(1)
    prev_close = df["close"].shift(1)

    long_cond = (
        (prev_lower >= lower_p) &
        (prev_body > 0) &
        (df["close"] > prev_close) &
        (df["close"] > df["open"])
    )

    signals = pd.Series("none", index=df.index)
    signals[long_cond] = "long"
    return signals


# ──────────────────────────────────────────────
# 패턴 비교 테스트
# ──────────────────────────────────────────────
def test_new_patterns():
    df = load_data()

    candidates = {
        "CompressionBurst":  pattern_compression_burst,
        "CloseDominance":    pattern_close_dominance,
        "LowerTailRebound":  pattern_lower_tail_rebound,
    }

    print("=" * 70)
    print("신규 패턴 후보 테스트 (trail=20%, hold=30, OOS 2021~2026)")
    print("=" * 70)

    results = {}
    for name, fn in candidates.items():
        sig = fn(df)
        n_signals = (sig == "long").sum()
        trades = backtest_trailing(df, sig, trail_pct=0.20, max_hold=30)
        oos = trades[trades["entry_date"] >= TEST_START].copy()
        monthly = monthly_report(trades, TEST_START)
        m = compute_metrics(oos, label=name)

        print(f"\n[{name}] 총 신호={n_signals}건")
        for k, v in m.items():
            print(f"  {k}: {v}")
        if len(monthly) > 0:
            print(f"  월평균: {monthly.mean():.2f}%  / 30%이상: {(monthly>=30).sum()}회/{len(monthly)}개월")
        results[name] = {"signals": n_signals, "oos": oos, "monthly": monthly, "metrics": m}

    # 기존 v3 vs 조합 비교
    print("\n" + "=" * 70)
    print("기존 v3 + 신규 패턴 조합 테스트")
    print("=" * 70)

    from patterns import pattern_tail_echo, pattern_bear_absorption, pattern_triple_rise

    def make_combo(extra_fns):
        def _combo(df):
            base_fns = [pattern_tail_echo, pattern_bear_absorption, pattern_triple_rise]
            sigs = [fn(df) for fn in base_fns + extra_fns]
            combo = sigs[0].copy()
            for s in sigs[1:]:
                combo[(combo == "none") & (s == "long")] = "long"
            return combo
        return _combo

    for name, fn in candidates.items():
        combo_fn = make_combo([fn])
        sig = combo_fn(df)
        n_signals = (sig == "long").sum()
        trades = backtest_trailing(df, sig, trail_pct=0.20, max_hold=30)
        oos = trades[trades["entry_date"] >= TEST_START].copy()
        monthly = monthly_report(trades, TEST_START)
        m = compute_metrics(oos, label=f"v3+{name}")

        print(f"\n[v3 + {name}] 총 신호={n_signals}건")
        for k, v in m.items():
            print(f"  {k}: {v}")
        if len(monthly) > 0:
            print(f"  월평균: {monthly.mean():.2f}%  / 30%이상: {(monthly>=30).sum()}회/{len(monthly)}개월")

    return results


if __name__ == "__main__":
    test_new_patterns()
