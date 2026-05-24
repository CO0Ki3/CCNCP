import pandas as pd
import numpy as np
from typing import Literal

SignalType = Literal["long", "short", "none"]

# ──────────────────────────────────────────────
# 공통 보조 함수
# ──────────────────────────────────────────────

def _body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()

def _range(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df["low"]

def _upper_wick(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)

def _lower_wick(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]

def _body_ratio(df: pd.DataFrame) -> pd.Series:
    rng = _range(df)
    return _body(df) / rng.replace(0, np.nan)

def _rolling_percentile(series: pd.Series, window: int, pct: float) -> pd.Series:
    """rolling window 내 pct 백분위수 (0~100)"""
    return series.rolling(window).quantile(pct / 100)


# ──────────────────────────────────────────────
# Pattern B (수정) — 꼬리 반향 (Tail Echo) · Long only
# ──────────────────────────────────────────────
# 변경: 고정 배수 2.0 → 롤링 P80 백분위수 기반 (NCP2 요청)
# 꼬리 크기가 최근 window일 기준 P80을 초과할 때만 유효 신호로 판단
# Long only: 위꼬리가 크고 오늘 종가가 전일 고점을 넘을 때

def pattern_tail_echo(
    df: pd.DataFrame,
    pct_threshold: float = 80.0,
    window: int = 60,
) -> pd.Series:
    upper = _upper_wick(df)
    lower = _lower_wick(df)
    body = _body(df)

    upper_p = _rolling_percentile(upper, window, pct_threshold).shift(1)
    lower_p = _rolling_percentile(lower, window, pct_threshold).shift(1)

    prev_upper = upper.shift(1)
    prev_lower = lower.shift(1)
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    prev_body = body.shift(1)

    signals = pd.Series("none", index=df.index)

    # Long: 전일 위꼬리가 P80 초과 + 오늘 종가가 전일 고점 돌파
    long_cond = (
        (prev_upper > upper_p) &
        (prev_body > 0) &
        (df["close"] > prev_high)
    )
    signals[long_cond] = "long"
    return signals


# ──────────────────────────────────────────────
# Pattern D (최종) — 하락 흡수 (Bear Absorption) · Long only
# ──────────────────────────────────────────────
# DualPressure 재설계 과정에서 도출 (NCP1·NCP2 공동)
# 아이디어: 강한 하락을 강한 상승으로 완전 회복 → 매수 세력 우위 확인
#
# 조건:
#   t-2: 강한 하락일 (body_ratio ≥ rolling P75, close < open)
#   t-1: 강한 상승일 (body_ratio ≥ rolling P60, close > open)
#         + close[t-1] > open[t-2]  (이틀치 하락 완전 회복)
# 신호: 오늘 Long 진입
#
# 파라미터: down_pct=75, up_pct=60, window=60

def pattern_bear_absorption(
    df: pd.DataFrame,
    down_pct: float = 75.0,
    up_pct: float = 60.0,
    window: int = 60,
) -> pd.Series:
    br  = _body_ratio(df)
    ret = df["close"] - df["open"]

    br_p_down = _rolling_percentile(br, window, down_pct).shift(2)
    br_p_up   = _rolling_percentile(br, window, up_pct).shift(1)

    is_down  = (ret.shift(2) < 0) & (br.shift(2) >= br_p_down)
    is_up    = (ret.shift(1) > 0) & (br.shift(1) >= br_p_up)
    recovery = df["close"].shift(1) > df["open"].shift(2)

    signals = pd.Series("none", index=df.index)
    signals[is_down & is_up & recovery & br_p_down.notna()] = "long"
    return signals


# ──────────────────────────────────────────────
# Pattern E (NCP2 창안) — 소화율 (Absorption)
# ──────────────────────────────────────────────
# 충격일: 몸통 비율 ≥ P70 AND 범위 ≥ 20일 P70
# 소화일: 충격일 다음날 범위 ≤ 20일 P30, 방향 = 충격일 반대
# 신호:   소화일 다음날 충격일 방향으로 진입
# 근거:   강한 충격 후 시장이 조용히 소화하면 추세 지속 가능성 높음

def pattern_absorption(
    df: pd.DataFrame,
    body_pct: float = 70.0,
    range_high_pct: float = 70.0,
    range_low_pct: float = 30.0,
    window: int = 20,
) -> pd.Series:
    br = _body_ratio(df)
    rng = _range(df)
    ret = df["close"] - df["open"]

    br_p = _rolling_percentile(br, window, body_pct)
    rng_high_p = _rolling_percentile(rng, window, range_high_pct)
    rng_low_p = _rolling_percentile(rng, window, range_low_pct)

    # 충격일 조건
    shock = (br >= br_p) & (rng >= rng_high_p)
    shock_dir = np.sign(ret)

    # 소화일 조건: 전일이 충격일이고, 오늘 범위 좁고 방향 반대
    absorb = (
        shock.shift(1) &
        (rng <= rng_low_p) &
        (np.sign(ret) != shock_dir.shift(1))
    )

    # 신호: 소화일 다음날 → 충격일 방향
    signals = pd.Series("none", index=df.index)
    shock_was_up = absorb & (shock_dir.shift(1) > 0)
    shock_was_down = absorb & (shock_dir.shift(1) < 0)
    signals[shock_was_up.shift(1).fillna(False)] = "long"
    signals[shock_was_down.shift(1).fillna(False)] = "short"
    return signals


# ──────────────────────────────────────────────
# Pattern F (NCP2 창안) — 꼬리 비대칭 누적 (Tail Asymmetry)
# ──────────────────────────────────────────────
# 5일 누적 위꼬리합 / 아래꼬리합 비율 계산
# 비율 ≥ 롤링 P85 → 매도 압력 우세 → Short
# 비율 ≤ 롤링 P15 → 매수 압력 우세 → Long
# 근거: 절대 변동성 무관, 비율 기반이라 범용성 높음

def pattern_tail_asymmetry(
    df: pd.DataFrame,
    accum_days: int = 5,
    high_pct: float = 85.0,
    low_pct: float = 15.0,
    window: int = 60,
) -> pd.Series:
    upper = _upper_wick(df)
    lower = _lower_wick(df)

    upper_sum = upper.rolling(accum_days).sum()
    lower_sum = lower.rolling(accum_days).sum()
    ratio = upper_sum / lower_sum.replace(0, np.nan)

    ratio_high_p = _rolling_percentile(ratio, window, high_pct)
    ratio_low_p = _rolling_percentile(ratio, window, low_pct)

    signals = pd.Series("none", index=df.index)
    signals[ratio >= ratio_high_p] = "short"
    signals[ratio <= ratio_low_p] = "long"
    return signals


# ──────────────────────────────────────────────
# 최종 패턴 후보 (4종)
# ──────────────────────────────────────────────
PATTERNS = {
    "TailEcho":        pattern_tail_echo,
    "BearAbsorption":  pattern_bear_absorption,
    "Absorption":      pattern_absorption,
    "TailAsymmetry":   pattern_tail_asymmetry,
}

# 최종 확정 패턴 (백테스트 통과)
FINAL_PATTERNS = {
    "TailEcho":       pattern_tail_echo,
    "BearAbsorption": pattern_bear_absorption,
}


def detect_all(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for name, fn in PATTERNS.items():
        result[name] = fn(df)
    return result


def signal_stats(df: pd.DataFrame, signals: pd.Series, hold_days: int = 3) -> dict:
    fwd_ret = df["close"].shift(-hold_days) / df["close"] - 1
    long_mask = signals == "long"
    short_mask = signals == "short"
    long_ret = fwd_ret[long_mask]
    short_ret = -fwd_ret[short_mask]
    all_ret = pd.concat([long_ret, short_ret])

    return {
        "신호 수 (long)":    int(long_mask.sum()),
        "신호 수 (short)":   int(short_mask.sum()),
        "평균 수익 (long)":  f"{long_ret.mean()*100:.2f}%" if len(long_ret) > 0 else "N/A",
        "평균 수익 (short)": f"{short_ret.mean()*100:.2f}%" if len(short_ret) > 0 else "N/A",
        "승률":              f"{(all_ret > 0).mean()*100:.1f}%" if len(all_ret) > 0 else "N/A",
        "평균 수익 (전체)":  f"{all_ret.mean()*100:.2f}%" if len(all_ret) > 0 else "N/A",
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from eda import load_data

    df = load_data()
    signals_df = detect_all(df)

    print("=" * 60)
    print("최종 패턴 후보 신호 통계 (3일 보유)")
    print("=" * 60)
    for name in PATTERNS:
        print(f"\n[{name}]")
        stats = signal_stats(df, signals_df[name], hold_days=3)
        for k, v in stats.items():
            print(f"  {k}: {v}")
