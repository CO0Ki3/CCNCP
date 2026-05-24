import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal

SignalType = Literal["long", "short", "none"]


@dataclass
class PatternSignal:
    date: pd.Timestamp
    pattern: str
    signal: SignalType
    strength: float  # 0~1, 신호 강도


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


def _log_return(df: pd.DataFrame) -> pd.Series:
    return np.log(df["close"] / df["close"].shift(1))


def _rolling_std(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).std()


# ──────────────────────────────────────────────
# Pattern A — 탄성 수렴 (Elastic Convergence)
# ──────────────────────────────────────────────
# 아이디어: 시장 참여자 의사결정이 점점 좁아지다가 방향 결정하는 순간 포착
# 조건:
#   - 직전 N일 연속으로 캔들 몸통이 감소 (squeeze)
#   - 오늘 몸통이 직전 N일 평균보다 크게 폭발 (burst)
# 파라미터: squeeze_days=4, burst_ratio=1.0

def pattern_elastic_convergence(
    df: pd.DataFrame,
    squeeze_days: int = 4,
    burst_ratio: float = 1.0,
) -> pd.Series:
    body = _body(df)
    signals = pd.Series(SignalType.__args__[2], index=df.index)  # "none"

    for i in range(squeeze_days, len(df)):
        window_bodies = body.iloc[i - squeeze_days: i]

        # 연속 감소 확인
        is_squeeze = all(
            window_bodies.iloc[j] > window_bodies.iloc[j + 1]
            for j in range(squeeze_days - 1)
        )
        if not is_squeeze:
            continue

        avg_body = window_bodies.mean()
        today_body = body.iloc[i]
        if avg_body == 0:
            continue

        # 폭발 조건
        if today_body > avg_body * burst_ratio:
            direction = df["close"].iloc[i] > df["open"].iloc[i]
            signals.iloc[i] = "long" if direction else "short"

    return signals


# ──────────────────────────────────────────────
# Pattern B — 꼬리 반향 (Tail Echo)
# ──────────────────────────────────────────────
# 아이디어: 전일 시장이 한 방향을 강하게 테스트하고 거부당한 뒤,
#           다음날 그 방향을 돌파하면 강한 모멘텀 신호
# 조건:
#   - 전일 한쪽 꼬리 > 전일 몸통 × tail_ratio
#   - 오늘 종가가 전일 꼬리 끝(고점/저점)을 넘어섬
# 파라미터: tail_ratio=2.0

def pattern_tail_echo(
    df: pd.DataFrame,
    tail_ratio: float = 2.0,
) -> pd.Series:
    body = _body(df)
    upper = _upper_wick(df)
    lower = _lower_wick(df)
    signals = pd.Series("none", index=df.index)

    prev_body = body.shift(1)
    prev_upper = upper.shift(1)
    prev_lower = lower.shift(1)
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)

    # 위꼬리 돌파: 전일 위꼬리가 컸고, 오늘 종가 > 전일 고점
    long_cond = (
        (prev_upper > prev_body * tail_ratio) &
        (prev_body > 0) &
        (df["close"] > prev_high)
    )
    # 아래꼬리 돌파: 전일 아래꼬리가 컸고, 오늘 종가 < 전일 저점
    short_cond = (
        (prev_lower > prev_body * tail_ratio) &
        (prev_body > 0) &
        (df["close"] < prev_low)
    )

    signals[long_cond] = "long"
    signals[short_cond] = "short"
    return signals


# ──────────────────────────────────────────────
# Pattern C — 변동성 단층 (Volatility Fault)
# ──────────────────────────────────────────────
# 아이디어: 첨도 17.2에서 착안. 극단적 범위의 캔들 이후 mean-reversion 경향
# 조건:
#   - 오늘 (high-low) > 직전 lookback일 평균 범위 × range_mult
#   - 다음날 open 기준 단층 방향 반대로 진입
# 파라미터: lookback=10, range_mult=2.5

def pattern_volatility_fault(
    df: pd.DataFrame,
    lookback: int = 10,
    range_mult: float = 2.5,
) -> pd.Series:
    rng = _range(df)
    avg_range = rng.shift(1).rolling(lookback).mean()
    signals = pd.Series("none", index=df.index)

    # 단층 발생 조건 (오늘)
    fault_up = (rng > avg_range * range_mult) & (df["close"] > df["open"])
    fault_down = (rng > avg_range * range_mult) & (df["close"] <= df["open"])

    # 신호는 다음날에 발생 (mean reversion)
    signals[fault_up.shift(1).fillna(False)] = "short"
    signals[fault_down.shift(1).fillna(False)] = "long"
    return signals


# ──────────────────────────────────────────────
# Pattern D — 이중 압력 (Dual Pressure)
# ──────────────────────────────────────────────
# 아이디어: 매수/매도 세력이 이틀 연속 팽팽하게 맞붙은 뒤 승자 방향 추종
# 조건:
#   - 2일 연속 반대 방향 움직임
#   - 각각 |수익률| > 일 표준편차(20일) × threshold
#   - 3번째 날 = 2번째 날 방향으로 진입
# 파라미터: threshold=1.5, std_window=20

def pattern_dual_pressure(
    df: pd.DataFrame,
    threshold: float = 1.5,
    std_window: int = 20,
) -> pd.Series:
    ret = _log_return(df)
    vol = _rolling_std(ret, std_window)
    signals = pd.Series("none", index=df.index)

    prev1_ret = ret.shift(1)
    prev2_ret = ret.shift(2)
    prev1_vol = vol.shift(1)

    big_move = (ret.shift(1).abs() > vol.shift(1) * threshold) & \
               (ret.shift(2).abs() > vol.shift(2) * threshold)
    opposite = np.sign(prev1_ret) != np.sign(prev2_ret)
    valid = big_move & opposite & prev1_vol.notna()

    long_cond = valid & (prev1_ret > 0)
    short_cond = valid & (prev1_ret < 0)

    signals[long_cond] = "long"
    signals[short_cond] = "short"
    return signals


# ──────────────────────────────────────────────
# 통합 실행
# ──────────────────────────────────────────────

PATTERNS = {
    "ElasticConvergence": pattern_elastic_convergence,
    "TailEcho": pattern_tail_echo,
    "VolatilityFault": pattern_volatility_fault,
    "DualPressure": pattern_dual_pressure,
}


def detect_all(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for name, fn in PATTERNS.items():
        result[name] = fn(df)
    return result


def signal_stats(df: pd.DataFrame, signals: pd.Series, hold_days: int = 3) -> dict:
    """패턴 신호 발생 후 hold_days일 수익률 통계"""
    fwd_ret = df["close"].shift(-hold_days) / df["close"] - 1

    long_mask = signals == "long"
    short_mask = signals == "short"

    long_ret = fwd_ret[long_mask]
    short_ret = -fwd_ret[short_mask]
    all_ret = pd.concat([long_ret, short_ret])

    return {
        "신호 수 (long)": int(long_mask.sum()),
        "신호 수 (short)": int(short_mask.sum()),
        "평균 수익 (long)": f"{long_ret.mean()*100:.2f}%" if len(long_ret) > 0 else "N/A",
        "평균 수익 (short)": f"{short_ret.mean()*100:.2f}%" if len(short_ret) > 0 else "N/A",
        "승률": f"{(all_ret > 0).mean()*100:.1f}%" if len(all_ret) > 0 else "N/A",
        "평균 수익 (전체)": f"{all_ret.mean()*100:.2f}%" if len(all_ret) > 0 else "N/A",
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from eda import load_data

    df = load_data()
    signals_df = detect_all(df)

    print("=" * 60)
    print("패턴 신호 통계 (3일 보유 기준)")
    print("=" * 60)
    for name in PATTERNS:
        print(f"\n[{name}]")
        stats = signal_stats(df, signals_df[name], hold_days=3)
        for k, v in stats.items():
            print(f"  {k}: {v}")
